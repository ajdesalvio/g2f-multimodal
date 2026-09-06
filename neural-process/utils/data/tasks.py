"""Task-level data layer for the Transformer Neural Process.

A *task* is a context set + a query set drawn from the training-visible pool.
This module builds the canonical ``[B, n, ...]`` task batches the
:class:`~set_func.models.neural_process.TransformerNeuralProcess` consumes:

- :class:`NPTaskBatch` — ``{context: G2FBatch, query: G2FBatch}``, each carrying a
  leading ``(task, sample)`` pair.
- :class:`PrecomputedPool` — assembles + pads every train row's per-view
  ``(coords, channels)`` ONCE; a task is then built by a cheap tensor
  ``index_select`` (no per-step ``assemble_view`` / padding). This is the fast
  path the :class:`TaskSampler` uses.
- :func:`np_collate` — the raw-dict collate: collates a list of ``B`` task specs
  into an :class:`NPTaskBatch`. It is the correctness **oracle** for the pool
  gather (a slow, obviously-correct reference path); the :class:`TaskSampler` does
  NOT use it — training always runs on the :class:`PrecomputedPool` fast path,
  augment included.
- :class:`TaskSampler` — an ``IterableDataset`` that yields ready
  :class:`NPTaskBatch` es with the role firewall (never draws ``Role.PREDICT``)
  and the shared-per-batch ``nc``/``nq`` size schedule. Worker-shardable
  (``num_workers > 0`` is safe — each worker takes a disjoint slice of the batches
  with its own RNG stream), so the dataloader can prefetch the next batch while the
  GPU is busy.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import einops
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .base import BaseBatch, G2FBatch, ViewBatch
from .dataset import g2f_collate_fn

logger = logging.getLogger(__name__)

# Roles the model is allowed to observe as context/query (the CV firewall:
# Role.PREDICT rows are never drawn in training). Stored as plain strings so the
# sampler stays decoupled from the cv.schemes import at module load.
_TRAINABLE_ROLES = ("FIT", "OBSERVE")

# Per-batch context-size (nc) sampling distributions over [nc_min, nc_max].
#   "uniform"     : nc ~ U{nc_min, nc_max} (flat; with nc_min<<nc_max the linear
#                   bulk already favours large contexts).
#   "log_uniform" : nc ~ round(exp(U(ln nc_min, ln nc_max))) — density ∝ 1/nc, so
#                   it spreads draws evenly across orders of magnitude and sees
#                   SMALL contexts more often than uniform. Use it to train one
#                   model across a wide nc range (16 → whole pool) rather than
#                   pinning nc near the test-time full-pool size.
_NC_SAMPLERS = ("uniform", "log_uniform")


@dataclass
class NPTaskBatch(BaseBatch):
    """A batch of ``B`` tasks: a context ``G2FBatch`` + a query ``G2FBatch``.

    Both sides carry the canonical task structure with a leading
    ``(task, sample)`` pair (``views[v].coords [B, n, T, D]``, ``s [B, n]``, …).
    ``B``/``nc``/``nq`` are read from ``.shape`` — there are no separate size
    fields and no task-index mask (tasks are isolated by the batch dim).
    """

    context: G2FBatch
    query: G2FBatch

    def to(self, device: torch.device) -> "NPTaskBatch":
        self.context = self.context.to(device)
        self.query = self.query.to(device)
        return self

    @property
    def s(self) -> torch.Tensor:
        """Query targets shaped ``[B, nq, 1]`` for the loss / metrics.

        Matches the decoder output ``[B, nq, 1]`` elementwise (so
        ``log_prob`` / ``mse_loss`` broadcast cleanly), and ``reshape(-1)``
        yields the task-major ``[B*nq]`` vector the pooled eval metrics use —
        the same ordering as ``output.mean.reshape(-1)``.
        """
        s = self.query.s
        # query.s is [B, nq] (scalar yield) or already [B, nq, 1] (collated).
        if s.ndim == 2:
            return einops.rearrange(s, "b n -> b n 1")
        return s


def _add_task_dim(batch: G2FBatch, num_tasks: int, n: int) -> G2FBatch:
    """Lift a flat ``[B*n, ...]`` ``G2FBatch`` to canonical ``[B, n, ...]``.

    Every field's leading axis is split ``(b n) -> b n`` with ``einops`` (the
    collate emits the rows task-major, so the split groups each task's samples).
    """

    def _split(t: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(t, "(b n) ... -> b n ...", b=num_tasks, n=n)

    views: dict[str, ViewBatch] = {}
    for name, v in batch.views.items():
        pad_mask = _split(v.pad_mask)  # [B, n, T]
        vb = ViewBatch(
            coords=_split(v.coords),
            channels=_split(v.channels),
            pad_mask=pad_mask,
            coord_names=v.coord_names,
            channel_names=v.channel_names,
        )
        # ViewBatch.__post_init__ squeezes a trailing size-1 axis, which would
        # corrupt a degenerate T==1 pad mask ([B, n, 1] -> [B, n]); reassign the
        # 3-D mask so the canonical [B, n, T] shape is always preserved.
        vb.pad_mask = pad_mask
        views[name] = vb

    derived = {k: _split(t) for k, t in batch.derived_features.items()}
    return G2FBatch(views=views, s=_split(batch.s), derived_features=derived)


def np_collate(
    tasks: Sequence[tuple[list[dict], list[dict]]],
    *,
    views=None,
    pad_value: float = 0.0,
    augment=None,
) -> NPTaskBatch:
    """Collate ``B`` task specs into a canonical :class:`NPTaskBatch`.

    The raw-dict path: assembles + pads per step. The training sampler always
    uses the :class:`PrecomputedPool` fast path instead (augment included —
    applied inside :meth:`PrecomputedPool.gather` via ``_apply_timepoint_dropout``,
    NOT here); this function is the correctness oracle for that gather.

    Args:
        tasks: ``B`` specs, each ``(context_samples[nc], query_samples[nq])``
            (lists of sample dicts). All tasks share the same ``nc``/``nq``
            (the sampler draws them once per batch).
        views: DL-target ``View`` specs forwarded to ``g2f_collate_fn`` (or
            ``None`` for the legacy ``main`` view).
        pad_value: Padding value for variable-length curves.
        augment: optional per-modality subsampling config; applied
            **before** padding, train-only (this collate is the train path).
            ``None`` or all-``[1.0, 1.0]`` ranges => no-op.

    Returns:
        An :class:`NPTaskBatch` with context ``[B, nc, …]`` and query
        ``[B, nq, …]``.
    """
    if not tasks:
        raise ValueError("np_collate received an empty task list")

    if augment is not None:
        from .augment import augment_tasks

        tasks = augment_tasks(tasks, augment)

    num_tasks = len(tasks)
    nc = len(tasks[0][0])
    nq = len(tasks[0][1])
    for ctx, qry in tasks:
        if len(ctx) != nc or len(qry) != nq:
            raise ValueError(
                "all tasks in a batch must share nc/nq "
                f"(expected nc={nc}, nq={nq})"
            )

    # Task-major flattening: task 0's nc samples, task 1's nc samples, …  so the
    # later `(b n) -> b n` split regroups each task's samples correctly.
    flat_ctx = [s for ctx, _ in tasks for s in ctx]
    flat_qry = [s for _, qry in tasks for s in qry]

    context = _add_task_dim(
        g2f_collate_fn(flat_ctx, pad_value, views), num_tasks, nc
    )
    query = _add_task_dim(
        g2f_collate_fn(flat_qry, pad_value, views), num_tasks, nq
    )
    return NPTaskBatch(context=context, query=query)


@dataclass
class _PoolView:
    """One view's pool-wide assembled tensors (padded to the global max ``T``)."""

    coords: torch.Tensor      # [N, Tg, D]
    channels: torch.Tensor    # [N, Tg, C]
    lengths: torch.Tensor     # [N]  true (pre-pad) length per row
    coord_names: list[str] | None
    channel_names: list[str] | None


def _stack_padded(
    coords_list: list[torch.Tensor],
    chan_list: list[torch.Tensor],
    pad_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad per-row ``(coords[T,D], channels[T,C])`` to the global max ``T``.

    Returns ``(coords[N,Tg,D], channels[N,Tg,C], lengths[N])``. Rows are stored
    full-length once; :meth:`PrecomputedPool.gather` slices each drawn group back
    to its own max true length, so padded positions never reach the encoder.
    """
    n = len(coords_list)
    lengths = torch.tensor([c.shape[0] for c in coords_list], dtype=torch.long)
    tg = int(lengths.max())
    d = coords_list[0].shape[1] if coords_list[0].ndim == 2 else 1
    c = chan_list[0].shape[1] if chan_list[0].ndim == 2 else 1
    coords = torch.full((n, tg, d), float(pad_value), dtype=torch.float32)
    channels = torch.full((n, tg, c), float(pad_value), dtype=torch.float32)
    for i, (co, ch) in enumerate(zip(coords_list, chan_list)):
        if co.ndim == 1:
            co = co.unsqueeze(-1)
        if ch.ndim == 1:
            ch = ch.unsqueeze(-1)
        t = co.shape[0]
        coords[i, :t] = co.to(torch.float32)
        channels[i, :t] = ch.to(torch.float32)
    return coords, channels, lengths


# Augment modality -> view name (standard dl_views: VI="main", weather="weather").
_AUG_MODALITY_TO_VIEW = {"vi": "main", "weather": "weather"}


def _active_aug_views(augment) -> dict[str, tuple[float, float, int]]:
    """Map each ACTIVE augment modality to ``view_name -> (lo, hi, min_kept)``.

    A modality with ``keep_frac_range == [1.0, 1.0]`` is a no-op and omitted, so
    an empty result means "no augmentation" (the pure fast path).
    """
    if augment is None:
        return {}
    from .augment import _MODALITY_FIELDS, _resolve_modality

    out: dict[str, tuple[float, float, int]] = {}
    for mod in _MODALITY_FIELDS:
        lo, hi, mk = _resolve_modality(augment, mod)
        if not (lo == 1.0 and hi == 1.0):
            out[_AUG_MODALITY_TO_VIEW[mod]] = (lo, hi, mk)
    return out


def _apply_timepoint_dropout(pad_flat: torch.Tensor, spec: dict, n: int) -> torch.Tensor:
    """Mask a random ``1 - rate`` fraction of each curve's real timepoints.

    ``pad_flat`` is ``[G*n, L]`` (True = padding); ``spec`` carries the per-task
    ``rate[G]`` (repeated across the task's ``n`` samples), ``min_kept``, and an
    RNG. Returns ``pad_flat | drop``, always retaining ``>= min_kept`` real
    timepoints per sample. Vectorised — no per-sample Python loop.
    """
    gn, length = pad_flat.shape
    valid = ~pad_flat
    rate = np.repeat(np.asarray(spec["rate"], dtype=np.float32), n)   # [G*n]
    p = torch.from_numpy(rate).unsqueeze(1)                           # [G*n, 1]
    r = torch.from_numpy(spec["rng"].random((gn, length)).astype(np.float32))
    keep = (r < p) & valid
    mk = int(spec["min_kept"])
    if mk > 0:
        # Always retain the min_kept smallest-r real timepoints (invalid rank last).
        r_rank = torch.where(valid, r, torch.full_like(r, float("inf")))
        topk = r_rank.argsort(dim=1)[:, :mk]
        forced = torch.zeros_like(keep)
        forced.scatter_(1, topk, True)
        keep = keep | (forced & valid)
    drop = valid & ~keep
    return pad_flat | drop


@dataclass
class PrecomputedPool:
    """Assembled + padded train pool for O(1)-per-step task construction.

    The TNP train pool is fixed for the whole run, yet the raw collate re-ran
    ``assemble_view`` + padding on the *same* rows every step — the dominant CPU
    cost, which starved the GPU. This materialises every row's per-view
    ``(coords, channels)`` ONCE (padded to the per-view global max ``T``), so a
    task is built by a cheap tensor ``index_select`` instead of thousands of
    per-sample Python calls.

    Scalars ``s[N]`` and fixed-size ``derived[key][N, F]`` ride alongside.
    :meth:`gather` slices each gathered group back to its own max true length, so
    the result is byte-equivalent (up to padded positions) to :func:`np_collate`.
    """

    views: dict[str, _PoolView]
    s: torch.Tensor
    derived: dict[str, torch.Tensor]
    n: int

    @classmethod
    def from_rows(cls, rows, views=None, pad_value: float = 0.0) -> "PrecomputedPool":
        n = len(rows)
        if n == 0:
            raise ValueError("PrecomputedPool: rows is empty.")

        out_views: dict[str, _PoolView] = {}
        if views:
            from .views import assemble_view

            for view in views:
                assembled = [assemble_view(r, view) for r in rows]
                coords, channels, lengths = _stack_padded(
                    [a.coords for a in assembled],
                    [a.channels for a in assembled],
                    pad_value,
                )
                out_views[view.name] = _PoolView(
                    coords=coords,
                    channels=channels,
                    lengths=lengths,
                    coord_names=list(assembled[0].coord_names),
                    channel_names=list(assembled[0].channel_names),
                )
        else:
            # Legacy default: a single "main" view from DAP coords + VI channels.
            coords, channels, lengths = _stack_padded(
                [r["dap"] for r in rows],
                [r["channels"] for r in rows],
                pad_value,
            )
            out_views["main"] = _PoolView(
                coords=coords,
                channels=channels,
                lengths=lengths,
                coord_names=["dap"],
                channel_names=rows[0].get("channel_names"),
            )

        s = torch.stack([r["yield_value"] for r in rows])
        derived: dict[str, torch.Tensor] = {}
        if "derived_features" in rows[0]:
            for key in rows[0]["derived_features"]:
                derived[key] = torch.stack(
                    [r["derived_features"][key] for r in rows]
                )
        return cls(views=out_views, s=s, derived=derived, n=n)

    def gather(self, idx: np.ndarray, *, aug: dict | None = None) -> G2FBatch:
        """Build a canonical ``[G, n, …]`` ``G2FBatch`` by indexing the pool.

        ``idx`` is a ``[G, n]`` integer array of pool row positions (task-major:
        row ``g``'s ``n`` samples). Each view is sliced to the gathered group's
        own max true length and given a fresh pad mask.

        ``aug`` optionally applies train-time per-modality timepoint subsampling
        as **extra padding**: ``aug[view_name] = {"rate": np.ndarray[G],
        "min_kept": int, "rng": Generator}`` drops, per task, a random
        ``1 - rate`` fraction of each curve's real timepoints by masking them.
        For a set encoder masking ≡ removing (masked elements contribute nothing
        to attention/PMA), so this achieves timepoint dropout without a
        per-sample Python loop. NOTE: each real timepoint is kept independently
        with probability ``rate`` (Bernoulli), so the kept COUNT is random
        (~``Binomial(T, rate)``) — this is *equivalent in expectation* to, but
        not the same subset distribution as, ``augment.py``'s fixed-count
        ``round(rate·T)`` subsample used on the per-sample path. ``min_kept``
        real timepoints are always retained.
        """
        idx = np.asarray(idx)
        g, n = idx.shape
        flat = torch.as_tensor(idx.reshape(-1), dtype=torch.long)

        out_views: dict[str, ViewBatch] = {}
        for name, V in self.views.items():
            lengths = V.lengths.index_select(0, flat)  # [G*n]
            maxlen = int(lengths.max())
            coords = V.coords.index_select(0, flat)[:, :maxlen]    # [G*n, L, D]
            channels = V.channels.index_select(0, flat)[:, :maxlen]
            pad_flat = (
                torch.arange(maxlen).unsqueeze(0) >= lengths.unsqueeze(1)
            )  # [G*n, L], True = padding
            if aug is not None and name in aug:
                pad_flat = _apply_timepoint_dropout(pad_flat, aug[name], n)
            coords = coords.reshape(g, n, maxlen, coords.shape[-1])
            channels = channels.reshape(g, n, maxlen, channels.shape[-1])
            pad_mask = pad_flat.reshape(g, n, maxlen)
            vb = ViewBatch(
                coords=coords,
                channels=channels,
                pad_mask=pad_mask,
                coord_names=V.coord_names,
                channel_names=V.channel_names,
            )
            # Guard the T==1 squeeze (see _add_task_dim): keep the 3-D mask.
            vb.pad_mask = pad_mask
            out_views[name] = vb

        s = self.s.index_select(0, flat).reshape(g, n, *self.s.shape[1:])
        derived = {
            k: t.index_select(0, flat).reshape(g, n, *t.shape[1:])
            for k, t in self.derived.items()
        }
        return G2FBatch(views=out_views, s=s, derived_features=derived)


def _identity_collate(batch):
    """Pass-through collate: the sampler already yields a ready ``NPTaskBatch``."""
    return batch


class TaskSampler(IterableDataset):
    """Yield ready :class:`NPTaskBatch` es from the training-visible pool.

    Each ``__iter__`` produces ``batches_per_epoch`` batches; each batch holds
    ``tasks_per_batch`` tasks that **share one ``nc`` and one ``nq``** (drawn once
    per batch). Pair with ``DataLoader(sampler, batch_size=None,
    collate_fn=_identity_collate)`` — ``batch_size=None`` disables auto-batching so
    each yielded :class:`NPTaskBatch` reaches the model intact.

    Fast path: a :class:`PrecomputedPool` is built once and each task is a cheap
    ``index_select`` — for BOTH the augment and no-augment cases. Per-modality
    timepoint subsampling (train-only) is applied inside :meth:`PrecomputedPool.gather`
    as extra mask padding (``_apply_timepoint_dropout``), so augment never leaves
    the fast path. (:func:`np_collate` is only the reference
    collate — the sampler never calls it.)

    Role firewall: only rows with role in ``{FIT, OBSERVE}`` are drawable
    (``Role.PREDICT`` is never drawn). If ``roles`` is given the firewall is
    enforced explicitly; otherwise the caller is trusted to pass the train pool.

    Size schedule: per batch ``nc`` is drawn from ``{nc_min..nc_max}``
    (discrete uniform by default; ``nc_sampler="log_uniform"`` draws
    log-uniformly) and ``nq ~ U{nq_min, nq_max}``, shared across the ``B`` tasks, subject to the
    joint ``nc + nq <= N`` constraint (redrawn if violated). Each task then draws
    ``nc + nq`` distinct rows and splits them into disjoint context/query.

    Worker sharding: under ``num_workers > 0`` each worker takes a disjoint slice
    of the ``batches_per_epoch`` batches (``batch_index % num_workers``) with its
    own RNG stream seeded from ``(seed, worker_id, epoch)`` — so workers never
    produce duplicate batches and the loader can prefetch while the GPU runs.
    """

    def __init__(
        self,
        pool: Sequence[dict],
        *,
        views=None,
        pad_value: float = 0.0,
        augment=None,
        tasks_per_batch: int = 1,
        batches_per_epoch: int = 100,
        nc_min: int | None = None,
        nc_max: int | None = None,
        nq_min: int | None = None,
        nq_max: int | None = None,
        nc_sampler: str = "uniform",
        seed: int = 0,
        roles: Sequence | None = None,
    ):
        super().__init__()
        self.rows = pool
        if nc_sampler not in _NC_SAMPLERS:
            raise ValueError(
                f"nc_sampler must be one of {_NC_SAMPLERS}; got {nc_sampler!r}"
            )
        self._nc_sampler = nc_sampler
        self._views = views
        self._pad_value = pad_value

        if roles is not None:
            if len(roles) != len(pool):
                raise ValueError(
                    f"roles length {len(roles)} != pool length {len(pool)}"
                )
            drawable = [
                i
                for i, r in enumerate(roles)
                if str(getattr(r, "value", r)) in _TRAINABLE_ROLES
            ]
        else:
            drawable = list(range(len(pool)))

        self._drawable = np.asarray(drawable, dtype=int)
        n = len(self._drawable)
        if n < 2:
            raise ValueError(
                f"TaskSampler needs >= 2 drawable (FIT/OBSERVE) rows, got {n}"
            )

        self.tasks_per_batch = tasks_per_batch
        self.batches_per_epoch = batches_per_epoch
        self.nc_min = 1 if nc_min is None else nc_min
        self.nq_min = 1 if nq_min is None else nq_min
        # nc_max / nq_max are CEILINGS: an unset bound, or one larger than the
        # pool allows, clamps to N-1 (each side must leave >= 1 for the other).
        # This makes a fixed config ceiling (e.g. nc_max=2048) degrade gracefully
        # to "the whole pool" on a smaller CV fold instead of erroring.
        self.nc_max = (n - 1) if nc_max is None else min(nc_max, n - 1)
        self.nq_max = (n - 1) if nq_max is None else min(nq_max, n - 1)
        if nc_max is not None and nc_max > n - 1:
            logger.info(
                "TaskSampler: nc_max=%d exceeds pool N-1=%d; clamped to %d.",
                nc_max, n - 1, self.nc_max,
            )
        if nq_max is not None and nq_max > n - 1:
            logger.info(
                "TaskSampler: nq_max=%d exceeds pool N-1=%d; clamped to %d.",
                nq_max, n - 1, self.nq_max,
            )
        self._validate_bounds(n)

        self._n = n
        self._seed = int(seed)
        self._epoch = 0

        # The (fixed) pool is materialised once, lazily on first iteration (so
        # constructing a sampler — or just drawing indices — never requires
        # assembling sample contents). Active augmentation is applied IN the pool
        # gather as vectorised timepoint masking, so it stays on the fast path
        # too; `_aug_views` is empty for the no-op default (pure gather).
        self._aug_views = _active_aug_views(augment)
        self._pool: PrecomputedPool | None = None

    def _validate_bounds(self, n: int) -> None:
        # Maxes are already clamped to <= N-1; only genuine misconfig raises
        # (min below 1, min above the clamped max, or a joint floor exceeding N).
        for name, lo, hi in (
            ("nc", self.nc_min, self.nc_max),
            ("nq", self.nq_min, self.nq_max),
        ):
            if not (1 <= lo <= hi):
                raise ValueError(
                    f"{name} bounds must satisfy 1 <= {name}_min <= {name}_max "
                    f"(after clamping to N-1={n - 1}); got [{lo}, {hi}]"
                )
        if self.nc_min + self.nq_min > n:
            raise ValueError(
                f"nc_min + nq_min = {self.nc_min + self.nq_min} > N={n}: "
                "no batch can satisfy the joint constraint"
            )

    def _draw_nc(self, rng) -> int:
        """Draw one context size in ``[nc_min, nc_max]`` per ``self._nc_sampler``."""
        if self._nc_sampler == "log_uniform":
            u = rng.uniform(math.log(self.nc_min), math.log(self.nc_max))
            return min(self.nc_max, max(self.nc_min, int(round(math.exp(u)))))
        return int(rng.integers(self.nc_min, self.nc_max + 1))

    def _draw_sizes(self, rng) -> tuple[int, int]:
        """Draw ``(nc, nq)`` shared across the batch, respecting ``nc+nq<=N``."""
        while True:
            nc = self._draw_nc(rng)
            nq = int(rng.integers(self.nq_min, self.nq_max + 1))
            if nc + nq <= self._n:
                return nc, nq

    def _draw_indices(self, rng, nc: int, nq: int):
        """Draw the batch's ``[B, nc]`` context + ``[B, nq]`` query pool indices."""
        b = self.tasks_per_batch
        ctx = np.empty((b, nc), dtype=np.int64)
        qry = np.empty((b, nq), dtype=np.int64)
        for t in range(b):
            chosen = rng.choice(self._drawable, size=nc + nq, replace=False)
            ctx[t] = chosen[:nc]
            qry[t] = chosen[nc:]
        return ctx, qry

    def __len__(self) -> int:
        return self.batches_per_epoch

    def iter_batch_indices(self):
        """Yield each batch's ``(ctx_idx[B,nc], qry_idx[B,nq])`` pool indices.

        The sampling core — role firewall, shared-per-batch ``nc``/``nq``, and
        worker sharding — with no materialisation, so drawing indices stays cheap
        and isolated. Under ``num_workers > 0`` each worker takes the ``bidx %
        num_workers == worker_id`` slice with its own ``(seed, worker, epoch)``
        RNG stream, so the union over workers is the full, duplicate-free epoch.
        """
        info = get_worker_info()
        wid = 0 if info is None else info.id
        nworkers = 1 if info is None else info.num_workers
        rng = np.random.default_rng([self._seed, wid, self._epoch])
        self._epoch += 1
        for bidx in range(self.batches_per_epoch):
            if bidx % nworkers != wid:
                continue
            nc, nq = self._draw_sizes(rng)
            yield self._draw_indices(rng, nc, nq)

    def _ensure_pool(self) -> None:
        if self._pool is None:
            self._pool = PrecomputedPool.from_rows(
                self.rows, self._views, self._pad_value
            )

    def _draw_aug(self, aug_rng) -> dict | None:
        """Per-batch augment spec: one keep-rate per task per active modality,
        shared across the task's samples AND its context/query."""
        if not self._aug_views:
            return None
        b = self.tasks_per_batch
        return {
            view: {
                "rate": aug_rng.uniform(lo, hi, size=b),
                "min_kept": mk,
                "rng": aug_rng,
            }
            for view, (lo, hi, mk) in self._aug_views.items()
        }

    def __iter__(self):
        self._ensure_pool()
        info = get_worker_info()
        wid = 0 if info is None else info.id
        # Independent stream for augment draws (same (seed, worker, epoch) basis
        # as the index stream, distinct salt). Snapshot epoch BEFORE
        # iter_batch_indices advances it, so both use the same epoch.
        aug_rng = (
            np.random.default_rng([self._seed, wid, self._epoch, 0xA06])
            if self._aug_views
            else None
        )
        for ctx_idx, qry_idx in self.iter_batch_indices():
            # Same rate per task for context and query; per-curve drops differ.
            aug = self._draw_aug(aug_rng)
            yield NPTaskBatch(
                context=self._pool.gather(ctx_idx, aug=aug),
                query=self._pool.gather(qry_idx, aug=aug),
            )


class ContextProvider:
    """Materialize the eval-time context set.

    On the val/test path the per-stream dataloaders yield **query-only** batches;
    the context the model conditions on is supplied here. By default the context
    is the **train split** (the post-carve ``FIT``/``OBSERVE`` pool — the same
    rows the train ``TaskSampler`` draws from); ``eval_context`` overrides it with
    a named eval stream.

    The materialized context is one flat ``G2FBatch`` of ``Nc`` samples (built via
    ``g2f_collate_fn``); :func:`np_forward` lifts it to ``[1, Nc, …]`` and pairs it
    with each query batch. ``context ∩ query = ∅`` is asserted by sample-object
    identity (rows are shared dict objects across splits/streams, so identity is a
    real overlap test) — true by construction for the carved ``val`` and the
    ``Role.PREDICT`` ``test``.
    """

    def __init__(self, dataset, eval_context: str = "train", pad_value: float = 0.0):
        self.dataset = dataset
        self.eval_context = eval_context
        self.pad_value = pad_value
        self._rows: list[dict] | None = None

    def _resolve_rows(self) -> list[dict]:
        spec = self.eval_context
        if spec is None or spec == "train":
            split = self.dataset.get_split("train")
        elif isinstance(spec, str):
            # A named eval stream (or base split) the dataset can resolve.
            split = self.dataset.get_split(spec)
        else:
            raise NotImplementedError(
                f"Unsupported eval_context spec {spec!r}; expected 'train' or a "
                "named eval-stream/split string."
            )
        return [split[i] for i in range(len(split))]

    def context_rows(self) -> list[dict]:
        if self._rows is None:
            self._rows = self._resolve_rows()
        return self._rows

    @staticmethod
    def _assert_disjoint(context_rows: list[dict], query_rows: list[dict]) -> None:
        ctx_ids = {id(r) for r in context_rows}
        n_overlap = sum(1 for r in query_rows if id(r) in ctx_ids)
        if n_overlap:
            raise AssertionError(
                f"context ∩ query is non-empty ({n_overlap} row(s)) — an "
                "out-of-sample eval stream overlaps the eval context, so its "
                "metric would silently degrade from generalization to "
                "reconstruction (the NP can attend to the query's own label). "
                "Check `eval_context`, or mark the stream `in_sample: true` if "
                "the overlap is intentional (a reconstruction probe like cv2)."
            )

    def build(self, *, views=None, query_rows: list[dict] | None = None) -> G2FBatch:
        """Collate the context rows into a flat ``[Nc, …]`` ``G2FBatch``."""
        rows = self.context_rows()
        if not rows:
            raise ValueError(
                f"ContextProvider resolved 0 context rows for "
                f"eval_context={self.eval_context!r}."
            )
        if query_rows is not None:
            self._assert_disjoint(rows, query_rows)
        if views is None:
            views = getattr(self.dataset, "dl_views", None)
        return g2f_collate_fn(rows, self.pad_value, views)
