"""Per-modality set-element (timepoint) subsampling (train-only).

Optional augmentation. ``augment_tasks`` (task-level, keeps a fixed
``round(rate·T)`` per curve) is invoked from :func:`utils.data.tasks.np_collate`,
which is now the **reference collate** — the live TNP training
path uses the :class:`~utils.data.tasks.PrecomputedPool` fast path, whose
``_apply_timepoint_dropout`` masks (rather than physically drops) timepoints and
keeps a *Bernoulli* count, equivalent in expectation. ``augment_samples`` (the
per-sample analogue) is the per-sample (flat-batch) train loader's path via
``augmented_collate_fn``. For each unit a keep-fraction is drawn per modality
(``p ~ U[lo, hi]``, independently for VI and weather); the kept timepoints are
chosen per curve (curves differ in length) but always at least ``min_kept`` and
in temporal order.

This lives in the train collate, **not** as an ``AxisTransform`` in
``assemble_view`` — an AxisTransform would run for every split and silently
subsample val/test curves. The val/test loaders use the plain ``g2f_collate_fn``
and never touch this, so "off at eval" is structural. ``keep_frac_range:
[1.0, 1.0]`` is a no-op (the task list is returned unchanged, byte-identical).

BY DESIGN, the TNP task and per-sample paths differ on two axes and are NOT the
same sampler:
  - granularity: TNP draws one rate per *task* (shared across its samples and
    its context↔query); the per-sample path draws one rate per *sample*. This
    is architectural — a TNP task is a joint context+query unit; a per-sample
    sample is independent — and should not be unified.
  - count: the TNP fast path keeps a *Bernoulli* count (random ~Binomial); the
    per-sample / oracle path keeps a *fixed* ``round(rate·T)``. Equivalent in
    expectation; left divergent deliberately (unifying would change TNP training
    dynamics for a second-order effect).
Consequence: treat augmentation as a *within-family* regularizer — compare
aug-vs-no-aug within TNP and within the per-sample path separately, never as an
identical knob held constant across the two families.
"""

from __future__ import annotations

import numpy as np

# Modality name -> (coord field, channel field) on the raw sample dict.
_MODALITY_FIELDS = {
    "vi": ("dap", "channels"),
    "weather": ("weather_dap", "weather_values"),
}


def _resolve_modality(aug, key: str) -> tuple[float, float, int]:
    """Read ``(lo, hi, min_kept)`` for one modality from a dict/DictConfig.

    Missing modality or missing range => the identity ``(1.0, 1.0, 1)``.
    """
    block = None
    if aug is not None:
        block = aug.get(key) if hasattr(aug, "get") else getattr(aug, key, None)
    if block is None:
        return 1.0, 1.0, 1
    rng = (
        block.get("keep_frac_range")
        if hasattr(block, "get")
        else getattr(block, "keep_frac_range", None)
    )
    lo, hi = (1.0, 1.0) if rng is None else (float(rng[0]), float(rng[1]))
    min_kept = (
        block.get("min_kept", 1)
        if hasattr(block, "get")
        else getattr(block, "min_kept", 1)
    )
    return lo, hi, int(min_kept)


def _is_noop(ranges: dict[str, tuple[float, float, int]]) -> bool:
    return all(lo == 1.0 and hi == 1.0 for lo, hi, _ in ranges.values())


def is_noop(aug) -> bool:
    """True if ``aug`` is a structural no-op (``None`` or every modality's
    ``keep_frac_range`` is ``[1.0, 1.0]``).

    The ``conf/augment/none.yaml`` default resolves to a *dict* of ``[1.0, 1.0]``
    ranges, not Python ``None`` — so callers that gate behaviour on "is
    augmentation active?" must use this, not an ``is None`` check.
    """
    if aug is None:
        return True
    ranges = {key: _resolve_modality(aug, key) for key in _MODALITY_FIELDS}
    return _is_noop(ranges)


def _subsample_curve(coord, chan, keep_frac: float, min_kept: int, rng):
    """Keep a temporally-ordered random subset of a curve's timepoints."""
    T = coord.shape[0]
    k = int(round(keep_frac * T))
    k = max(min_kept, k)
    k = min(k, T)
    if k >= T:
        return coord, chan
    idx = np.sort(rng.choice(T, size=k, replace=False))
    idx = idx.tolist()
    return coord[idx], chan[idx]


def _subsample_sample(sample, ranges, rates, rng):
    """Copy ``sample`` with each active modality's curve subsampled at ``rates``.

    ``ranges`` maps modality -> ``(lo, hi, min_kept)`` and ``rates`` maps
    modality -> the drawn keep-fraction. The shared dataset row is never
    mutated (the returned dict is a shallow copy with replaced curve fields).
    """
    new = dict(sample)
    for modality, (coord_f, chan_f) in _MODALITY_FIELDS.items():
        if coord_f not in sample or chan_f not in sample:
            continue
        lo, hi, min_kept = ranges[modality]
        if lo == 1.0 and hi == 1.0:
            continue
        coord, chan = _subsample_curve(
            sample[coord_f], sample[chan_f], rates[modality], min_kept, rng
        )
        new[coord_f] = coord
        new[chan_f] = chan
    return new


def augment_tasks(tasks, aug, rng=None):
    """Return task specs with per-task, per-modality curve subsampling applied.

    Args:
        tasks: list of ``B`` specs ``(context_samples, query_samples)``.
        aug: augmentation config (dict / DictConfig / object) with optional
            ``vi`` and ``weather`` blocks, each
            ``{keep_frac_range: [lo, hi], min_kept: int}``. ``None`` => no-op.
        rng: optional ``numpy.random.Generator`` (defaults to a fresh unseeded
            one, so draws vary across batches).

    Returns:
        New task specs with **copied** subsampled sample dicts (the shared
        dataset rows are never mutated). Returned unchanged when ``aug`` is
        ``None`` or every modality range is ``[1.0, 1.0]``.
    """
    if aug is None:
        return tasks

    ranges = {key: _resolve_modality(aug, key) for key in _MODALITY_FIELDS}
    if _is_noop(ranges):
        return tasks

    rng = np.random.default_rng() if rng is None else rng

    out = []
    for ctx, qry in tasks:
        # One rate per modality, shared across all samples in this task.
        rates = {
            modality: float(rng.uniform(lo, hi))
            for modality, (lo, hi, _) in ranges.items()
        }
        out.append(
            (
                [_subsample_sample(s, ranges, rates, rng) for s in ctx],
                [_subsample_sample(s, ranges, rates, rng) for s in qry],
            )
        )
    return out


def augment_samples(samples, aug, rng=None):
    """Return sample dicts with per-**sample**, per-modality curve subsampling.

    The per-sample analogue of :func:`augment_tasks` for the per-sample
    train path: each sample is its own unit, so one keep-fraction per modality
    is drawn independently per sample (``p ~ U[lo, hi]``) and applied to that
    sample's curves. Train-only; the shared dataset rows are never mutated.
    Returned unchanged when ``aug`` is ``None`` or every modality range is
    ``[1.0, 1.0]``.

    Args:
        samples: list of raw sample dicts (one G2F record each).
        aug: augmentation config (see :func:`augment_tasks`). ``None`` => no-op.
        rng: optional ``numpy.random.Generator`` (defaults to a fresh unseeded
            one, so draws vary across batches — parity with ``np_collate``).
    """
    if aug is None:
        return samples

    ranges = {key: _resolve_modality(aug, key) for key in _MODALITY_FIELDS}
    if _is_noop(ranges):
        return samples

    rng = np.random.default_rng() if rng is None else rng

    out = []
    for sample in samples:
        rates = {
            modality: float(rng.uniform(lo, hi))
            for modality, (lo, hi, _) in ranges.items()
        }
        out.append(_subsample_sample(sample, ranges, rates, rng))
    return out


def augmented_collate_fn(batch, collate_fn, aug):
    """Subsample each sample's curves (train-only), then delegate to ``collate_fn``.

    A module-level, picklable wrapper (survives DataLoader worker pickling when
    bound with :func:`functools.partial`) that gives the per-sample
    (flat-batch) train loader the same timepoint augmentation the TNP path
    gets via :func:`utils.data.tasks.np_collate`.
    """
    return collate_fn(augment_samples(batch, aug))
