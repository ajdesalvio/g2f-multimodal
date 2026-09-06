"""One-hot encodings of metadata columns as derived features.

Provides per-sample one-hot vectors for `Env.Year`, `Pedigree`, `Year`,
or any other metadata column as `derived_features` entries. The
primary use case is feeding `env_onehot` into
`InteractionKernelBuilder` as the second component of a kernel
Hadamard product — this reproduces the R reference's `KG_GE_*` / `KP_PE`
within-env interaction terms via the Hadamard ↔ tensor-product
identity:

    K_int(i, j) = ⟨φ_A(i), φ_A(j)⟩ · ⟨onehot_env(i), onehot_env(j)⟩
                = K_A(i, j) · 𝟙[env_i == env_j]
                = (Z_e Z_e^T)(i, j) · K_A(i, j)
                = KG_GE_A(i, j)

Runs as the first feature processor (priority 0; only the
`axis_source` axis builder runs earlier, at -1) so its outputs are
available as
`derived_features` to every downstream processor.

Leakage policy: `fit_scope="all"` (default). This vocabulary is over
IDENTITY LABELS (env / pedigree / year), NOT fitted statistics — you know
which environment / pedigree / year a held-out sample belongs to (the label
is observable a priori), and a held-out category's one-hot column is all-zero
across the train rows, so it carries no train-learnable response signal. That
is why `"all"` is acceptable here even though it is NOT leakage-free for
`enviromic` (whose `"all"` pulls the held-out environment's *weather values*
into the kernel fit — a future year's weather is not available at training
time; see that module, which therefore defaults to `"train"`). `"all"` is also
the practical default because `fit_scope="train"` raises `KeyError` on a
held-out category at transform (a new env / pedigree absent from the train
vocab). `genomic` likewise defaults to `"all"` (a new line's genotype is
known at prediction time).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import ClassVar

import numpy as np

from .base import BaseProcessor, OrchestratorContext
from .registry import register_processor

logger = logging.getLogger(__name__)

# Sentinel value for the canonical Env.Year combined identifier.
# Routes through `ctx.env_year()` rather than a raw column read.
_ENV_YEAR_KEY = "Env.Year"


@register_processor
class MetadataFeaturesProcessor(BaseProcessor):
    """One-hot encodings of metadata columns as derived features.

    Parameters
    ----------
    sources : list of dicts, each with:
        - ``name`` (str) — output key in `derived_features`.
        - ``column`` (str) — metadata column to one-hot. Special value
          ``"Env.Year"`` routes through `ctx.env_year()` for the
          canonical fold-identifier series. Anything else is read
          directly from `metadata_df[column]` and must exist there.
    fit_scope : ``"all"`` (default) or ``"train"``. Determines which
        rows the vocabulary covers; values seen at transform but not
        present at fit raise `KeyError`.
    """

    # ── BaseProcessor metadata ──────────────────────────────────────
    name: ClassVar[str] = "metadata_features"
    priority: ClassVar[int] = 0
    # Pure metadata: no batch fields, no eigen sources, no upstream
    # dependencies, no coverage filter (every sample has metadata).

    @classmethod
    def from_config(cls, config: dict) -> "MetadataFeaturesProcessor":
        """Hydra-config factory used by the orchestrator."""
        return cls(
            sources=config.get("sources", []),
            fit_scope=config.get("fit_scope", "all"),
        )

    def __init__(
        self,
        sources: list[dict],
        fit_scope: str = "all",
    ):
        self.sources = sources
        if fit_scope not in ("all", "train"):
            raise ValueError(
                f"fit_scope must be 'all' or 'train', got {fit_scope!r}."
            )
        self.fit_scope = fit_scope
        self._validate_sources()
        # Per-source ordered vocabulary; populated by fit() / load_fitted().
        self._vocab: dict[str, list[str]] = {}

    def _validate_sources(self) -> None:
        names = set()
        for src in self.sources:
            if "name" not in src or "column" not in src:
                raise ValueError(
                    f"Each metadata_features source needs 'name' and "
                    f"'column' keys; got {src!r}."
                )
            if src["name"] in names:
                raise ValueError(
                    f"Duplicate source name: {src['name']!r}."
                )
            names.add(src["name"])

    # ── BaseProcessor lifecycle ────────────────────────────────────

    def fit(self, ctx: OrchestratorContext) -> None:
        """Build per-source ordered vocabulary from the fitting set."""
        for src in self.sources:
            values = self._extract_values(
                ctx, src["column"], scope=self.fit_scope,
            )
            self._vocab[src["name"]] = sorted(set(values))
        logger.info(
            "metadata_features fit: %s",
            {k: len(v) for k, v in self._vocab.items()},
        )

    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """One-hot encode each source for the named split."""
        if not ctx.split_data(split):
            return None
        result: dict[str, np.ndarray] = {}
        for src in self.sources:
            values = self._extract_values(ctx, src["column"], scope=split)
            vocab = self._vocab[src["name"]]
            result[src["name"]] = self._onehot(values, vocab, src["name"])
        return result

    @property
    def feature_dims(self) -> dict[str, int]:
        return {name: len(vocab) for name, vocab in self._vocab.items()}

    def raw_kernel(
        self,
        ctx: OrchestratorContext,
        source_name: str,
        sample_indices: np.ndarray,
    ) -> np.ndarray:
        """Return ``Z @ Z^T`` for the one-hot encoding of the requested
        rows. For ``env_onehot`` this is the R reference's ``K_E_identity``:
        ``K[i, j] = 1`` iff samples ``i`` and ``j`` share the column
        value.
        """
        src = next(s for s in self.sources if s["name"] == source_name)
        column = src["column"]
        if column == _ENV_YEAR_KEY:
            series = ctx.env_year()
        else:
            series = ctx.metadata_df[column].astype(str)
        values = series.loc[sample_indices].tolist()
        vocab = self._vocab[source_name]
        Z = self._onehot(values, vocab, source_name)
        return (Z.astype(np.float64) @ Z.astype(np.float64).T)

    # ── Cache: save / load fitted state ────────────────────────────

    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Hash source config + fit_scope + the resolved per-source
        values that drive the vocabulary.
        """
        h = hashlib.sha256()
        h.update(b"metadata_features_v1")
        src_repr = json.dumps(
            [{"name": s["name"], "column": s["column"]}
             for s in self.sources],
            sort_keys=True,
        ).encode()
        h.update(src_repr)
        h.update(f"|fit_scope={self.fit_scope}".encode())
        for src in self.sources:
            values = self._extract_values(
                ctx, src["column"], scope=self.fit_scope,
            )
            h.update(b"|values=")
            h.update("|".join(sorted(set(values))).encode())
        return h.hexdigest()

    def save_cache(self, cache_dir: str, key: str) -> None:
        """Persist per-source vocabularies to
        ``cache_dir/metadata_features_<key[:16]>.npz``.
        """
        if not self._vocab:
            raise RuntimeError("Cannot save_cache before fit().")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(
            cache_dir, f"metadata_features_{key[:16]}.npz",
        )
        payload: dict[str, np.ndarray] = {
            "config_json": np.array(
                json.dumps({
                    "sources": self.sources,
                    "fit_scope": self.fit_scope,
                }),
                dtype=object,
            ),
            "source_names": np.array(
                list(self._vocab.keys()), dtype=object,
            ),
        }
        for name, vocab in self._vocab.items():
            payload[f"vocab[{name}]"] = np.array(vocab, dtype=object)
        tmp_path = path + f".tmp.{os.getpid()}.npz"
        np.savez(tmp_path, **payload)
        try:
            os.replace(tmp_path, path)
        except FileNotFoundError:
            pass
        logger.info("metadata_features cache saved: %s", path)

    def load_fitted(self, cache_dir: str, key: str) -> None:
        """Restore per-source vocabularies from cache."""
        path = os.path.join(
            cache_dir, f"metadata_features_{key[:16]}.npz",
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"metadata_features cache not found: {path}"
            )
        data = np.load(path, allow_pickle=True)
        saved = json.loads(str(data["config_json"].item()))
        if (
            saved.get("sources") != self.sources
            or saved.get("fit_scope") != self.fit_scope
        ):
            raise ValueError(
                f"metadata_features config mismatch: cache has "
                f"{saved}, current has sources={self.sources}, "
                f"fit_scope={self.fit_scope}."
            )
        names = [str(n) for n in data["source_names"]]
        self._vocab = {
            name: [str(v) for v in data[f"vocab[{name}]"]]
            for name in names
        }
        logger.info("metadata_features cache loaded: %s", path)

    # ── Private helpers ────────────────────────────────────────────

    def _extract_values(
        self, ctx: OrchestratorContext, column: str, scope: str,
    ) -> list[str]:
        """Pull metadata values from `ctx` for the requested scope.

        ``scope`` is either "all" / "train" (both used at fit time) or
        a single split name "train"/"val"/"test" (used at transform).
        """
        if column == _ENV_YEAR_KEY:
            series = ctx.env_year()
        else:
            if column not in ctx.metadata_df.columns:
                raise KeyError(
                    f"metadata_features source column {column!r} not "
                    f"in metadata_df. Available columns: "
                    f"{list(ctx.metadata_df.columns)}."
                )
            series = ctx.metadata_df[column].astype(str)
        if scope == "all":
            indices: list[int] = []
            for s in ("train", "val", "test"):
                indices.extend(ctx.split_indices.get(s, []))
        else:
            indices = ctx.split_indices.get(scope, [])
        return series.loc[indices].tolist()

    def _onehot(
        self, values: list[str], vocab: list[str], source_name: str,
    ) -> np.ndarray:
        """One-hot encode ``values`` against ``vocab``.

        Raises `KeyError` listing the offending values when any value
        is not in the fit-time vocabulary (almost always means the
        config used `fit_scope="train"` and a held-out env / pedigree
        appears at transform).
        """
        v_to_i = {v: i for i, v in enumerate(vocab)}
        unseen = sorted(set(values) - set(vocab))
        if unseen:
            raise KeyError(
                f"metadata_features source {source_name!r}: values "
                f"{unseen} were not seen at fit time. Either change "
                f"the source's fit_scope to 'all' (default), or "
                f"adjust the vocabulary."
            )
        arr = np.zeros((len(values), len(vocab)), dtype=np.float32)
        for i, v in enumerate(values):
            arr[i, v_to_i[v]] = 1.0
        return arr
