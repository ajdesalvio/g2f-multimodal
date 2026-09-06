# Feature Processing Layer

The feature-processing layer extracts fixed-size features from auxiliary
data modalities (genotype, phenotype via FPC scores, weather, environment
similarity) and attaches them to `G2FDataset` samples so downstream DL
models and FPCA baselines consume them uniformly as `derived_features`
or as additional batch fields on `G2FBatch`.

**Unified API.** `FeatureProcessor` orchestrates all enabled processors
in priority order via the **registry** (`utils/data/processing/registry.py`),
runs them during `G2FDataset.__init__`, and exposes the same state
twice:
1. **Train mode:** `fit_transform(dataset)` — fit each processor on the
   train split only by default (`genomic` defaults to `fit_scope="all"`
   — genotype is pre-planting; the kernel processors flip to `all` in
   the transductive BGLR/GBLUP composers), transform every split,
   attach features.
2. **Predict mode:** `load_and_transform(dataset, cache_keys)` — load
   fitted state from the per-processor cache (content-addressed by
   `cache_keys[proc_name]`) and run transform-only on every split.

`train.py` writes cache keys to `processing_metadata.json` in the
artifact directory after `fit_transform` completes. The eval path reads
them back via `setup._resolve_processing_cache_keys` so the predict path
reuses the exact fitted state that produced the training checkpoint.

**`persist_cache: false`** on a processor block skips that processor's
cache *write* (the fit still runs; only the on-disk entry is omitted).
Every shipped FPCA config sets it on the heavy kernel processors
(`genomic`, `phenomic`, `weather_kernel`, `interaction`), whose entries
run to gigabytes per split and are only ever read by predict-mode
`load_fitted` — which the FPCA pipeline does not use. FPC-score caches
(`vi_fpca`, `weather_fpca`) and the metadata vocabulary still persist.
The orchestrator records unpersisted processors in
`processing_metadata.json`, and predict-mode key resolution refuses such
metadata with an explicit message instead of failing deep inside
`load_fitted`. Leave the default (`true`) for any DL config that will be
evaluated from a checkpoint.

## Adding a new processor

Every processor is a registered subclass of `BaseProcessor`. Adding one
is **a single decoration**, not a tour through the codebase:

```python
# utils/data/processing/myproc.py
from typing import ClassVar
from .base import BaseProcessor, OrchestratorContext
from .registry import register_processor


@register_processor
class MyProcessor(BaseProcessor):
    name: ClassVar[str] = "myproc"
    priority: ClassVar[int] = 11                 # runs after interaction (priority 10)
    requires: ClassVar[tuple[str, ...]] = ()     # upstream proc names
    has_eigen_sources: ClassVar[bool] = False    # D1 rank check?
    produces_views: ClassVar[tuple[str, ...]] = ()         # e.g. ("weather",)
    produces_batch_fields: ClassVar[tuple[str, ...]] = ()  # legacy, vestigial
    eigen_source_types: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def from_config(cls, config: dict) -> "MyProcessor":
        return cls(**{k: v for k, v in config.items() if k != "enabled"})

    def __init__(self, ...): ...
    def fit(self, ctx: OrchestratorContext) -> None: ...
    def transform(self, ctx, split) -> dict[str, np.ndarray] | None: ...
    def cache_key(self, ctx) -> str: ...
    def save_cache(self, cache_dir, key) -> None: ...    # optional
    def load_fitted(self, cache_dir, key) -> None: ...   # optional

    @property
    def feature_dims(self) -> dict[str, int]: ...        # default {}

    @classmethod
    def coverage_mask(cls, metadata_df, proc_config) -> np.ndarray | None:
        return None  # override only for processors with partial coverage
```

Then **import the module** from `utils/data/processing/__init__.py` so the
`@register_processor` decorator runs at import time. That's it — the
orchestrator picks it up via the registry; cross-cutting checks
(`rank_check`, `_validate_config`, `_filter_to_modality_coverage`)
read the new processor's ClassVars automatically. **No edits to the
orchestrator, validator, or coverage filter required.**

## Execution order

All twelve processors share a fixed priority order (per `cls.priority`)
so dependencies are respected without graph-resolution logic:

| Priority | Processor | Depends on | Emits |
|---|---|---|---|
| -1 | `axis_source` | weather CSV (GDD) | attaches per-sample `gdd` / `agdd` derived time axes (no `derived_features`) |
| 0 | `metadata_features` | `metadata_df` | `derived_features[src.name]` (env_onehot, pedigree_onehot, ...) |
| 1 | `vi_fpca` | raw VI curves | `derived_features["vi_fpc_scores"]` |
| 2 | `phenomic` | `vi_fpc_scores` | `derived_features[src.name]` (per source) |
| 3 | `weather_fpca` | weather CSV | `derived_features["weather_fpc_scores"]` |
| 4 | `weather_kernel` | `weather_fpc_scores` | `derived_features[src.name]` (env-level K_W eigen V·√D, env-broadcast per sample) |
| 5 | `genomic` | genomic CSV | `derived_features[src.name]` (per source) |
| 6 | `enviromic` | weather CSV | `derived_features[src.name]` (per source) |
| 7 | `raw_weather` | weather CSV | `weather_dap` / `weather_values` per sample (in-place) |
| 8 | `raw_vi` | raw VI curves | Slices the per-sample `channels` matrix columns to a VI subset, in place |
| 9 | `weather_concat` | weather CSV | Widens the per-sample `channels` matrix in place (adds W columns) |
| 10 | `interaction` | upstream `derived_features` | `derived_features[src.name]` (Hadamard kernels per source) |

`axis_source` runs first (priority -1) to attach the derived `gdd` / `agdd`
time axes before any consumer reads them. `metadata_features` (priority 0) is
the first *feature* processor, so its outputs (e.g. `env_onehot`) are available
to every downstream processor — most notably as components of `interaction`
Hadamard kernels.
`raw_vi` (priority 8) slices the native VI columns *before*
`weather_concat` appends its weather tail to the same tensor.
`weather_concat` runs late (priority 9) because it changes
`effective_y_dim`, which `_inject_set_encoder_ydims`
reads afterwards. `interaction` runs last (priority 10) so every
upstream processor's `derived_features` are attached before it
combines them.

`phenomic` declares `requires=("vi_fpca",)` because it builds its
kernel from FPC scores — `FeatureProcessor._validate_dependencies`
enforces this at construction. The `interaction` processor doesn't
declare static `requires`: its dependencies are config-driven (the
`components` list per source), and missing dependencies surface at
fit time with a fold-named error message.

## The 12 processors

### `axis_source` — derived GDD / AGDD time axes (priority −1)

Reads the weather CSV's `GDD` column (or recomputes GDD from
Tmax/Tmin), builds per-env cumulative AGDD, and attaches per-sample
`gdd` / `agdd` axes via a DAP lookup so any View can use them as the
time coordinate. `dedup: true` builds the deduped clean-AGDD grid, and
`vi_dedup: true` additionally places the per-sample VI axis on the
reference `dap.gdd` scale (collapsed DAPs dropped). Runs before every
feature processor; configured by the `conf/axis/` group.

### `metadata_features` — one-hot encodings of metadata columns

Stateless-ish: builds an ordered vocabulary per source at fit time and
emits one-hot vectors at transform time. Sources declare a `column`
name (any column of `metadata_df`); the special value `"Env.Year"`
routes through `ctx.env_year()` for the canonical fold-identifier.

```yaml
metadata_features:
  enabled: true
  fit_scope: all                      # "all" or "train"
  sources:
    - name: env_onehot
      column: Env.Year                # canonical fold ID
    - name: pedigree_onehot
      column: Pedigree
```

Feature dim per source = vocabulary size after fit. `fit_scope: all`
(default) covers every value in train ∪ val ∪ test (no leakage —
metadata is observable pre-yield); `fit_scope: train` raises `KeyError`
at transform on values not seen at fit.

The primary use case is feeding `env_onehot` into `interaction` as
the second component of a Hadamard kernel — this reproduces the R reference's
`KG_GE_*` / `KP_PE` within-env interaction terms exactly via the
kernel-tensor identity.

### `vi_fpca` — VI FPCA via R `fdapace`

Fits sparse FPCA on train VI curves (one FPCA per VI in the fit set —
all 37 by default) and
projects all splits via CE/BLUP. Invokes `baselines/fpca_compute.R`
in `--mode train` (fit + project) then `--mode predict` (for
`transform_split` in predict mode). The R script stores the fitted
model objects as `fpca_models.rds` in the cache directory so predict
mode can project new curves without re-fitting. `n_components` is a
post-load slice — the R script outputs all available components and
the Python side slices down, so sweeping `n_components` reuses the
same cache entry.

Feature name: `vi_fpc_scores` (dim = `n_subset × n_components`,
defaulting to `37 × n_components` when no subset is set).

#### `axis: dap | agdd` and the `vi_dedup` AGDD convention

`vi_fpca.axis=agdd` swaps the fdapace time coordinate from DAP to the
per-sample `agdd` attached by `axis_source` (selected via the
`shared_t-agdd` / `fpca_t-agdd_*` axis configs). **Which** AGDD value each VI
observation gets is controlled by `axis_source.vi_dedup`:

- `vi_dedup: false` (default) — every VI obs DAP maps to the **full daily
  raw-`cumsum`** AGDD (no NaN; zero-GDD days carry the flat value). Our
  cleaner variant; safe for DL `agdd` coords / `agdd`-as-channel views.
- `vi_dedup: true` (set by `conf/axis/shared_t-agdd`) — **bit-exact reference**
  (`AGDD_FPCA_Projections_LOEO_V3.R`): VI obs are placed on the **deduped
  clean-AGDD scale** (`dap.gdd` = re-`cumsum` of the per-`(Env,AGDD)`-group
  averaged GDD), and an obs on a **collapsed** (non-representative zero-GDD)
  DAP becomes **NaN** — dropped by the tall-frame builder's `is.finite(t)`
  filter, exactly as the reference `left_join(dap.gdd)` NA does. Requires
  `axis_source.dedup: true`. Weather-FPCA already uses this deduped scale
  on AGDD, so `vi_dedup: true` makes VI and weather agree with the R reference.

  The two coincide for envs with no zero-GDD days; they differ only near
  cold-season edges (e.g. a few days in `WIH1/2/3.2020`). Per-curve VI
  dedup is **not** applied — VI obs are weeks apart so two never share an
  AGDD; this is purely the axis-scale + NaN-drop convention.

#### `vi_subset` is a post-load slice (Design B)

`vi_subset` selects which vegetation indices to emit. It does **not**
filter raw curves at ingestion: the per-sample `channels`
tensor always carries every VI in the dataset, R fits every VI in the
*fit set* (`fit_vars`, default `null` = all — the compute-set knob
mirroring `weather_fpca.fit_vars`, folded into the cache key only when
restricted; `vi_subset ⊆ fit_vars`), and the cache stores the full
per-split score matrix for that fit set. The subset
is applied as a column slice when emitting features. Two runs differing
only in `vi_subset` reuse the **same** cache entry; switching subsets
never re-invokes the R subprocess. This mirrors the existing post-load
semantics of `n_components`. Cache key sentinel: `vi_fpca`.

The processor knob (`dataset.processing.vi_fpca.vi_subset`) is normally
set via the `vi_subset` config group at `conf/vi_subset/<name>.yaml`
(see root README §Feature-subset ablation). The group also sets the
`vi_chan` slug piece, which renders into the VI peer's channel token
(`vi.c=<name>`) of `model_name`, so artifact paths and W&B run names stay
collision-safe across subset choices. Same pattern for the weather
variable subset via the `weather_subset` group (the `wthr_chan` piece →
`wthr.c=<name>`), which sets all four weather-consuming processors
atomically — `weather_vars` on enviromic / weather_concat / raw_weather,
and `fit_vars` / `emit_vars` on weather_fpca.

The `vi_subset` group sets the same subset on **two** consumers at once:
`vi_fpca` (both `fit_vars`, the compute set, and `vi_subset`, this
post-load score slice — for FPCA models) and
`raw_vi.vi_subset` (the raw-curve column slice, for DL models — see the
`raw_vi` processor below). Each model enables only one of the two
processors, so the other's keys are inert. The DL raw-VI slice and the
FPCA score slice never co-run, and enabling `raw_vi` never touches the
FPCA cache.

#### `missing_values`: how interpolated cells reach fdapace

`missing_values` controls whether VI cells that were originally NaN
(and have since been linearly interpolated by the reader) enter
fdapace as observations:

| Value | Behavior |
|-------|----------|
| `skip` *(default)* | Drop cells flagged in the per-sample `vi_nan_mask` before exporting the tall CSV. fdapace only sees originally-real observations — matches the R reference, which feeds tall-format observations with no NaN rows. |
| `interpolate` | Pass interpolated cells through to fdapace as if observed (legacy scheme). |

The two modes produce different fitted bases and live under separate
content-addressed cache entries — `compute_vi_fpca_cache_key`
incorporates `missing_values` so the two modes never collide. The
mode is also recorded in `metadata.json` and verified at
`load_fitted` time. The DL pipeline ignores `vi_nan_mask` and
consumes the dense interpolated values.

The canonical VI ordering used by the slice comes from
`G2FDataset.vi_names` (alphabetic by VI name, captured at ingestion
and persisted in the dataset's `.pt` cache). The processor records
that list in its own `metadata.json` so predict-mode `load_fitted`
can resolve a subset against it without rerunning the reader.

Asking for a VI not present in the dataset raises `ValueError` at fit
time. Per-VI FPCA independence makes the sliced output bit-identical
to a fit-on-subset path.

#### `fit_mask` and the OOD-validation caveat (eval-streams B1)

`vi_fpca` fits its FPCA basis over the **row-level `fit_mask`** — the FIT
role (see [`docs/cv_schemes.md`](../../../docs/cv_schemes.md)
§"Per-modality fit scope" for the fit-scope topic in general). This basis
is *unsupervised* (it never sees yields), so fitting it over FIT∪OBSERVE is
leak-safe for the **scored** quadrants — but it is **not** automatically
honest for a DL **OOD validation carve**.

The DL eval-streams plan adds a `carve` stream that pulls an out-of-distribution
validation holdout out of the train pool (e.g. unseen genotypes under
`grouped_variety`). Those carved val rows are *still in the FIT set* the basis
is fitted over, so the FPCA representation a held-out val row is encoded with
was partly fitted on that same row — an OOD val carve leaks through the basis.

Two invariants make this a non-trivial fix rather than a one-liner, so it is
**deferred** and **guarded** instead:

- The val carve deliberately does **not** touch `fit_mask` (the carve is a
  training-loop concern; predict-mode even drops the carve so CV2 scores over
  *all* FIT rows for R/BGLR parity).
- The `vi_fpca` cache key does **not** encode the val fit-set, so two configs
  differing only in their val carve would collide on one cached basis. (The
  key stays FIT-only because predict mode depends on it; entries do carry
  per-split content digests that are verified on every hit, so a same-size
  but differently composed observe/val/test split is a cache *miss* rather
  than a silently mis-permuted score matrix. Legacy entries without digests
  load with a warning.)

**Guardrail (shipped):** configuring a DL `carve` eval stream while `vi_fpca`
is enabled is a hard `ValueError` at the VALIDATE gate
(`setup._check_fpca_basis_guardrail`), tagged
`# TODO(eval-streams): OOD val + FPCA-basis features`. The check is gated on
a carve stream actually being present: an observe-only eval-stream bundle
(e.g. `random_kfold`) has nothing to leak, so it composes with `vi_fpca`
cleanly. The standard DL
set-encoder models learn the VI representation end-to-end and do **not** enable
`vi_fpca` (so `fit_mask is None` and the exclusion would be a no-op) — hence the
deferral is safe. Lifting it requires extending the `vi_fpca` cache key to
encode the val fit-set and excluding the carved val rows from calibration.

### `phenomic` — RKHS over VI FPC scores

Builds a phenomic similarity kernel from the `vi_fpc_scores` feature
produced by `vi_fpca`. Two source types:
- `phenomic_relmat`: eigen-projection of the centered train kernel,
  top `n_components` eigenvectors (kernel PCA). `n_components: null`
  keeps all eigenvectors (full-rank).
- `phenomic_relmat_rows`: rows of the centered kernel as features
  (each sample = similarity vector to every training sample).

Both support `linear` (scaled dot product) and `rbf` (Gaussian) kernels.

#### `fit_scope` (kernel fitting set)

| Config value | Behavior |
|---|---|
| `train` *(default)* | Kernel fit on the train split only. Val/test rows are produced via cross-kernel projection on the train-fitted eigenspace. Preserves historical behavior of every existing config. |
| `all` | Kernel fit on the union of train+val+test FPC scores (themselves leak-safe — `vi_fpca` always fits FPCA on train only and projects held-out splits via CE/BLUP). The eigenbasis spans every row in the dataset, equivalent to a full N×N `KP <- tcrossprod(VI_all) / ncol(VI_all)`; required for the BGLR `RKHS(KP)` term. |

`fit_scope=all` does **not** alter `vi_fpca`'s leak-safe contract — it
only widens the fitting set of the **phenomic** kernel/eigenbasis.
Yields are never used by phenomic; the union it stacks contains only
already-projected FPC scores. `fit_scope` participates in the cache
key (sentinel `phenomic_v4` for `train`, `phenomic_v5` for `all`), so
`train` and `all` runs never collide on the same hash and `load_fitted`
raises on a scope mismatch.

The eigendecomposition applies a relative noise filter (`eps=1e-10`)
under **every** fit scope. Earlier releases skipped it under `all`,
which kept roughly N/2 spurious ~1e-17 modes that defeated the rank
check and bloated cache and compute; the `_v5` sentinel under `all`
evicts exactly those stale entries. The same rule and conditional
sentinel bump apply to `genomic`, `interaction`, `weather_kernel` and
`enviromic`. R-reference parity for the transductive BGLR/GBLUP path is
unaffected: it consumes the unfiltered `raw_kernel()` and relies on
BGLR's own `tolD`, not on this eigenbasis.

#### Per-source `scaling` (input pre-processing)

Each source declares an optional `scaling` field that controls how
`vi_fpc_scores` is transformed before the kernel is built. Two sources
with different scalings get separate fitted state (their own kernel,
centering, eigendecomposition); two sources sharing a scaling reuse
state and only differ in slicing.

| Config value | Slug code | Scope | Method | Leakage |
|---|---|---|---|---|
| `none` *(default)* | *(no infix)* | — | — | n/a — pass-through |
| `global_zscore` | `gz` | Global | Per-column z-score with **train-only** stats | Strict |
| `within_env_zscore` | `wez` | Per-env | Per-column z-score using each env's own samples | Transductive — held-out env's stats come from its own samples; harmless for Pearson r |

The slug code is the **single source of truth** mapping the `scaling`
config value to its slug spelling (see `score_scalers._SCALING_REGISTRY`).
Under the unified per-encoder grammar (see `baselines/README.md`) it is the
`.s=` facet on the owning peer token — e.g. `vi.…s=gz`, `vi.…s=wez`,
`wthr.…s=gz`; `none` is omitted.

Example multi-scaling config (one model with three KPs for ablation):

```yaml
phenomic:
  enabled: true
  kernel: linear
  sources:
    - name: phenomic_relmat_raw
      type: phenomic_relmat
      n_components: 20
      scaling: none
    - name: phenomic_relmat_gz
      type: phenomic_relmat
      n_components: 20
      scaling: global_zscore
    - name: phenomic_relmat_wez
      type: phenomic_relmat
      n_components: 20
      scaling: within_env_zscore
```

Each per-source `.npz` cache entry covers exactly one scaling, so the
three coexist under different cache keys; bumping/removing one leaves
the others warm. `within_env_zscore` requires the orchestrator to pass
env labels at fit and transform time (already wired); a transform call
on an env not seen at fit raises `KeyError`.

Future scaling schemes (max-norm, unit-norm) can be added by appending
to `_SCALING_REGISTRY` with their own scope/method codes (`gm`/`wem`,
`gu`/`weu`); the rest of the pipeline picks them up automatically.

### `weather_fpca` — weather FPCA via R `fdapace`

One independent univariate FPCA per weather variable (same R script).
`fit_scope: "all"` fits on all envs' weather curves (weather is public,
no leakage); `fit_scope: "train"` restricts the fit to training envs
and projects others via CE/BLUP.

The cache stores the computed per-env scores directly so predict-mode
`transform_envs(env_list)` is a pure in-memory lookup (no R subprocess
needed when every requested env was present at fit time, which is the
normal LOO case).

Feature name: `weather_fpc_scores` (dim = `n_weather_vars × n_components`).

`axis: dap` (default) fits FPCA on the native daily grid; `axis: agdd`
substitutes the deduped clean-AGDD grid from the shared axis table
— the R FPCA call is identical, only `Lt` (the time coordinate) changes.
`dedup: true` (default) builds the unique monotonic AGDD axis fdapace
needs; `gdd` configures AGDD construction (defaults to the CSV `GDD`
column = reference-exact). Set together by the `shared_t-agdd` axis config.
The substituted AGDD curves flow into the content-addressed cache key, so
DAP and AGDD fits land in distinct entries.

#### `fit_vars` (compute scope) vs `emit_vars` (post-load slice)

Two **orthogonal** variable knobs, kept strictly separate:

- **`fit_vars`** — the *fit set*: which variables FPCA is actually
  **computed** for (fed to R, stored in the cache). `null` (default)
  fits every non-metadata CSV column — the historical "Design B"
  behaviour. Restrict it (e.g. `[PTR]`) where the fit is expensive:
  on the AGDD axis the cost is **per-variable** (a continuous,
  env-specific time grid blows up fdapace's covariance smoothing), so
  fitting only the variable of interest cuts compute ~`n_vars`×. By
  per-variable FPCA independence this is **bit-identical** to a full
  fit sliced to the same variable — it only avoids computing the
  unused fits. Because the cache stores exactly the fit set, `fit_vars`
  **is folded into the cache key** (only when restricted, so the
  default `null` key stays byte-identical to old caches).
- **`emit_vars`** — the *emit set*: which variables' scores are sliced
  out and handed downstream. `null` = every variable in the fit set.
  Pure **post-load slice** — it is **not** part of the cache key, so two
  runs with `emit_vars=null` and `emit_vars=[PTR]` (same `fit_vars`)
  reuse the **same** cache entry and switching emit subsets never
  re-invokes R. Mirrors the post-load semantics of `n_components`.

The only required relationship is **`emit_vars ⊆ fit_vars`** — you can
only slice out a variable that was fit. It's checked at construction
when both are explicit lists, and at slice time otherwise (clear error
naming the fit set).

Cache key: `weather_fpca_v4` over `(fit_scope, missing_values,
env_content_hash, [train_envs if fit_scope=train], axis[, dedup, gdd if
axis≠dap], [fit_vars if restricted])`. `missing_values` (`drop` | `ffill`,
see the baselines README) is tagged unconditionally so the two NaN
policies never share an entry. `emit_vars`, `n_components` and `scaling`
are post-load slices and are **not** in the key.

> **TODO (deferred): a restricted `fit_vars` run does not reuse a cached
> full (`fit_vars=null`) fit**, even though the full entry already stores
> the requested variable's scores (bit-identical by per-variable FPCA
> independence). It misses because `fit_vars` is in the key and
> `env_content_hash` only covers the loaded columns, so it recomputes —
> needlessly costly on AGDD. Fix sketch (superset-aware lookup vs.
> per-variable cache granularity, both gated on a variable-set-independent
> identity hash, plus relaxing `load_fitted`'s `fit_vars` equality check to
> a superset check) is tracked in the root `README.md` § *Known Issues and
> TODOs → FPCA baseline*.

The `global_zscore` scaler is fit per call on the post-slice columns,
so its mean/std shape-match whatever emit subset is currently active.
Per-column scaler stats commute with the variable-block slice
(slice-then-scale ≡ scale-then-slice on those columns), so emit-subset
features are bit-identical to a full-fit-then-slice path. Scaler
state is no longer persisted to disk — it derives in microseconds
from the cached fit-set scores plus the training-time train env list
(stored in `metadata.json`, alongside `fit_vars`).

Asking to emit a variable that was not fit raises `ValueError`.
Predict mode (`load_fitted` → `transform_envs`) reads the fit-set var
list, the train envs and `fit_vars` back out of `metadata.json`,
validates the fit set matches, and refits the scaler in memory.

#### `scaling` (single processor-level knob)

| Config value | Slug code | Behavior |
|---|---|---|
| `none` *(default)* | *(no infix)* | Pass-through. R fdapace output → slice → emit. |
| `global_zscore` | `gz` | Per-column mean/std fitted on the **train envs only** (always — regardless of `fit_scope`); applied to every env. Mirrors the R reference's `safe_scale_matrix` on weather scores. |

`within_env_zscore` is rejected at construction — env-level features have one row per env, so within-env std is degenerate. The scaler is **not** persisted to disk under Design B — it's refit in memory at every `fit_and_transform_all` and `load_fitted` call from the cached full per-env score matrix and the metadata-stored `train_envs`. Changing `scaling` reuses the underlying R fit at zero compute cost.

### `weather_kernel` — env-level K_W from weather FPC scores

Builds an environment-similarity kernel `K_W` from the
`weather_fpc_scores` feature produced by `weather_fpca`. `fit_scope`
governs the eigenbasis pooling (`train` by default; the transductive
BGLR/GBLUP composers set `all`); the score z-stats are always
train-env-only:

```
weather_score_matrix : (n_envs, n_vars * K)
weather_score_matrix <- safe_scale_matrix(weather_score_matrix)
K_W <- tcrossprod(weather_score_matrix) / ncol(weather_score_matrix)
KE_W <- Ze %*% K_W %*% t(Ze)             # project to obs level
```

The processor stops at the env-level eigen projection `V √D` (or the
raw kernel rows for `weather_relmat_rows`); the `Ze K_W Ze^T`
projection happens implicitly when the orchestrator broadcasts each
sample's env-row to its `derived_features` slot. Two samples in the
same env share the same feature vector by construction.

Source types:
- `weather_relmat`: per-sample feature equals
  `V_W[env_index_for_sample] * sqrt(D_W)`. For BGLR's BRR(V·√d), the
  implicit kernel between samples i, j equals `K_W[env_i, env_j]` —
  which is exactly `KE_W[i, j]`.
- `weather_relmat_rows`: per-sample feature equals the sample-env row of `K_W` over the fit envs
  corresponding to the sample's env (length `n_envs`).

Per-source `scaling`: same restricted registry as `weather_fpca` and
`enviromic` — `none` or `global_zscore` (mirrors the R reference's
`safe_scale_matrix` on weather scores). `within_env_zscore` is
rejected — env-level features have one row per env.

Processor-level `center_kernel: bool` (default `False`, BGLR/R-parity —
mirrors the R reference's uncentered `K_W` in weather-GxE; set `True` for textbook
double-centering).

Slug: this kernel is the weather encoder — the `wthr` peer in the unified
grammar (see `baselines/README.md`), carrying
`wthr.t=<axis>.c=<chan>.k=<viK>.s=<scaling>.rank=<eigK>[.uc]`, where `k` reads
from `weather_fpca.n_components` and `rank` is the eigen K of `K_W` (integer
or `full`). Concrete: the corrected R-reference weather-GxE PTR-only setup is
`wthr.t=dap.c=ptr.k=1.s=gz.rank=full.uc` (1 weather FPC on the V2 weather
curves, linear kernel of safe-scaled scores, full filtered eigen rank,
uncentered). The weather-GxE audit found K=1 on V2 —
not K=6 on V3 — reproduces the R reference's results; `wTraits <- c("PTR")`.

### Kernel double-centering (`center_kernel`)

`phenomic`, `genomic`, `enviromic`, `interaction`, and `weather_kernel`
each accept a processor-level `center_kernel: bool` (default `false`,
BGLR/R-parity — R applies no kernel centering). When `false` (default),
the raw kernel is eigen-decomposed and cross-kernel projections skip the
centering step — passes unmodified `K_A` / `K_D` / `K_P` / `K_W` into BGLR
RKHS terms. When `true`, the kernel is double-centered (`H K H`) before
eigendecomposition — the textbook kernel-PCA recipe; cross-kernel
projections then subtract the same training row/column means + grand mean.

| `center_kernel` | Slug suffix on each affected token | Where it actually matters |
|---|---|---|
| `false` *(default)* | `_uc` | `phenomic`, `enviromic` |
| `true` | *(no suffix)* | `phenomic`, `enviromic` (visible feature change), `genomic` (no-op — see note) |

**Genomic note.** VanRaden's `K_A = ZZ' / (2·Σ p(1−p))` is built on
already-centered dosages `Z = M − 2p`, so `K_A`'s rows sum to exactly
zero by construction and double-centering is a mathematical identity.
Same for Vitezica's `K_D`. The `center_kernel` knob is therefore
**a no-op on the genomic side** — both modes produce numerically
identical features. The knob is still wired so configs can declare
the choice consistently with phenomic / enviromic.

Cache key includes `center_kernel`; `load_fitted` rejects mismatches.
Slug: the phenomic kernel is carried by the `vi` peer's facets — an
uncentered K=4 kernel reads `vi.…k=4.rank=full.uc` (see `baselines/README.md`).

### `genomic` — GRM / dosage PCA

Loads the raw dosage matrix from a wide-format CSV via `load_genomic_csv`,
which caches the parsed result as an `.npz` keyed by file basename + mtime
(the ~566 MB CSV takes ~30s to parse; the cache makes subsequent runs
instant). The cache write is atomic (tmp + rename) to avoid corruption
under parallel jobs.

Computes additive (Van Raden 2008) and/or dominance (Vitezica 2013)
Genomic Relationship Matrices, centered and (optionally) eigendecomposed.
Source types:
- `additive_grm` / `dominance_grm`: kernel PCA eigenfeatures. FPCA peers spell
  this `ga` / `gd` with a `.rank=<K>` facet. DL set-encoder peers (e.g. the TNP
  model) mark the eigen representation with the `.eig` qualifier (parallel to
  `.rows` below) and carry the same `.rank=<K>` count facet as the FPCA peers:
  `ga.eig.rank=<K>` / `gd.eig.rank=<K>`, where `<K>` is `params.geno_k`. The
  rank is part of the model identity so that two runs differing only in the
  number of eigen basis vectors kept land in distinct artifact dirs / W&B runs
  instead of colliding.
- `additive_grm_rows` / `dominance_grm_rows`: centered kernel rows (DL peers
  `ga.rows` / `gd.rows`)
- `raw`: PCA on centered SNP dosages (peer `gr` with a `.rank=` facet)
- `raw_dosage`: literal per-pedigree SNP dosage matrix — no GRM, no PCA
  (width = SNP count); peer `gdos`, intended for a DL peer-MLP branch

`fit_scope: "all"` (default) fits allele frequencies, GRMs, and
eigendecompositions on every pedigree in the dataset (train ∪ val ∪
test); val/test projection is then a direct index into the precomputed
features (no cross-kernel needed). Genotypes are observed pre-planting
and are not yield-derived, so this introduces no leakage — same
rationale as `weather_fpca` / `enviromic`. `fit_scope: "train"`
restricts the fit to training pedigrees; allele frequencies from the
train subset are applied to val/test dosages and val/test features are
produced via cross-GRM projection.

### `enviromic` — env similarity kernel from weather

Builds environment-to-environment similarity from aggregated weather
(`mean`, `std`, or `mean_std`) via linear or RBF kernels.
`fit_scope` defaults to `train` (its module docstring explains why
`"all"` is not leakage-free under env holdout). Source types mirror
genomic
(`enviromic_relmat`, `enviromic_relmat_rows`, plus a `raw_weather`
aggregation-only variant).

`weather_vars` (processor-level, default `null`) selects a subset of
weather variable columns from the CSV before aggregation. `null`
keeps every non-metadata column; setting e.g. `[PTR]` mirrors the R reference's
weather-GxE K_W which uses photothermal ratio only. Cache key includes
the subset; `load_fitted` rejects mismatches.

`axis: dap` (default) aggregates over the native daily grid; `axis: agdd`
aggregates over the deduped clean-AGDD grid from the shared axis table
with `dedup`/`gdd` configuring that grid exactly as for
`weather_fpca` / `vi_fpca`. Set by the `shared_t-agdd` axis config.

#### Per-source `scaling`

Same registry semantics as `phenomic`, restricted to env-level
methods. Two sources with different scalings get separate kernel
state; sources sharing a scaling reuse it.

| Config value | Slug code | Behavior |
|---|---|---|
| `none` *(default)* | *(no infix)* | Aggregate matrix passed straight to the kernel. |
| `global_zscore` | `gz` | Per-column z-score on the aggregate matrix using fitting-set stats, then kernel + center + eigendecompose. Mirrors the R reference's `safe_scale_matrix` on the env×weather aggregate. |

`within_env_zscore` is rejected at construction (env-level → degenerate). Slug examples: `wgzLe17` (global-z, linear, eigen K=17), `wgzrows` (global-z kernel rows). Per-scaling state round-trips through `save_cache` / `load_fitted` so multiple scalings coexist as one `.npz`.

### `raw_weather` — daily weather curves as a separate set

Attaches per-sample `weather_dap` (`(T_w, 1)`), `weather_values`
(`(T_w, W)`), and `weather_channel_names`. These back the config-driven
`weather` View (`coords:[weather_dap] channels:[weather.*]`), which
`g2f_collate_fn` assembles into `G2FBatch.views["weather"]` (a
`ViewBatch`) via `assemble_view` — ready for the TNP `SampleTokenizer`'s
weather set encoder, peer to the `vi` encoder. (`axis: dap|agdd` flips the
`weather_dap` coordinate to the deduped clean-AGDD grid; see below.)

#### Independent per-axis scaling (`dap_scaling` / `value_scaling`)

The DAP axis and the weather-value axis each take their own scaling knob, so they can be normalized independently:

| Config value | Slug code | Behavior |
|---|---|---|
| `global_zscore` *(default for both axes)* | `gz` | Mean/std pooled over training-env rows. Per-column for values; scalar for DAPs. Applied to every env's daily curve. |
| `none` | *(no infix)* | Pass raw values through unchanged. |

`within_env_zscore` is rejected on both axes — daily rows belong to one env each, so per-env z-scoring would erase the between-env variation that's the whole point of conditioning on weather. Cache key includes both `dap_scaling` and `value_scaling`; `load_fitted` rejects mismatches.

#### Time axis (`axis` / `dedup` / `gdd`)

`axis: dap` (default) keeps the native daily grid; `axis: agdd` replaces
each env's `(DAP, values)` with the deduped clean-AGDD grid from the
shared axis table, so `weather_dap` now carries AGDD — the
set-encoder analog of `weather_fpca.axis=agdd`. `dedup: true` (default)
builds the unique monotonic AGDD grid; `gdd` configures AGDD construction
(defaults to the CSV `GDD` column = reference-exact). Selected via the
`shared_t-agdd` axis config. The `agdd` provenance is tagged into the
cache key, so the default DAP key stays byte-identical to pre-AGDD caches.

#### Extra coordinate axes (`extra_axes` / `extra_axes_scaling`)

`extra_axes` (default `[]`, subset of `[gdd, agdd]`) attaches the daily
`gdd`/`agdd` time axes as separate per-sample `weather_gdd` / `weather_agdd`
`(T_w, 1)` columns on the **native daily** weather grid (un-deduped raw
`cumsum`), aligned row-for-row with `weather_dap`. The config-driven
`weather` View then takes them as extra channels
(`channels:[weather.*, weather_gdd, weather_agdd]`) or coords — the DL
"weather encoder sees gdd/agdd" path (`dl_t-dap-gdd-agdd` axis
config). Only supported on `axis='dap'` (the
AGDD grid already carries AGDD as its coordinate, so extra gdd/agdd
channels would be redundant). `extra_axes_scaling` (`global_zscore`
default / `none`) z-scores them from **train-env stats only** (one
scalar mean/std per axis, leakage-safe — same pooling as `dap_scaling`).
Cache key and `.npz` round-trip include `extra_axes` + `extra_axes_scaling`
(tagged only when non-empty, so the default key is unchanged); the daily
vectors themselves are rebuilt deterministically from the CSV on
`load_fitted` rather than persisted.

### `raw_vi` — raw VI subset for the DL set encoder

Stateless and in-place: slices each sample's `channels` matrix
columns down to a named VI subset (e.g. `[NGRDI]`), resolving names
against `ctx.vi_names()` (the canonical per-column ordering). `vi_subset:
null` is a pass-through no-op. This is the DL analog of
`vi_fpca.vi_subset` — but where the FPCA path slices *cached FPC scores*,
`raw_vi` slices the *raw curves* that feed the TNP VI set encoder.
Opt-in (`enabled: true` only in DL model fragments with a raw VI branch);
disabled for FPCA, so the FPCA fit-then-slice caching stays intact.
Changing the subset lowers `effective_y_dim`, which is auto-injected into
`params.y_dim` (see Validator integration) — no model-config edit needed.

### `weather_concat` — weather lookup at VI DAPs

Stateless: for each sample, looks up the environment's daily weather
at the VI observation DAPs and widens the `channels` matrix from
`(T, 37)` to `(T, 37 + W)` in place (extending `channel_names`).
Increases `effective_y_dim` accordingly; `params.y_dim` is auto-injected
to match (see Validator integration). It has **no time axis of its own** —
the appended columns are sampled at the VI observation times, so they ride
on whatever coordinate the consuming View uses (DAP or AGDD). That is why
`weather_concat` has no `axis` knob (unlike `raw_weather` / `weather_fpca`
/ `enviromic`): flipping the main view to AGDD already re-indexes its
columns.

### Channel composition: in-place processors vs. View `channels` selectors

`raw_vi` and `weather_concat` compose the VI set-encoder's channel matrix
by **mutating the single shared `channels` tensor in priority order**:
`vi_fpca` reads the full matrix first (priority 1), `raw_vi` slices it to a
subset (8), then `weather_concat` appends its weather tail (9). The DL
collate's `main` View then selects `channels: ["vi.*"]` — i.e. "take every
column currently present." This is correct and sufficient **as long as
every consumer wants the same channel composition** from that one matrix.

The View layer can also express channel composition *declaratively* via
selectors (`channels: ["vi:NGRDI", "weather.*", ...]`; see
`utils/data/views.py`). We deliberately **do not** route `raw_vi` /
`weather_concat` through View selectors today — the in-place processors are
simpler and parity-stable, and the declarative path is not needed
while there is one channel composition shared across consumers.

**When to fold `raw_vi` / `weather_concat` into View `channels` selectors**
— do it only when a concrete need forces *per-view divergence* in channel
composition, which the single in-place matrix cannot express:

1. **Two set-encoder peers needing different VI subsets** of the same VI
   matrix (e.g. peer A on `[NDVI, EVI]`, peer B on `[NGRDI]`). `raw_vi`
   mutates the one matrix to one subset; View selectors give each peer its
   own.
2. **Weather-concat columns wanted by one DL view but not another** (e.g. a
   plain-VI peer alongside a VI+weather peer). The in-place append reaches
   every `vi.*` consumer; per-view selectors don't.
3. **Per-view axis for the weather-concat columns** (e.g. a DAP-axis VI peer
   and an AGDD-axis VI peer that both carry weather columns) — the single
   in-place append is indexed once, so divergent placement needs selectors.
4. **An actual ordering bug or a new channel-composing processor** that makes
   the implicit priority-8-then-9 coupling fragile.

Until one of these is a real requirement, folding is pure churn with parity
risk across every `vi_subset × weather_concat` combination, and buys nothing
functional.

**Two hard constraints when the fold does happen:**

- **Keep the FPCA path a post-fit slice (G4).** `vi_fpca.vi_subset` must
  remain a slice of the cached FPC-score fit (content-addressed on the
  `fit_vars` fit set), never a curve-emit-time channel selection — otherwise
  the FPCA cache forks per emit subset. The DL `raw_vi` fold and the FPCA slice are
  independent paths; folding one must not touch the other.
- **Validate against a frozen baseline.** Fold only with a parity reference
  in hand — the full-scale baseline metrics/scores from a clean run — so the
  refactor is verifiable bit-for-bit rather than a leap.

### `interaction` — Hadamard-of-kernels over derived features

Builds k-way interaction kernels by element-wise multiplication of
linear kernels over each component:

```
K_int(i, j) = ∏_m  K_m(i, j)  =  ⟨ ⊗_m φ_m(i),  ⊗_m φ_m(j) ⟩
```

For linear kernels this is identical to the linear kernel of the
per-sample tensor product, so one source covers any k-way interaction
by listing K already-attached `derived_features` names. The R reference's
within-env / weather-modulated interaction kernels reduce to
specific k=2 cases:

| Components | R-reference kernel |
|---|---|
| `[genomic_add, env_onehot]` | KG_GE_A |
| `[genomic_dom, env_onehot]` | KG_GE_D |
| `[phenomic_relmat, env_onehot]` | KP_PE |
| `[genomic_add, weather_fpc_scores]` | KG_GE_AW |
| `[genomic_dom, weather_fpc_scores]` | KG_GE_DW |
| `[phenomic_relmat, weather_fpc_scores]` | KP_PW |

Three-way and higher interactions (e.g. GxExW) are expressible as
`components: [genomic_add, env_onehot, weather_fpc_scores]` — the
processor's complexity is invariant in k.

```yaml
interaction:
  enabled: true
  center_kernel: false                  # BGLR-parity (textbook is true)
  sources:
    - name: GxE_a
      type: interaction_relmat          # eigen-projected
      components: [genomic_add, env_onehot]
      fit_scope: all                    # train | all (transductive); *_rows rejects all
      n_components_max: 1180            # soft rank ceiling
    - name: PxE
      type: interaction_relmat_rows     # raw centered kernel rows
      components: [phenomic_relmat, env_onehot]
```

Source types:
- `interaction_relmat` — eigen-projected kernel (analog of
  `phenomic_relmat`).
- `interaction_relmat_rows` — raw centered kernel rows (analog of
  `phenomic_relmat_rows`).

Cache invalidation rides on content hashes of the per-component
**train feature stacks**, so any upstream processor cache change
automatically propagates downstream — no orchestrator-handle
plumbing required.

For now: linear component kernels only. The Hadamard product is PSD
(Schur product theorem) so the kernel is mathematically valid for
any kernel choice per component, but the tensor-product identity
only holds for linear kernels — keeping it linear keeps the math
clean and the auto-construction trivial.

## Processor API contract

Every registered processor implements `BaseProcessor`
(`utils/data/processing/base.py`). The lifecycle is uniform — the
orchestrator dispatches to every processor through the same calls:

```python
class SomeProcessor(BaseProcessor):
    # ── ClassVar metadata (drives registry queries) ──
    name: ClassVar[str]                                 # required
    priority: ClassVar[int]                             # required
    requires: ClassVar[tuple[str, ...]] = ()
    produces_views: ClassVar[tuple[str, ...]] = ()      # e.g. ("weather",)
    produces_batch_fields: ClassVar[tuple[str, ...]] = ()  # legacy, vestigial
    has_eigen_sources: ClassVar[bool] = False
    eigen_source_types: ClassVar[frozenset[str]] = frozenset()

    # ── Lifecycle (every processor implements these) ──
    @classmethod
    def from_config(cls, config: dict) -> "SomeProcessor":
        """Hydra-config factory used by the orchestrator."""

    def fit(self, ctx: OrchestratorContext) -> None:
        """Learn representations; `ctx` exposes splits, metadata,
        upstream processor handles."""

    def transform(self, ctx, split: str) -> dict[str, np.ndarray] | None:
        """Project the named split using fitted state. Return per-source
        features dict, or `None` for in-place processors (orchestrator
        skips the attach step)."""

    def cache_key(self, ctx) -> str:
        """Content-addressed key covering every fitted-state input.
        Excludes post-load slice params like `n_components`."""

    # ── Optional persistence (default no-ops) ──
    def save_cache(self, cache_dir, key) -> None: ...
    def load_fitted(self, cache_dir, key) -> None: ...

    # ── Reporting ──
    @property
    def feature_dims(self) -> dict[str, int]: ...    # default {}

    # ── Coverage filter (classmethod — runs before fit) ──
    @classmethod
    def coverage_mask(
        cls, metadata_df, proc_config,
    ) -> np.ndarray | None:
        """Return a boolean keep_mask, or `None` for no filtering."""
```

`OrchestratorContext` (passed to every lifecycle method) exposes typed
helpers — `ctx.split_data(split)`, `ctx.metadata_df`, `ctx.split_indices`,
`ctx.env_year()`, `ctx.envs_for_split(split)`, `ctx.pedigrees_for_split(split)`,
`ctx.stack_derived_feature(name, split)` — so processors don't reach
into dataset internals.

**Two output styles:**
- **Returns-features-dict** (`metadata_features`, `vi_fpca`, `phenomic`,
  `weather_fpca`, `weather_kernel`, `genomic`, `enviromic`, `interaction`):
  `transform(ctx, split)` returns `{source_name: array(N, dim)}` and the
  orchestrator attaches it to each sample's `derived_features`.
- **In-place** (`axis_source`, `raw_vi`, `weather_concat`, `raw_weather`):
  `transform(ctx, split)` returns `None` after mutating sample dicts
  (attaching `gdd` / `agdd` axes, slicing or widening the `channels`
  matrix, or attaching `weather_dap` / `weather_values` /
  `weather_channel_names` fields).

**`vi_fpca` / `weather_fpca` R-subprocess optimisation:**
Their `fit(ctx)` internally calls `fit_and_transform_all(...)` which
runs R once on train+val+test together (one subprocess instead of
three). `transform(ctx, split)` then returns the pre-computed scores
for the requested split. After `load_fitted` (predict mode), the same
`transform(ctx, split)` re-runs R per split.

## Cache layout

Every processor writes under `{parent_of_data_dir}/.cache/` — i.e. the
`.cache/` directory sits next to the CSV data directory rather than
inside it — or under the orchestrator's `cache_dir` if explicitly set.
Cache entries use the first 16 hex chars of the processor's
content-addressed key as the filename tag so multiple configs coexist:

```
{cache_dir}/
├── vi_fpca/
│   └── <key[:16]>/
│       ├── fpc_scores.npz       # raw all-VIs × all-components scores (hash-ordered)
│       ├── fpca_models.rds      # R fdapace fitted objects (for predict mode)
│       ├── metadata.json        # max_k, n_vis, vi_names_full, per-split sample counts
│       └── fingerprint          # key includes fit_vars when restricted; vi_subset / n_components are post-load slices, not stored
├── weather_fpca/
│   └── <key[:16]>/
│       ├── fpc_scores.npz       # per-env FIT-SET scores [n_envs, n_fit_vars*max_k] + env_names
│       ├── metadata.json        # max_k, var_names_full (fit set), train_envs, fit_scope, fit_vars
│       └── fingerprint          # key includes fit_vars when restricted; emit_vars / n_components / scaling are post-load slices, not stored
├── metadata_features_<key[:16]>.npz  # one-hot vocab(s) for the metadata source(s)
├── weather_kernel_<key[:16]>.npz  # all/fit env orders, env score matrix, config_json, per-scaling kernel+eigen state
├── genomic_<key[:16]>.npz       # allele_freq, train_dosage, grm_*_{V,D,col_means,grand_mean}, pca_{U,S,Vt}
├── phenomic_<key[:16]>.npz      # per-scaling train_scores_scaled, col_means, grand_mean, K_centered, V, D
├── enviromic_<key[:16]>.npz     # env_order, agg_matrix, col_means, grand_mean, K_centered, V, D
├── interaction_<key[:16]>.npz   # Hadamard-kernel eigendecomp (V, D) per interaction source
├── weather_concat_<key[:16]>.npz  # per-env (daps, values) + var_names
└── raw_weather_<key[:16]>.npz     # per-env (daps, values) + var_names + norm stats
```

The cache key derivation excludes post-load slice parameters
(`n_components` for FPCA / eigen sources; `weather_fpca.emit_vars`,
`vi_fpca.vi_subset`) so a K-sweep or emit-subset change does not
trigger recomputation. Any change to a fitted-state input — train
pedigree list, dosage CSV mtime, weather curve content, train env list,
or the `weather_fpca.fit_vars` / `vi_fpca.fit_vars` *fit sets* (what's
actually computed) —
produces a new key and a fresh cache entry.

**Parallel-safe writes.** When multiple SLURM jobs run concurrently
(e.g. per-fold parallel submission), several jobs may compute the same
cache key if their train splits are identical. All cache writes —
processor-level `save_cache`, the raw genomic dosage cache in
`load_genomic_csv`, and the top-level dataset cache (`DataReader.load`
→ `g2f_data_<key>.pt`, whose key is split-independent so *every* job
shares it) — use a PID-tagged temporary file (`<path>.tmp.<pid>`) and
atomically rename it to the final path via `os.replace`. If the rename
fails because a sibling process already placed the final file, the
error is silently ignored — the cache is warm regardless of which
process won the race. Without atomic writes, a concurrent reader can
see a partially-written file: a truncated `.npz` fails with
`BadZipFile` (the zip central directory is written last, so a truncated
file has invalid magic bytes), and a truncated `torch.save` `.pt`
likewise fails to load.

## Config contract

Every processor is enabled via `dataset.processing.<name>.enabled: true`
in the composed config. `conf/data/g2f.yaml` enumerates all 12 processors
with `enabled: false` defaults plus processor-specific parameters.
Model fragments override individual blocks — see configs under
`conf/model/dl/` and `conf/model/fpca/` for examples of how each
processor is enabled and parameterized per model.

## Validator integration

`utils/experiment/setup.py::_validate_config` runs after
`FeatureProcessor.fit_transform` / `load_and_transform` and before model
instantiation. It enforces one invariant tied to the processing layer:
kernel eigen sources may not declare inconsistent `n_components` /
`n_components_max` (the D1 rank-consistency check, shared with the FPCA
path via `rank_check.py`).

View ↔ processor consistency is handled upstream of the validator by the
dataset itself: `G2FDataset._filter_views_to_enabled_processors` drops any
configured View whose owning processor is absent or disabled (e.g. the
`weather` View needs `raw_weather`). The owner map is queried from
`registry.view_owners()` — each processor's `produces_views` ClassVar
drives this. Adding a new processor that backs a View requires no edit
to the dataset. (The legacy `produces_batch_fields` /
`batch_field_owners()` mechanism still exists but is now vestigial —
every processor declares `()` and backs Views instead.)

The set-encoder peer dims are **data-driven**, not validated.
`_inject_set_encoder_ydims` (run right after validation) overwrites the VI
`params.y_dim` from
`proc.effective_y_dim` (VI subset + any weather_concat tail). Every
set-encoder peer's coord/channel dims — the VI `x_dim`/`y_dim` and the
weather peer's `weather_x_dim`/`weather_y_dim` — are then filled from each
peer's **assembled view** by `_inject_view_dims`: the weather
peer is a real `dataset.views` entry, so its dims come from the assembled
weather view, not a dedicated injector branch. So a stale model-config
y-dim — e.g. the g2f default 37 under `vi_subset=ngrdi` — is corrected,
not rejected; the subset/view config is the single source of truth.
Finally, `_inject_np_geno_dim` fills the TNP tokenizer's geno-MLP input
width (`params.geno_in_dim`) from the configured `geno_key` feature widths
(a no-op for models without that param).

Like the view filter, the **D1 rank-consistency check** in
`utils/data/processing/rank_check.py` is registry-driven: it queries
`registry.names_with_eigen_sources()` so any processor declaring
`has_eigen_sources: ClassVar[bool] = True` is automatically covered.
And `G2FDataset._filter_to_modality_coverage` iterates the registry
calling each processor's `coverage_mask(metadata_df, proc_config)`
classmethod — adding a new partial-coverage modality means overriding
that classmethod, no edit to `dataset.py`.

These checks fire only when a processor is actually attached — vanilla
VI-only configs (no `dataset.processing` block, or all processors
disabled) skip validation entirely.

## Cross-cutting checks at a glance

| Check | Source of truth | Adding a new processor requires |
|---|---|---|
| Execution priority | `cls.priority` ClassVar | declare it on the class |
| Inter-processor deps | `cls.requires` ClassVar | declare it on the class |
| D1 rank consistency | `cls.has_eigen_sources` + `cls.eigen_source_types` ClassVars | declare them on the class |
| View backing filter | `cls.produces_views` ClassVar (via `view_owners()`) | declare it on the class |
| Modality coverage filter | `cls.coverage_mask(metadata_df, proc_config)` classmethod | override it |

All five queries route through `utils/data/processing/registry.py`.
The orchestrator, validator, rank check, and dataset coverage filter
contain **no hardcoded processor names** — they iterate the registry.

