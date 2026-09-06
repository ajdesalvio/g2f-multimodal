# FPCA + Regression Baselines

Classical statistical baselines for crop yield prediction using Functional Principal Component Analysis (FPCA) to extract features from vegetation index (VI) curves, followed by regression to predict yield.

**Status:** Implemented and validated end-to-end via smoke test

## Why FPCA?

The VI curves are on **irregular, per-sample grids** — different environments have different measurement schedules, so samples have different numbers of timepoints at different DAPs. Standard PCA on the data matrix is not applicable. Sparse FPCA (R's `fdapace`) handles different observation grids per sample with smoothed covariance estimation and GCV bandwidth selection.

## How FPC Scores Are Computed

FPCA models (mean function, eigenfunctions, noise variance) are **fitted on training data only** using R's `fdapace::FPCA()`. FPC scores for the test split are then computed by projecting the test curves onto the learned basis via `fdapace::predict()`, which uses the Conditional Expectation (CE/BLUP) formula. The functional basis is never influenced by test curves.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `fpca_compute.R` | R script: sparse FPCA via fdapace (invoked by `VIFPCAProcessor`) |
| `fpca_core.py / fpca_train.py` | CLI entry point: thin wrapper over `FeatureProcessor` |
| `fpca_predict.py` | predict-mode shim (delegates to `fpca_core.predict`; currently raises `NotImplementedError`) |
| `regressors.py` | sklearn regressor registry: factory functions for each method |
| `bglr_regressor.py` | `BGLRRegressor` — multi-kernel RKHS via the `bglr_compute.R` subprocess |
| `bglr_compute.R` | R script: BGLR multi-ETA RKHS fit (invoked by `BGLRRegressor`) |
| `bglr_rank1.patch` | one-line BGLR source patch for rank-1 RKHS kernels (see `R_HPC_SETUP.md`) |
| `gblup_regressor.py` | `GBLUPRegressor` — frequentist twin of BGLR: multi-kernel REML/BLUP via the `gblup_compute.R` subprocess |
| `gblup_compute.R` | R script: REML variance components via `sommer` + explicit BLUP/kriging (invoked by `GBLUPRegressor`) |
| `README.md` | this document |
| `R_HPC_SETUP.md` | R install guide (HPC module stack, `R_LIBS_USER`, BGLR rank-1 patch) |

Feature extraction (VI FPCA, phenomic/genomic/enviromic kernels, weather FPCA)
now lives under `utils/data/processing/` and is orchestrated by
`FeatureProcessor`. The baseline script just consumes `derived_features`
from the enriched `G2FDataset`.

### Config files (`conf/experiment/fpca/`)

Each config is fully self-contained — every eigen K is baked in, no
`???` fields. The filename is the facet-free peer mnemonic (same peers as
the resolved `model_name:`, including the env `eid` token, but in legacy
listing order and without the `.key=value` facets); the **artifact directory**
is named by the resolved `model_name:` (+ `hp_tag`), not the filename.

All current FPCA baselines belong to the **R-mirror modality-ablation
family** — multi-kernel RKHS models that combine genomic / phenomic /
enviromic relationship matrices into one stacked feature vector. The
historical raw-VI family (`fpca_*_k*`
and `fpca_weather_fpca_*`) was removed — the R-mirror family is
strictly more principled (pre-computed eigen-projected kernels on VI
FPC scores rather than raw stacked FPC scores into a regressor).
Naming follows the token grammar below. This is where new FPCA
baselines should go.

See `conf/experiment/fpca/fpca_*.yaml` for the full list.

#### Modality-ablation family (R-mirror) — token grammar

> **Unified per-encoder grammar (current).** DL and FPCA slugs now share one
> grammar. Each peer is `<name>` optionally followed by `.key=value` facets,
> peers alphabetically sorted, joined by `_`. The VI and weather encoders use
> the **same** `vi` / `wthr` names in both families and carry their time axis
> and channel set inline: `vi.t=<time>.c=<channels>` (and `wthr.t=…c=…`), where
> `channels` = subset (`full`/`ngrdi`/`ptr`) plus any `+aug` (e.g. `+gdd+agdd`).
> Facets: `t` time, `c` channels, `k` upstream FPC count, `s` scaling
> (`gz`/`wez`; omit `none`), `rank` eigen rank (int / `full`), `.uc` uncentered,
> `.kern=rbf` (omit linear), `.rows` (DL GRM-row peers). The `vi.t`/`vi.c`
> pieces come from the `axis` / `vi_subset` / `weather_subset` groups; the
> kernel facets are model-intrinsic. `model_slug = model_name + hp_tag` — there
> are no separate `_vis=`/`_ws=`/`_axis=` tags any more.
>
> **Vocabulary renames** (kill the old `w`/`e`/`p`/`v` overloading):
> `p`/`v` (VI phenomic/raw) → **`vi`**; `wf` (weather sparse FPCA) → **`wthr`**;
> `w` (enviromic K_W) → **`enviro`**; env-identity one-hot `e` → **`eid`**;
> interactions `gxea`/`gxed`→`gaXeid`/`gdXeid`, `gxwa`/`gxwd`→`gaXwthr`/`gdXwthr`,
> `pxe`→`viXeid`, `pxw`→`viXwthr`; DL GRM rows `gar`/`gdr`→`ga.rows`/`gd.rows`;
> FPCA genomic `gae`/`gde`→`ga.rank=…`/`gd.rank=…`. The standalone env one-hot
> **main effect** (BRR `Z_e`, or a stacked `env_onehot` feature) renders as a
> bare **`eid`** peer; alpha-sorted it leads the slug (`fpca_eid_…`). It is
> distinct from the `*Xeid` interaction peers (which only mark `env_onehot` as
> an interaction *partner*): a model can carry `eid` alone (env-GxE/weather-GxE, BRR `Z_e`),
> the `*Xeid` interactions alone, or both.
>
> Worked examples — DL: `istnp_L4_H4x16_D64_FF64_ist.k256_vi.enc=tfm.d64.L2.h4_ga.rows_gd.rows_vi.t=dap.c=full.tsub=0.75-1.0_loss=nll`.
> FPCA weather-GxE (agdd, with env `Z_e`): `fpca_eid_ga.rank=full.uc_gaXwthr.rank=full_gd.rank=full.uc_gdXwthr.rank=full_vi.t=agdd.c=ngrdi.k=5.s=wez.rank=full.uc_viXwthr.rank=full_wthr.t=agdd.c=ptr.k=1.s=gz.rank=full.uc_bglr_loss=mse`.
> The env-free counterpart drops the leading `eid_`.
>
> The legacy compact token reference below documents the *concepts* (kernel vs
> raw, scaling, centering, rank, regressor pairing) — these are unchanged; only
> the surface spelling moved to the `key=value` form above. FPCA config
> *filenames* are the facet-free peer mnemonic (e.g.
> `fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_bglr_mse`): same peers as the
> resolved `model_name`, in the legacy listing order and without the `.key=value`
> facets. They now carry the env `eid` main-effect token too — an env-bearing
> fragment is `fpca_eid_…`, an env-free one (main-effects, or the env-free weather-GxE) is not.

Quick reference (legacy compact form — concepts only):

```
fpca_<feat_codes>_<reg>_mse
```

where `<feat_codes>` is a sorted list of per-modality tokens (each
carrying its own eigen K and kernel knobs inline), and `<reg>` is a
short regressor code.

**Peer registry** — one entry per feature source. (Renamed from the
pre-unified spellings to kill the `w`/`e`/`p`/`v` overloading: `p`/`v`→`vi`,
`wf`→`wthr`, `w`→`enviro`, env-onehot `e`→`eid`, `gxe*`→`*Xeid`,
`gxw*`/`pxw`→`*Xwthr`.)

| Peer | Modality | Processor |
| ---- | -------- | --------- |
| `ga` / `gd` | genomic additive (K_A, VanRaden) / dominance (K_D, Vitezica) | `GenomicFeatureBuilder` |
| `gr` | genomic raw PCA (centered SNP dosages, full-rank SVD) | `GenomicFeatureBuilder` |
| `gdos` | genomic raw SNP dosage (literal markers; no kernel/PCA) | `GenomicFeatureBuilder` |
| `vi` | VI encoder — phenomic kernel KP from vi_fpca scores (`.rank=` present), **or** raw VI FPC scores (no `.rank=`) | `PhenomicFeatureBuilder` / `VIFPCAProcessor` |
| `wthr` | weather encoder — per-variable sparse FPCA on raw curves, **or** the env-level K_W kernel from those scores | `WeatherFPCAProcessor` / `WeatherKernelProcessor` |
| `enviro` | enviromic K_W from aggregated weather summaries | `EnviromicFeatureBuilder` |
| `eid` | environment-identity one-hot — as a **standalone main-effect peer** when the env one-hot enters the model directly (BGLR/GBLUP `env_brr_key: env_onehot`, or `env_onehot` listed in a regressor's `feature_keys`), **and** as the interaction partner inside `*Xeid` names. The main-effect peer is bare (no facets) and alpha-sorts first (`fpca_eid_ga…`). Env-free fragments (main-effects, and the env-free weather-interaction weather-GxE) set `env_brr_key: null`, omit `env_onehot`, and therefore carry **no** `eid` token — that absence (the bare `fpca_ga…` stem vs the env-bearing `fpca_eid_ga…`) is what makes an env-free slug collision-safe against its env-bearing twin, so no extra suffix is needed. | `metadata_features` |
| `gaXeid` / `gdXeid` | G_a/G_d × env-identity Hadamard kernel (the R reference's KG_GE_A / _D) | `InteractionKernelBuilder` |
| `gaXwthr` / `gdXwthr` | G_a/G_d × weather Hadamard kernel (KG_GE_AW / _DW) | `InteractionKernelBuilder` |
| `viXeid` / `viXwthr` | phenomic × env-identity / × weather Hadamard kernel (KP_PE / KP_PW) | `InteractionKernelBuilder` |

The six `*X*` interaction peers reproduce the R reference's same-env and
weather-modulated interaction kernels. Each maps to one source in the
`interaction` processor with a 2-element `components` list — e.g. `gaXeid`
↔ `components: [genomic_add, env_onehot]` (the `env_onehot` upstream feature
comes from `metadata_features`). Three-way and higher interactions follow the
same shape with a longer `components` list and a longer compound peer name
(e.g. `gaXeidXwthr`); those aren't built today. (Source *names* inside the
`interaction` config — `gxea`, `gxwa`, … — are unchanged internal identifiers;
only the slug spelling moved to `gaXeid`, `gaXwthr`, ….)

`ga`, `gd`, `gr`, and `gdos` all use `GenomicFeatureBuilder` but
produce distinct feature constructions from the same input SNP dosage
matrix: `ga` eigen-projects the VanRaden additive GRM, `gd`
eigen-projects the Vitezica dominance GRM, `gr` does full-rank
SVD on the centered SNP matrix (equivalent to min-norm OLS on raw
SNP dosages, up to a basis rotation), and `gdos` passes the literal
SNP dosage matrix through with no kernel and no PCA (`type: raw_dosage`).
For OLS, `gr` preserves more information than `ga`+`gd` combined and
collapses the add/dominance distinction because raw dosages contain
both; `gdos` carries the same information uncompressed (width = SNP
count, ~239k for G2F), so it was intended for a DL peer-MLP branch
rather than the FPCA regressors (that DL analog has since been
retired).

**"Kernel vs raw-FPC"** — the same upstream processor produces either an
eigen-projected kernel or raw FPC scores; the two are **not interchangeable**
between regressors. The `.rank=` facet is what distinguishes them:

- **VI FPC scores** (`vi` peer, built on `vi_fpca`):
  - `vi.k=<K>.rank=<eigK>` builds the phenomic relationship matrix KP and
    eigen-projects it (smooth, low-dim) — the kernel form.
  - `vi.k=<K>` (no `.rank=`) uses the raw FPC scores directly (shape `[N, 37*K]`).
- **Weather** — two distinct pipelines:
  - `enviro` (`EnviromicFeatureBuilder`): aggregates weather (mean-over-time
    across vars) → flat per-env vector → linear kernel across envs → eigen.
  - `wthr` (`WeatherFPCAProcessor` / `WeatherKernelProcessor`): per-variable
    sparse FPCA on the raw daily curves → top-K FPC scores; `wthr.k=<K>` is the
    raw scores, `wthr.k=<K>.rank=<eigK>` the env-level K_W kernel from them.

**Which form pairs with which regressor:**

- **`ols` / ordinary least squares** works best with the raw-FPC forms
  (`vi.k=..` and `wthr.k=..`, no `.rank=`). OLS has no regularizer that
  benefits from a smoothness prior, and the raw FPC representation preserves
  more information at higher dimensionality (fine because train splits are
  large enough to stay well-conditioned).
- **`bglr`** pairs with the eigen-projected `*_relmat` tokens
  (the kernel-projection acts as a functional smoothness prior):
  (a) gets a separate variance component σ²_k per kernel via the
  multi-ETA RKHS construction — the R reference's per-kernel variances
  recovered exactly; and
  (b) consumes each `*_relmat` block at full filtered eigen-rank
  (no `n_components` truncation, except `weather_fpca` which stays
  at the FPCA-reduced rank the R reference uses upstream of BGLR).
  Each raw obs-level N×N kernel block is passed to `bglr_compute.R`,
  which eigen-decomposes it (`eigen(K, symmetric=TRUE)`) and hands BGLR
  `list(V=vectors, d=values, model='RKHS')` — matching the R reference;
  the env block uses `list(X=Z_e, model='BRR')`, appended last. See the
  architecture §"BGLR" for the kernel-rank semantics under BGLR's RKHS
  filter.

This "regressor chooses representation" rule is why a `bglr` config and an
`ols` config for the same underlying modality would use structurally
different tokens. Not a typo — the difference is load-bearing for the
comparison to be fair. (The shipped set is `bglr`/`gblup` plus the weather-GxE
`ols` endpoint; see the "Shipped configs" note under Training.)

**Facet grammar.** Each peer is `<name>` followed by `.key=value` facets, in
this order:

```
<peer>[.t=<axis>][.c=<channels>][.k=<viK>][.s=<scaling>][.kern=rbf][.rank=<K>][.uc]
```

| Facet | Meaning |
| ----- | ------- |
| `.t=<axis>` | time coordinate(s) of a set encoder: `dap`, `agdd`, `dap+agdd`, … (VI/weather peers) |
| `.c=<channels>` | channel set = subset (`full`/`ngrdi`/`ptr`) + any `+aug` (e.g. `+gdd+agdd`) (VI/weather peers) |
| `.k=<viK>` | upstream FPC count — `vi_fpca` / `weather_fpca` `n_components` |
| `.s=<scaling>` | score scaling, `gz` / `wez` (omit `none`); see registry below |
| `.kern=rbf` | kernel type (omit linear, the default) |
| `.rank=<K>` | eigen rank — integer or `full`. **Absent ⇒ raw scores, no kernel step.** |
| `.uc` | uncentered kernel (BGLR-parity; processor `center_kernel: false`) |
| `.rows` | raw kernel/GRM rows — no eigen-projection (DL row-MLP peers `ga.rows`/`gd.rows`) |

`.t=`/`.c=` are supplied by the `axis` / `vi_subset` / `weather_subset` groups;
the rest are model-intrinsic. `.uc` mirrors the BGLR reference passing
unmodified `K_A`/`K_D`/`K_P`/`K_W` into RKHS terms; for `ga`/`gd` it is a
documented no-op (VanRaden `K_A` rows already sum to zero, so double-centering
is an identity). Per-peer applicability:

| Peer | `.t`/`.c` | `.k` | `.rank` | kernel |
| ---- | --------- | ---- | ------- | ------ |
| `ga` / `gd` | — | — | `=<K>`/`=full` (FPCA) **or** `.rows` (DL) | fixed |
| `gr` | — | — | `=<K>`/`=full` — SVD rank (`full` = all ~1180) | n/a |
| `gdos` | — | — | — (fixed width = SNP count) | n/a |
| `vi` | yes | yes | present ⇒ phenomic kernel; absent ⇒ raw scores | linear/`rbf` |
| `wthr` | yes | yes | present ⇒ K_W kernel; absent ⇒ raw scores | linear/`rbf` |
| `enviro` | — | — | `=<K>`/`=full` or `.rows` | linear |
| `gaXeid`/`gaXwthr`/`viXeid`/`viXwthr`/… | — | — | `=full` or `=<K>` | linear, fixed |

Interaction peers take `.rank=full`/`.rank=<K>` like the kernels they're built
from (`interaction.center_kernel: false` is the BGLR-parity default). See
[`utils/data/processing/README.md`](../utils/data/processing/README.md)
for the per-processor source spec.

**Scaling registry** (per-source `scaling` knob in
`dataset.processing.phenomic.sources[*].scaling` and analogous fields
on `enviromic.sources[*]`; `weather_fpca.scaling` is a single
processor-level knob since that processor emits one feature). Single
source of truth lives at
`utils/data/processing/score_scalers._SCALING_REGISTRY`; the slug
grammar mirrors it 1:1.

| Config value (`scaling:`) | `.s=` code | Scope | Applies to |
| ------------------------- | ---------- | ----- | ---------- |
| `none` *(default)* | *(omitted)* | — | all score consumers |
| `global_zscore` | `gz` | global (fitting-set envs only — train under `fit_scope=train`, all under `fit_scope=all`) | `vi` (per-pedigree-env), `wthr`, `enviro` (env-level) |
| `within_env_zscore` | `wez` | per-env (transductive) | `vi` only — **rejected at config time** for env-level `wthr`/`enviro` (one row per env makes within-env std degenerate) |
| `global_maxnorm` *(reserved)* | `gm` | global | `vi`, `wthr`, `enviro` |
| `within_env_maxnorm` *(reserved)* | `wem` | per-env | `vi` only |

Scope tokens (`g`, `we`) and method letters (`z`, `m`, `u`) compose freely —
a new `<scope><method>` pair is one registry entry, no grammar change.
Env-level peers (`wthr`, `enviro`) enforce an `_ALLOWED_SCALINGS` set so a
`wez` request fails fast rather than silently producing zeros.

- Raw VI scores: `vi.k=<K>` with no `.rank=` — the raw output of
  `vi_fpca.fit_and_transform_all`, shape `[N, 37*K]`. E.g. `vi.k=4` is K=4
  raw FPC scores (148 features).
- Raw weather scores: `wthr.k=<K>` with no `.rank=` — per-variable sparse
  FPCA scores from `WeatherFPCAProcessor`, concatenated. The shipped configs
  use the PTR-only / 5-var core subset (set via `weather_subset` → the `.c=`
  channel token); a different variable set is a distinct `.c=` and thus a
  distinct slug, per the collision-safety contract.
- `.kern=rbf` selects the RBF kernel; linear is the omitted default.
- `.rank=` is an integer or `full` (full-rank eigen-projection).
- Peers are sorted alphabetically within a slug.

**Regressor short codes**:

| Short  | sklearn / registry | Role                                |
| ------ | ------------------ | ----------------------------------- |
| `bglr` | `bglr`             | BGLR multi-kernel Bayesian RKHS via R subprocess (transductive — see architecture §"BGLR" below) |
| `gblup`| `gblup`            | Multi-kernel REML/BLUP via `sommer` R subprocess — the frequentist twin of `bglr` (same kernel stack, transductive) |
| `ols`  | `ols`              | OLS — unregularized frequentist ablation |

`bglr` is wired through `cv/slug_util.py` the same way every other
short code is: filename = `model_name` = slug. There is no
regressor-short-code map to update.

**Concrete token examples:**

(`.t=`/`.c=` on `vi`/`wthr` come from the `axis` / `vi_subset` / `weather_subset`
groups; shown here with representative values.)

| Intent                                                         | Peer token  |
| -------------------------------------------------------------- | ----------- |
| Genomic additive / dominance, eigen full (VanRaden / Vitezica, linear) | `ga.rank=full` / `gd.rank=full` |
| Genomic additive **rows** (DL peer-MLP)                        | `ga.rows`   |
| Genomic raw PCA, full-rank (~1180 features for G2F)            | `gr.rank=full` |
| Genomic raw PCA, top-100 SVD components                        | `gr.rank=100` |
| Genomic raw SNP dosage, literal markers (no kernel/PCA, ~239k features) | `gdos` |
| Phenomic, vi K=4, KP eigen full, linear                        | `vi.t=dap.c=full.k=4.rank=full` |
| Phenomic, vi K=4, KP eigen K=20, RBF                           | `vi.t=dap.c=full.k=4.rank=20.kern=rbf` |
| Phenomic with **within-env z-score**, eigen full, uncentered (reference-style) | `vi.t=dap.c=ngrdi.k=5.s=wez.rank=full.uc` |
| Raw VI FPC scores, K=4 (no kernel; 37×4 = 148 features)        | `vi.t=dap.c=full.k=4` |
| Raw VI FPC scores, K=4, **within-env z-scored**                | `vi.t=dap.c=full.k=4.s=wez` |
| Enviromic kernel, **global-z** aggregates, linear, eigen K=17 (reference-style) | `enviro.s=gz.rank=17` |
| Enviromic kernel rows / uncentered                             | `enviro.s=gz.rows` / `enviro.rank=17.uc` |
| Weather sparse FPCA scores, K=4, **global-z** (no kernel)      | `wthr.t=dap.c=ptr.k=4.s=gz` |
| Weather **kernelized** K_W from PTR (1 FPC), global-z, full eigen, uncentered (weather-GxE recipe — K=1 on V2) | `wthr.t=dap.c=ptr.k=1.s=gz.rank=full.uc` |
| Weather kernelized K_W rows                                    | `wthr.t=dap.c=ptr.k=4.s=gz.rows` |
| G_a/G_d × env-identity Hadamard kernel, full eigen (KG_GE_A/_D) | `gaXeid.rank=full` / `gdXeid.rank=full` |
| G_a/G_d × weather Hadamard kernel, full eigen (KG_GE_AW/_DW)   | `gaXwthr.rank=full` / `gdXwthr.rank=full` |
| Phenomic × env-identity / × weather Hadamard kernel (KP_PE / KP_PW) | `viXeid.rank=full` / `viXwthr.rank=full` |

**Modality-ablation slug roadmap.** The full design surface is a
curated factorial over `{G, p, w}` where G is treated as one modality
with two sub-representations (`ga` alone vs `ga+gd`). Each modality
subset produces both a kernel-projected slug (for `bglr`/`gblup`) and an
`ols` slug (raw FPC features) following the token grammar above. The
`gr` raw-genomic representation is an OLS-specific alternative that
collapses the additive/dominance distinction.

To see which configs are currently built, list the experiment
directory:

```bash
ls conf/experiment/fpca/
```

Subsets 4 (`enviro`), 8 (`ga+enviro`), and 9 (`ga+gd+enviro`) were dropped —
pure weather and weather+genomics-without-phenomic are not meaningful
ablation points for this dataset. Every remaining `enviro`-bearing subset
also carries phenomic. The LOO(Env, Year) caveat drops the R same-env and
weather-modulated interaction models.

**Why `enviro.rank=17` and not `rank=18`.** With 19 (Env, Year) combinations
and mean-centering the kernel, the enviromic similarity matrix has rank 17 in
this dataset, not 18. The `.rank=` facet encodes the **actual** eigen count
used at fit time, so two different requests never clamp to the same effective
K under the hood.

**Invariants** (not encoded in the slug — document any deviation as an extra
facet, e.g. `ga.rank=20.kern=rbf` or `vi.…rank=full.kern=rbf`):
- `genomic.sources[*]` uses VanRaden / Vitezica (no kernel knob)
- `genomic.fit_scope = all` — GRM fitted on train ∪ val ∪ test
  pedigrees. Genotype is observed pre-planting (no leakage), same
  rationale as `enviromic` / `weather_fpca`. Configs that flip to
  `genomic.fit_scope: train` should carry a `_train_scope` slug
  suffix to stay collision-safe under the contract above.
- `phenomic.kernel = linear`, `gamma = null`
- `enviromic.sources[0].kernel = linear`, `gamma = null`,
  `aggregation = mean`, `fit_scope = all`

Available regressors (the full registry, not just short codes):
`ols`, `gpr`, `bglr`, `gblup`.
Available vi_fpca K in use: 5 for every modality-ablation config in
this wave (the removed raw-VI family used 4, 6, 8).

#### DL slug grammar (raw-feature family)

DL configs under `conf/experiment/dl/` use the **same unified grammar** as
FPCA (see the callout at the top of this section). The shape of a DL slug is:

```
istnp_<archparams>_<peers...>_loss=<loss>
```

where the slug-prefix `istnp` (induced-set Transformer Neural Process,
model class `TransformerNeuralProcess`) names the model family,
`<archparams>` is the architecture block — the cross-sample IST transformer
dims (`L`/`H`x/`D`/`FF`), the induced-point count (`ist.k<K>`), and the
per-modality sub-encoder specs (`vi.enc=tfm.d<..>.L<..>.h<..>`, plus
`wthr.enc=tfm.…` when a weather encoder is present) — and
`<peers...>` is the alphabetically-sorted peer list. The VI set encoder is the
`vi` peer (`vi.t=<time>.c=<channels>`); the separate raw-weather set encoder is
the `wthr` peer (`wthr.t=…c=…`); genomic GRM-row peer-MLPs are `ga.rows` /
`gd.rows` (additive / dominance). An istnp slug therefore reads e.g.
`istnp_L4_H4x16_D64_FF64_ist.k256_vi.enc=tfm.d64.L2.h4_wthr.enc=tfm.d32.L1.h2_ga.rows_gd.rows_vi.t=dap.c=full.tsub=0.75-1.0_wthr.t=dap.c=full.tsub=0.5-1.0_loss=nll`.

**DL config *file* names** (the `.yaml` name, distinct from the runtime slug
above) follow: `<family>_<modalities>_<geno>[__<facet>…]__[<size>]__<loss>`.
- `<family>` = `istnp` (induced-set TNP);
  `<modalities>` = `vi` ± `_wthr`; `<geno>` = `ga_gd` / omitted.
- Each later facet is `__`-separated, in order: geno representation
  (`__grm-rows` / `__grm-eig`), per-encoder type (`__vienc-<kind>` /
  `__wthrenc-<kind>`), channel subset (`__vi-<chan>` / `__wthr-<chan>`), time
  axis (`__t-<axis>`), train-time augmentation (`__aug-<modalities>`).
- **Train-time augmentation** (`__aug-<modalities>`, an *experiment*-level facet
  set by the `/augment` group, like the time axis): lists the modalities given
  train-only timepoint-subsampling — `__aug-vi` (VI only) or `__aug-vi-wthr`
  (VI + weather). OMIT it when there is no augmentation (the `none` default),
  mirroring the omit-when-default rule for full channels / native axis / single
  size. The file-name facet only flags *which* modalities are augmented; the
  runtime slug separately carries the keep-fraction ranges (`.tsub=0.75-1.0`).
- **The loss (`__mse` / `__nll`) is ALWAYS the final `__` facet**, and an
  **optional size letter (`__S` / `__L`) immediately precedes it** — include the
  size only to distinguish capacity variants, and OMIT it when a family ships a
  single size (e.g. the istnp configs carry no size). This mirrors the runtime
  slug, which always ends `_loss=<loss>`.

Examples: `istnp_vi__vienc-tfm__aug-vi__nll` (VI only, no genomics),
`istnp_vi_ga_gd__grm-rows__vienc-tfm__aug-vi__nll` (VI + GRM rows, VI aug),
`istnp_vi_wthr_ga_gd__grm-rows__vienc-tfm__wthrenc-tfm__aug-vi-wthr__nll`
(VI + weather + GRM rows, VI + weather aug). (No shipped
config omits augmentation or carries a size letter / explicit time axis; those facets
read illustratively as an omitted `__aug-…`, `…__L__nll`, `…__t-dap-gdd-agdd__nll`.)

The presence of the `vi` / `wthr` peer tokens *is* the signal that each set
encoder exists, so the legacy `vs` / `wthrs` / `vws` arch-block tokens are
gone. The fused VI+weather variant (formerly `vws`, via `weather_concat`) would
widen the VI peer's channels rather than add a `wthr` peer — distinct from the
two-peer `vi … wthr …` separate-branch form. The VI peer (`raw_vi`) and the
weather peer (`raw_weather`) are each variable-subsettable via the `vi_subset`
/ `weather_subset` groups, which set the `vi.c=` / `wthr.c=` channel tokens.

**Auxiliary peer-MLP branches.** Beyond the `vi` / `wthr` set encoders, a DL
config may add fixed-length peer-MLP branches: genomic GRM eigenfeatures
(`ga.rank=<K>` / `gd.rank=<K>`) or GRM rows (`ga.rows` / `gd.rows`) — the same
peers the FPCA family uses. (A raw-SNP-dosage `gdos` DL branch existed
historically but is retired.)

**Weather integration — three strategies**, of which only one is shipped:

1. **Separate set encoder (shipped)** — raw daily weather curves through a
   dedicated set encoder inside the `SampleTokenizer`, peer to the VI
   encoder: the `wthr` peer, e.g.
   `istnp_vi_wthr_ga_gd__grm-rows__vienc-tfm__wthrenc-tfm__aug-vi-wthr__nll` →
   `…_vi.t=dap.c=full…_wthr.t=dap.c=full…_…`
   (`conf/experiment/dl/istnp_vi_wthr_ga_gd__grm-rows__vienc-tfm__wthrenc-tfm__aug-vi-wthr__nll.yaml`).
2. **Weather-FPCA MLP branch** *(not shipped)* — sparse FPCA scores fed to a
   peer MLP; would surface as its own derived-feature peer.
3. **VI fusion** *(not shipped)* — concatenate daily weather onto the VI
   feature vector at VI-observation days, widening the VI peer's `vi.c=`
   channels rather than adding a peer.

No slug token is reserved for (2)/(3) yet; the old `wf` / `ws` / `vws`
spellings are retired.

**Genomic-as-peer-MLP variants** (genomic info fed straight to
a peer MLP instead of an additive/dominance GRM kernel):

| Genomic branch                                  | Config name         | Config                                |
| ----------------------------------------------- | -------------------- | ------------------------------------- |
| K_A + K_D kernel rows                           | `istnp_vi_ga_gd__grm-rows__vienc-tfm__aug-vi__nll`  | `conf/experiment/dl/istnp_vi_ga_gd__grm-rows__vienc-tfm__aug-vi__nll.yaml` |

**Peer ordering.** Under the unified grammar *all* peers — set encoders
(`vi`, `wthr`) and MLP/kernel branches (`ga`, `gd`, …) alike — are
sorted alphabetically with no special-casing (the old "VI set encoder pinned
first" exception is retired). So `istnp_vi_ga_gd__grm-rows__vienc-tfm__aug-vi__nll` resolves to
`istnp_…_ga.rows_gd.rows_vi.t=…` and a weather model interleaves the `wthr` peer
in its alphabetic slot. A single rule keeps sweeps along any axis (genomic,
weather, axis) producing comparable adjacent slugs.

---

## Prerequisites

**R** packages:
```r
install.packages(c("fdapace", "optparse", "BGLR", "jsonlite", "sommer"))
```

`BGLR` and `jsonlite` are required for the `bglr` regressor (see
architecture §"BGLR" below); `sommer` is required for the `gblup`
regressor. Pure-FPCA-and-sklearn baselines run without them.

For HPC environments (module loading, `R_LIBS_USER`, compatible GCC/R versions),
see [`R_HPC_SETUP.md`](R_HPC_SETUP.md) for the install recipe and
[`../environments/hprc-grace/r-packages.txt`](../environments/hprc-grace/r-packages.txt)
for the exact R package versions used on Grace.

**Python** packages (should already be in environment):
```
scikit-learn, scipy, pandas, numpy, joblib
```

See [`../environments/`](../environments/) for exact Python + R version
pins on both HPRC Grace (production) and the local MacBook (dev).

---

## Usage

All configuration (dataset, FPCA params, regressor choice and
hyperparameters, output directory) is specified via Hydra composition
over the `conf/` tree. The training entry point is
`baselines.fpca_train`, a three-line `@hydra.main` shim that composes
the root config and delegates to `baselines.fpca_core.train`. Pick an
experiment via `+experiment=fpca/<slug>`.

### Training

> **Shipped configs.** `conf/experiment/fpca/` currently contains **14**
> pre-composed configs (env-GxE = env-identity interactions `…gaXeid…viXeid…`,
> weather-GxE = weather interactions `…gaXwthr…wthr…`, and G+P main-effects =
> pure main effects `…ga_gd_vi…`, no GxE/env):
> - **`bglr`** ×9 — G+P main-effects, env-GxE, and weather-GxE, each over the default
>   `dap` axis and a `__t-agdd` variant, plus the env-free variants: weather-GxE
>   (`fpca_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_bglr_mse`, `dap` and `__t-agdd`)
>   and env-GxE (`fpca_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse`, `dap`). The
>   env-free variants are the reported weather and weather-free kernel models.
> - **`gblup`** ×4 — the frequentist twin, same env-GxE/weather-GxE × dap/agdd grid.
> - **`ols`** ×1 — inductive regressor over the weather-GxE features,
>   `dap` axis.
>
> Registry entries without a pre-generated config (e.g. `gpr`) can be run
> by authoring a config under `conf/experiment/fpca/` first.

Pick any experiment from `conf/experiment/fpca/`:
```bash
python -m baselines.fpca_train +experiment=fpca/<slug>
```

For example, to run the shipped weather-GxE model:
```bash
python -m baselines.fpca_train \
  +experiment=fpca/fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_bglr_mse
```

### Overriding parameters from CLI

Any key can be overridden with a bare `key=value` argument — Hydra
parses these natively, no `--` prefix. Use `+key=value` to force-add
a key not declared in the schema. Eigen Ks and the regressor choice
are baked into each config (their literal values appear in the slug
itself); override only for ad-hoc exploration:
```bash
python -m baselines.fpca_train +experiment=fpca/fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse \
    misc.seed=42 dataset.random_state=42
```

Sweep across all built FPCA configs with a shell loop:
```bash
for yaml in conf/experiment/fpca/fpca_*.yaml; do
  slug=$(basename "$yaml" .yaml)
  python -m baselines.fpca_train +experiment=fpca/${slug}
done
```

### Feature-subset ablation

`vi_subset` and `weather_subset` are independent ablation axes layered
on top of every experiment composer. Default `full` (no-op); switch
to a non-default group to clobber the model fragment's hardcoded
subset values and set the `vi.c=` / `wthr.c=` channel tokens of the slug.

```bash
# Full features (default)
python -m baselines.fpca_train +experiment=fpca/fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse

# The R-reference env-GxE/weather-GxE reproduction — single VI (NGRDI) + single weather var (PTR)
python -m baselines.fpca_train \
    +experiment=fpca/fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_bglr_mse \
    vi_subset=ngrdi weather_subset=ptr

# Sweep both axes in two seeds
for vi in full ngrdi; do
  for ws in full ptr; do
    for seed in 0 1; do
      python -m baselines.fpca_train \
          +experiment=fpca/fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse \
          vi_subset=${vi} weather_subset=${ws} misc.seed=${seed}
    done
  done
done
```

Adding new ablation points (e.g., a top-5 VI subset): drop a new
`conf/vi_subset/<name>.yaml` setting both `dataset.processing.vi_fpca.vi_subset`
(FPCA score slice) and `dataset.processing.raw_vi.vi_subset` (DL raw-curve
slice) to your list, plus `vi_chan: <name>` (the channel-subset slug piece,
which renders into the `vi.c=<name>` token of `model_name`). Same shape for
`conf/weather_subset/<name>.yaml` (set `wthr_chan: <name>` → `wthr.c=<name>`).
No edits to existing experiment composers needed.

### Inference (predict on new curves)

The `baselines.fpca_predict` entry point (a sibling `@hydra.main` shim
to `fpca_train`, added with the Hydra entry points) currently raises
`NotImplementedError`. DL predict-mode evaluation on a held-out split
works end-to-end via `eval.py` (reads cache keys from
`processing_metadata.json` and calls
`FeatureProcessor.load_and_transform`), and re-implementing the same
path for the FPCA baseline is tracked as follow-up work. For now, the
FPCA baseline is a train-only entry point — test metrics are produced
by the single `fpca_train` invocation, not by a separate predict run.

### Output Artifacts

Artifacts are saved to the directory specified by `misc.artifacts_dir` in the config.

| File | Contents |
|------|----------|
| `regression_model.joblib` | Fitted sklearn regression model |
| `metrics.json` | `meta` (model name, model_slug, seed, regressor, counts) + `by_metric` — one entry per scored `cv_label` (`{n, rmse, pearson_r, spearman_r, r_w, r_w_n_blocks, rho_w, rho_w_n_blocks}`); `env_year_loo` → `test`, `cv_2_1` → `CV1`/`CV2`, `cv_0_00` → `CV0`/`CV00`. `r_w`/`rho_w` are the Tiezzi et al. (2017) within-`Env.Year`-block inverse-variance weighted Pearson/Spearman correlations (computed by the shared `cv.scoring.score_by_label` → `weighted_block_correlation`, identical to the DL path). See [`cv/README.md`](../cv/README.md). |
| `predictions.csv` | Per-row predictions (D8 all-row schema), written just before `metrics.json` |

(Unlike the DL `train.py`, the FPCA baseline writes no separate
`processing_metadata.json` — the enabled processors and feature dims are
recorded under `meta.processing` inside `metrics.json` instead.)

Fitted processor state (R `fpca_models.rds`, genomic/phenomic/enviromic
`.npz` bundles, per-env weather arrays, etc.) lives under the dataset's
`{parent_of_data_dir}/.cache/` directory, content-addressed by SHA-256 cache
keys (the DL `train.py` records these in a `processing_metadata.json` for
predict-mode reuse). See
[`utils/data/processing/README.md`](../utils/data/processing/README.md)
for the full cache layout and the `save_cache` / `load_fitted` contract.

Each processor (VI FPCA, phenomic, genomic, enviromic, weather FPCA,
weather concat, raw weather) writes its fitted state to the dataset's
`.cache/` directory under a processor-specific subpath, content-addressed
by the first 16 hex chars of a SHA-256 key derived from the inputs the
processor fitted on (excluding any post-load slice parameters like
`n_components`). Different configs coexist; changing any fitted-state
input invalidates the cache automatically; `n_components` sweeps share
the same cache entry. See
[`utils/data/processing/README.md`](../utils/data/processing/README.md)
for the per-processor cache layout.

### W&B Logging

**Disabled by default for FPCA baselines.** Every fpca model fragment
under `conf/model/fpca/` overrides
`misc.wandb_logging_enabled: false`. The baseline emits one test
scalar per fold via `metrics.json`, which downstream analysis reads
straight from disk — the wandb side-channel adds no information here, and
the long-slug tag rejection (see Known issue below) made every BGLR
SLURM job exit FAILED until the override landed.

If you do flip the flag back on (`misc.wandb_logging_enabled: true`),
`fpca_core.py / fpca_train.py` calls `wandb.init` after training and
writes the per-`cv_label` metrics flattened as `<label>/<metric>` scalars
(e.g. `test/pearson_r`, or `CV0/rmse` for `cv_0_00`) to `run.summary`,
matching the DL pipeline. No time-series points are logged — the
baseline has no training curves. The run is tagged via
`misc.wandb_group` and `misc.wandb_tags`; for LOO jobs these are
injected by `fpca_job.job`.

`metrics.json` is always written regardless of W&B settings.

> **Known issue — `wandb.init` rejects tags longer than 64 characters.**
> wandb's `Settings.run_tags` pydantic validator caps tag length at 64 characters; the resolved weather-GxE `model_name` (the long per-encoder form, e.g. `fpca_eid_ga.rank=full.uc_gaXwthr.rank=full_..._wthr.t=dap.c=ptr.k=1.s=gz.rank=full.uc_bglr_loss=mse`, ~184 chars) blows past this when passed as a tag (the `fpca_job.job` scripts build `misc.wandb_tags=[...,'${SLUG}',...]` and the BGLR model fragments also list `${model_name}`). **Resolved:** both pipelines now route tags through `utils/experiment/utils.py::sanitize_wandb_tags` (truncate + hash any tag >64 chars) before `wandb.init`, so long slugs no longer crash the run. The `wandb_logging_enabled: false` overrides above keep FPCA W&B off by default regardless.

### All CLI Options

**`python -m baselines.fpca_train`** — Hydra `@hydra.main` shim.
Composes `conf/config.yaml` and delegates to `fpca_core.train`.
Takes bare `key=value` Hydra overrides — no `--` prefix.

- `+experiment=fpca/<slug>` (required) — select a pre-composed
  experiment under `conf/experiment/fpca/`. Every experiment composer
  pulls its own `data` and `model/fpca` fragments (plus the
  `vi_subset` / `weather_subset` / `axis` groups) via its `defaults:`
  list; the CV split comes from the data fragment, or a role-based
  `+cv_spec=<scheme>` override.
- Any `key=value` after the `+experiment=` override is a direct
  Hydra override on the composed config (e.g. `misc.seed=42`,
  `regressor.alpha=0.5`, `dataset.processing.vi_fpca.n_components=6`,
  `misc.fold_env=DEH1.2020`).
- `+key=value` force-adds a key not declared in the schema (e.g.
  `+dataset.smoke_n=100` for a runtime-only sub-sample knob that
  lives outside `data/g2f.yaml`).

**`python -m baselines.fpca_predict`** — predict-mode sibling shim.
Currently raises `NotImplementedError` — see "Inference" above.

---

## Available Regressors

| Regressor | Key | Description |
|-----------|-----|-------------|
| OLS | `ols` | Ordinary least squares |
| GPR | `gpr` | Gaussian process regression; configurable kernel (`dot_product`, `rbf`) |
| BGLR | `bglr` | Multi-kernel Bayesian RKHS via Gibbs sampling (R subprocess; transductive — see architecture §"BGLR") |
| GBLUP | `gblup` | Multi-kernel REML/BLUP via `sommer` (R subprocess; transductive) — frequentist twin of `bglr` |

`gpr` estimates regularization strength from
data, making it the closest pure-Python equivalent to BGLR RKHS
models. `bglr` calls the R reference Gibbs sampler directly and
is the strongest match — at the cost of a Python↔R subprocess hop
and the transductive train+test-together fit shape.

`gblup` is the **frequentist twin** of `bglr`: the *same* multi-kernel
mixed model `y = 1·μ + Σ_k u_k + e` with `u_k ~ N(0, K_k·σ²_k)`, but the
variance components are estimated by **REML** (`sommer::mmer`) instead of
drawn by Gibbs, and predictions are the **BLUP/kriging** point estimate.
Conditional on the variance components, BLUP equals the Bayesian
posterior mean, so the two differ only in how the `σ²` are handled
(REML point estimate vs. full posterior averaging under
scaled-inverse-χ² priors). It reuses the identical kernel stack — the
env-GxE/weather-GxE `gblup` configs are byte-for-byte copies of the `bglr` ones with
only `regressor.name` swapped — so a comparison is a one-token change.
The env `BRR(Z_e)` term is folded into a linear kernel `K_env = Z_e Z_eᵀ`
(the RKHS form of a ridge on the env one-hot). Unlike `bglr`, `gblup`
needs no rank-1 patch: `V_tr` is SPD for any `σ²_e > 0`, so weather-GxE's rank-1
weather kernel is a non-event.

---

## Adding a New Regressor

1. Edit `baselines/regressors.py` and add a decorated factory function:
   ```python
   @register("svr")
   def _build_svr(cfg):
       from sklearn.svm import SVR
       return SVR(C=cfg.get("C", 1.0), kernel=cfg.get("kernel", "rbf"))
   ```

2. Pick a modality subset and create a model fragment, plus a
   matching 3-line experiment composer. Example for genomic-additive +
   SVR (using the R-mirror grammar with a new `svr` short code you'd
   also add to the regressor short-code registry above):

   ```yaml
   # conf/model/fpca/fpca_ga_svr_mse.yaml   (filename = facet-free peer mnemonic)
   # @package _global_
   model_name: fpca_ga.rank=20_svr_loss=mse

   dataset:
     processing:
       genomic:
         enabled: true
         csv_path: ./dataset-files/g2f/G2F_2020_2021_Genomic_Data.csv
         sources:
           - name: genomic_add
             type: additive_grm
             n_components: 20

   regressor:
     name: svr
     C: 1.0
     kernel: rbf
     loss_name: mse
     feature_keys: [genomic_add]

   misc:
     wandb_group: ${model_name}
     wandb_tags:
       - ${model_name}
       - "seed=${misc.seed}"
       - baseline
       - r-mirror
       - modality-ablation
   ```

   Follow the unified grammar above: the **filename** is a facet-free peer
   mnemonic (`fpca_ga_svr_mse`), while the **`model_name`** carries the facets
   (`fpca_ga.rank=20_svr_loss=mse`) and the per-encoder `.t=`/`.c=` tokens come
   from the groups. See any existing config under `conf/model/fpca/`.

3. Run:
   ```bash
   python -m baselines.fpca_train +experiment=fpca/fpca_ga_svr_mse
   ```

---

## Architecture

### Hybrid R/Python Pipeline

Training flow (the baseline is now a thin wrapper over the unified
feature-processing layer):

```
Python                                                    R (via subprocess)
  |                                                         |
  +- Build G2FDataset (FeatureProcessor.fit_transform)       |
  |    +- VIFPCAProcessor.fit_and_transform_all              |
  |    |    +- _extract_sparse_curves → tall CSV ----------->|
  |    |    |                                                +- fdapace::FPCA per VI (train only)
  |    |    |                                                +- project all splits via CE/BLUP
  |    |    |<---------------------------------------------  +- write FPC scores CSV + fpca_models.rds
  |    |    +- slice to n_components, attach as              |
  |    |       derived_features["vi_fpc_scores"] on every    |
  |    |       sample dict                                   |
  |    +- (optional) PhenomicFeatureBuilder / GenomicFeatureBuilder / ...
  +- baselines.fpca_train reads selected features from  |
  |    sample["derived_features"] and stacks them into X     |
  +- sklearn regressor fit → regression_model.joblib         |
  +- compute metrics → metrics.json                          |
```

Inference flow: `baselines.fpca_predict` is currently
unimplemented (see "Inference" above). For DL predict-mode evaluation
the path is `eval.py` → `setup._resolve_processing_cache_keys` →
`FeatureProcessor.load_and_transform(ds, cache_keys)` → each
processor's `load_fitted` → cached `fpca_models.rds` is reused by R
in `--mode predict` for VI FPCA.

### Data Flow Detail

VI FPC score extraction lives in
`utils/data/processing/vi_fpca.py::VIFPCAProcessor`. It wraps the
existing R subprocess call via `_extract_sparse_curves` (tall DataFrame
construction, content-addressed `sample_id` assignment) and
`_read_fpc_scores_all` (output CSV → `(N, 37*max_k)` matrix, sliced to
`n_components` in Python so `n_components` remains a post-load knob).
Results are cached to `{parent_of_data_dir}/.cache/vi_fpca/<key[:16]>/` with
both the FPC score `.npz` and `fpca_models.rds` — the latter lets
predict mode re-run R's `--mode predict` against the same fitted basis
without refitting. See
[`utils/data/processing/README.md`](../utils/data/processing/README.md).

```
G2FDataset (FeatureProcessor.fit_transform)
  → vi_fpca processor runs R once for all splits
  → every sample dict gets derived_features["vi_fpc_scores"] : Tensor(37*K)
  → baselines/fpca_core._collect_xy stacks selected feature keys
    (regressor.feature_keys, defaults to all derived features) into
    X : ndarray(N, sum_of_dims) and extracts y from yield_value
  → sklearn regressor.fit(X_train, y_train) → regression_model.joblib
  → _write_predictions → predictions.csv
  → compute metrics → metrics.json
```

### FPCA: What Is Learned and How Prediction Works

#### What FPCA estimates during training

For each of the 37 vegetation indices, `fdapace::FPCA()` estimates the following
from the sparse training curves:

1. **Mean function** `μ(t)` — the average VI curve shape across training samples,
   estimated via local polynomial smoothing with GCV-selected bandwidth. This is
   a continuous function, not tied to any particular sample's observation grid.

2. **Covariance surface** `G(s, t)` — the smoothed covariance between the VI
   values at any two timepoints `s` and `t`. Estimated from the raw cross-products
   of centered observations, then smoothed with a 2D kernel (bandwidth by GCV).

3. **Eigenfunctions** `φ_k(t)` for `k = 1, ..., K` — the principal modes of
   variation, obtained by eigendecomposition of `G(s, t)`. These are the
   functional analogs of PCA eigenvectors: `φ_1` captures the direction of
   greatest variation, `φ_2` the next orthogonal direction, and so on.

4. **Eigenvalues** `λ_k` — the variance explained by each eigenfunction.

5. **Noise variance** `σ²` — the estimated measurement error variance,
   separated from the signal covariance.

All of these are continuous functions or scalars — they are not tied to any
particular observation grid.

#### How FPC scores are computed for a new curve

Given a new sparse curve for one VI, observed at irregular timepoints
`{(t_1, y_1), ..., (t_J, y_J)}`, the goal is to estimate the K-dimensional
score vector `ξ = (ξ_1, ..., ξ_K)` that best represents this curve in the
learned eigenfunction basis.

The underlying generative model is:

```
Y(t_j) = μ(t_j) + Σ_k ξ_k · φ_k(t_j) + ε_j
```

where `ξ_k ~ N(0, λ_k)` are independent random scores and `ε_j ~ N(0, σ²)`
is measurement noise.

The scores are computed via the **Conditional Expectation (CE)** formula — the
Best Linear Unbiased Predictor (BLUP) under the Gaussian assumption:

```
ξ = Λ_K · Φ_K(t)ᵀ · Σ_Y(t)⁻¹ · (y − μ(t))
```

| Symbol        | Shape   | Description                                       |
|---------------|---------|---------------------------------------------------|
| `y`           | `J × 1` | Observed VI values at the sample's timepoints     |
| `μ(t)`        | `J × 1` | Mean function evaluated at the observed timepoints |
| `y − μ(t)`    | `J × 1` | Centered observations                             |
| `Φ_K(t)`      | `J × K` | Eigenfunctions evaluated at the observed timepoints|
| `Λ_K`         | `K × K` | `diag(λ_1, ..., λ_K)` — eigenvalue diagonal       |
| `Σ_Y(t)`      | `J × J` | Covariance of observed data (see below)            |
| `ξ`           | `K × 1` | Estimated FPC scores                               |

The covariance matrix of the observations is:

```
Σ_Y(t) = Φ_K(t) · Λ_K · Φ_K(t)ᵀ + σ² · I
```

This is the signal covariance (low-rank, from the eigenfunctions) plus the
measurement noise covariance (diagonal). Inverting `Σ_Y` optimally weights the
observations, down-weighting redundant or noisy timepoints.

**Step-by-step for a single sample and one VI:**

1. Evaluate the learned mean function at the sample's timepoints: `μ(t_1), ..., μ(t_J)`.
2. Center the observations: `y_j − μ(t_j)` for each `j`.
3. Evaluate each eigenfunction at the sample's timepoints: build the `J × K` matrix `Φ_K(t)`.
4. Construct `Σ_Y(t) = Φ_K · Λ_K · Φ_K^T + σ² I` (a `J × J` matrix).
5. Solve the linear system `Σ_Y(t) · z = (y − μ(t))` to get `z`.
6. Compute `ξ = Λ_K · Φ_K(t)^T · z` — the K-dimensional score vector.

**Key properties:**

- **Handles irregular grids natively**: the formula only evaluates `μ` and `φ_k`
  at whatever timepoints the sample has. No interpolation or grid alignment needed.
- **Handles variable sequence lengths**: a sample with 5 timepoints produces a
  `5×5` matrix `Σ_Y`; a sample with 20 timepoints produces `20×20`. Both yield
  the same K-dimensional score vector.
- **Borrows strength from the population**: the smooth eigenfunctions learned
  from all training data act as a prior — even very sparse curves get reasonable
  score estimates.
- **Noise-aware**: the `σ² I` term prevents overfitting to noisy observations.

#### Feature assembly

After computing K scores per VI for each sample, `_read_fpc_scores_all()` assembles
them into a flat feature vector:

```
x_i = [FPC1_vi0, ..., FPCK_vi0, FPC1_vi1, ..., FPCK_vi1, ..., FPC1_vi36, ..., FPCK_vi36]
```

With K components per VI and 37 VIs this gives a **37·K-dimensional**
feature vector per sample (K=5 → 185 in the shipped configs).
If FPCA estimates fewer than K eigenfunctions for some VI (low-rank data), the
remaining dimensions are zero-padded.

### Key Design Decisions

1. **Hybrid R/Python** — R's `fdapace` handles sparse FPCA (variable grids per sample) with smoothed covariance and GCV bandwidth. No Python equivalent.
2. **Per-VI FPCA** (37 separate) — K components per VI → 37·K total features (K=5 shipped).
3. **Subprocess + CSV** — simple, debuggable, no rpy2 dependency.
4. **Train-only FPCA basis** — FPCA fitted on train only; test projected via CE/BLUP. The inductive regressors consume train scores only; BGLR/GBLUP consume transductive kernels built from those scores (test yields masked).
5. **Config reuse** — Hydra composition (`conf/data/g2f.yaml` + per-model composer) guarantees identical dataset config as the DL pipeline, so train/test splits match fold-for-fold.
6. **Normalization off** — FPCA handles centering internally.
7. **Regressor registry** — flat factory pattern in `regressors.py` makes adding new methods trivial. The `loss_name` field in the regressor config dispatches to different training objectives (`mse` is currently the only supported loss; new losses are added to `_SUPPORTED_LOSSES` and handled in `build_regressor`).

### fdapace Settings

- `dataType = 'Sparse'`
- `methodMuCovEst = 'smooth'`
- `methodBwCov = 'GCV'`
- `methodBwMu = 'GCV'`

### Missing-cell policy (`vi_fpca.missing_values`)

The reader linearly interpolates NaN VI cells so the DL set encoder
can consume dense tensors. The VI FPCA processor then chooses what
to feed fdapace:

| Value | Behavior |
|-------|----------|
| `skip` *(default)* | Drop interpolated cells via the per-sample `vi_nan_mask` recorded by the reader. fdapace only sees originally-real observations — matches the R reference, which feeds tall-format observations with no NaN rows. |
| `interpolate` | Pass interpolated cells through to fdapace as if observed. Legacy scheme. |

The two modes produce different fitted bases and live under separate
content-addressed cache entries (`compute_vi_fpca_cache_key`
incorporates the mode), so switching does not corrupt prior caches.
Set via `dataset.processing.vi_fpca.missing_values` in any FPCA model
fragment, or override at the CLI:

```bash
python -m baselines.fpca_train +experiment=fpca/<slug> \
    dataset.processing.vi_fpca.missing_values=interpolate
```

### Weather FPCA parity knobs (`weather_fpca.missing_values`, in-sample scores)

The weather FPCA processor mirrors the R reference in three places that
matter for the PTR FPC1 scores:

| Knob | Default | Behavior |
|------|---------|----------|
| `missing_values` | `drop` | NaN weather cells stay NaN and fdapace removes them per curve, so a missing day is a gap (the R reference). `ffill` forward/back-fills within each env — a gap-free grid for consumers that need one (the DL raw-weather encoder pins it explicitly). Only PTR and PAR_TEMP carry NaNs in the shipped CSV. |
| in-sample scores | `xiEst` | Training-set scores are read straight from the fitted object (`fpca_obj$xiEst`), as the R reference does; `predict()` is used only for held-out curves. Re-projecting the fitting set gives nearly, but not exactly, the same numbers. |
| `nRegGrid` | fdapace default | Set to 100 only on the full 19-environment weather fit, matching the R reference; every other fit uses fdapace's default (51). |

Non-finite values are clipped before fdapace sees them, and inputs are passed
as float64. With these defaults the LOEO weather FPC scores reproduce the R
reference to machine precision; the VI FPCA path applies the same `xiEst`
rule in train mode.

### BGLR (multi-kernel Bayesian RKHS via R subprocess)

The `bglr` regressor uses the RKHS(K) parameterization. Each
kernel block produces a separate ETA term, so per-kernel variance
components σ²_k are sampled jointly via Gibbs — producing
per-RKHS-term variances that no single-prior sklearn regressor
can match.

> ⚠️ **Requires the rank-1-patched BGLR.** Stock CRAN BGLR 1.1.4 aborts
> with `invalid 'times' argument` when any kernel block has a single
> super-`tolD` eigenvalue (a rank-1 kernel) — e.g. weather-GxE's weather kernel
> `K_W` (`weather_fpca.n_components=1`). R drops the
> 1-column eigenvector matrix to a vector and the outer product fails.
> Install the `drop = FALSE` fix with `bash scripts/patch_bglr.sh`; see
> [`R_HPC_SETUP.md`](R_HPC_SETUP.md) ("Patch BGLR for rank-1 RKHS terms")
> and [`bglr_rank1.patch`](bglr_rank1.patch).

**Transductive interface.** BGLR has no inductive predict step.
Train + test rows are passed to BGLR together, test `y` is masked
to NA, and `fit$yHat` is read back for the full vector. The Python
side handles this via `BGLRRegressor.fit_predict_blocks` and a
dedicated branch in `fpca_core.train` that pops the env block out
of `feature_keys` and routes it to its own ETA term.

**ETA layout.** One RKHS term per `feature_keys` kernel entry, in the
order they appear. The Python wrapper passes each raw obs-level N×N
kernel `K_k` to `bglr_compute.R`, which eigen-decomposes it and builds
`list(V=vectors, d=values, model='RKHS')` (no reconstruction from V·√d in
Python). The env block
(`env_brr_key: env_onehot`) is appended last as
`list(X=Z_e, model='BRR')` to match the R reference's `Z_e` ETA-last
convention. The R reference's same-env (env-GxE/weather-GxE) fragments set
`env_brr_key: env_onehot` and list `env_onehot` in
`regressor.feature_keys`; this env main effect renders as the
leading **`eid`** slug peer (see the peer registry). The
**env-free** fragments — main-effects (`fpca_ga_gd_vi_bglr_mse`) and the
env-free weather-interaction weather-GxE (`fpca_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_bglr_mse`,
the env-bearing twin's bare-stem counterpart) — deliberately set
`env_brr_key: null`, omit `env_onehot`, and so carry **no**
`eid` peer. (NB: although the ETA is appended *last*, the slug
sorts peers alphabetically, so `eid` appears *first*.)

**Full-rank kernel inputs.** BGLR fragments leave both
`n_components` and `n_components_max` unset on every `*_relmat`
source (genomic, phenomic, enviromic, six Hadamard interactions),
so each kernel block consumes the full filtered eigen-subspace.
`weather_fpca` is the lone exception (`n_components: 1` in the shipped
weather-GxE recipe — the pre-BGLR FPCA reduction to a rank-1 K_W).

**`fit_scope=all` invariant.** Every kernel processor must run
with `fit_scope=all` so the eigen-projected V·√d covers the rows
BGLR sees (train + test). `genomic`, `enviromic`, and the
`weather_kernel` layer (`weather_relmat`) run at `fit_scope=all`;
`weather_fpca` itself stays `fit_scope=train` (the FPCA basis is fit on
train envs only and held-out curves are CE/BLUP-projected). `phenomic` defaults to
train-only and the BGLR/GBLUP variants flip it to `all` explicitly;
the inductive `ols` variant keeps `fit_scope=train`.

**Kernel-rank semantics.** The transductive wrapper passes each raw
obs-level N×N kernel block to `bglr_compute.R`, which eigen-decomposes it
(`eigen(K, symmetric=TRUE)`) and hands BGLR `list(V=vectors, d=values,
model='RKHS')`. BGLR's RKHS handler then keeps only the
eigenvalue columns above its own tolerance (`tolD=1e-10`) — the
`V[, tmp, drop=FALSE]` site the rank-1 patch fixes — so any rank reported
by BGLR (`length(fit$ETA[[k]]$d)`) is `≤` the number of positive
eigenvalues of `K`. For PSD kernels (genomic /
phenomic / enviromic / Hadamard interactions of PSD factors) the
upstream filter drops only numerical noise — the relative
variance lost is typically <1e-12, well below BGLR's own
internal threshold.

**Reproducibility.** Canonical CRAN BGLR has no `seed=` argument
(would error with `unused argument`); reproducibility relies
entirely on `set.seed()` driving the base R RNG that BGLR samples
from. `misc.seed` flows: `misc.seed` → `regressor.seed` via
`setdefault` in `fpca_core` → `manifest.json["seed"]` →
`set.seed(manifest$seed)` in `bglr_compute.R` BEFORE ETA
construction.

**Threading.** Thread caps are pinned by the Python parent
(`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` all set to `"1"` via
`subprocess.run(env=...)`), not by `Sys.setenv()` inside R.
OpenBLAS / MKL / Accelerate size their thread pools at first
BLAS call and may have already done so by the time R runs — the
parent-env approach is uniformly correct across all four BLAS
backends.

**Hyperparameters.** `n_iter=10000`, `burn_in=1000` defaults.
Override via `regressor.n_iter=N regressor.burn_in=B` for fast smokes.

The env-GxE/weather-GxE BGLR configs follow a component-uniformity
invariant: every interaction component is an eigen-projected relmat
(V·√d) or the env one-hot.
