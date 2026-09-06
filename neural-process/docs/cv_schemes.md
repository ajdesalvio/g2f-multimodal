# CV schemes (CV0/CV00/CV1/CV2) and the FIT / OBSERVE / PREDICT roles

A plain-language reference for the cross-validation design used by both the
FPCA/BGLR baselines and the DL models. Implemented in `cv/schemes/*.py`;
a faithful port of the R reference pipeline (`DAP_CV_Prediction_*`).

## The setup: a variety × environment grid

Every observation is one **variety** (a corn line, defined by its DNA;
a.k.a. genotype / "female") grown in one **environment** (a location×year,
e.g. `DEH1.2020`). The target is **yield**. Think of a table:

```
                Env A   Env B   Env C   Env D
   Variety 1     12      14       9      ?
   Variety 2     11      ?       10      8
   Variety 3      ?      13      11      9
   Variety 4(new) ?       ?       ?      ?
```

We measured some cells and predict the empty ones. Side information makes the
"new" cases tractable: a new variety carries **DNA** (similarity to tested
varieties); a new environment carries **weather** and in-season **vegetation
index (VI)** measurements.

## The 2×2: four kinds of held-out cell

Two yes/no questions define which prediction problem a held-out cell poses:

1. Has this **variety** (row) been tested anywhere else?
2. Has anything been grown in this **environment** (column) before?

|                              | Env tested (known column) | Env brand-new (unknown column) |
| ---------------------------- | ------------------------- | ------------------------------ |
| **Variety tested** (row)     | **CV2**                   | **CV0**                        |
| **Variety brand-new** (row)  | **CV1**                   | **CV00**                       |

- **CV2 — "fill the gaps."** Both variety and environment are familiar; only
  this combination is missing. Patches holes in sparse trials. *Easiest.*
- **CV1 — "is this new variety any good?"** New variety, known environments.
  Predict promising lines from DNA before field-testing them widely (the
  core of genomic selection).
- **CV0 — "known varieties in a new place."** Familiar varieties, new
  location/year. Deploying to a new region or forecasting next season.
- **CV00 — "new variety AND new place."** Both unknown; no direct anchor on
  either axis. The true deployment frontier. *Hardest.*

`env_year_loo` (leave-one-environment-out) is the degenerate case: a whole
environment column is held out with no variety-fold structure — it blends
CV0/CV00 into a single `test` label.

## FIT / OBSERVE / PREDICT — the three roles

Key idea: **we hide *yields* (the answer), never *plants*.** Every row's
features (DNA, weather, VI) are always visible; only the yield is masked.
A model fit has two steps, and a row can be in one, both, or neither:

1. **Unsupervised feature calibration** — fitting the VI FPCA basis, kernel
   eigendecompositions, and normalization stats. Uses *features only*.
2. **Supervised fit** — learning features → yield. Uses *features + yield*.

| Role        | In yield (supervised) fit? | Calibrates features? | Scored? |
| ----------- | -------------------------- | -------------------- | ------- |
| **FIT**     | yes                        | **yes**              | in-sample only (CV2) |
| **OBSERVE** | yes                        | no (projected)       | no      |
| **PREDICT** | **no — yield hidden**      | no (projected)       | **yes** (CV0/00/1/2) |

- **FIT** rows train the model *and* define the feature basis. They are the
  designed, foldable reference panel ("common" varieties, with a fold number).
- **OBSERVE** rows are extra varieties that have yield data but are *not* part
  of the fold design ("non-common", no fold number). Their yields are used as
  training signal, but they do not define the basis — they are projected onto
  the basis the FIT rows built, and are never scored.
- **PREDICT** rows have their yield masked to NA; the model predicts them and
  they are graded.

Why split FIT vs OBSERVE? Only step 1 honors it: the basis is fit on a clean,
controlled reference panel and *applied by projection* to everyone else, so
OBSERVE and PREDICT rows receive identical representation treatment. It is a
deliberate convention (and exact R parity), not a leakage necessity — for
OBSERVE rows there is no leakage either way; the trade-off is a more
controlled basis vs. a slightly data-richer one.

`env_year_loo` has **no OBSERVE rows** (`fit_mask is None`): every non-held-out
row is FIT. OBSERVE only appears in the fold-structured schemes
(CV1/CV2/CV0/CV00).

## Worked example

4 common varieties (folds 1/2), 2 non-common extras (X1/X2), 3 environments.

**CV2/CV1 — hold out Fold 1, keep all environments:**

```
          EnvA  EnvB  EnvC   role          why
 V1(F1)   TEST  TEST  TEST   PREDICT→CV1   new variety, known env
 V2(F1)   TEST  TEST  TEST   PREDICT→CV1
 V3(F2)   FIT   FIT   FIT    FIT→CV2        trains + calibrates (in-sample score)
 V4(F2)   FIT   FIT   FIT    FIT→CV2
 X1(—)    OBS   OBS   OBS    OBSERVE        trains only, never scored
 X2(—)    OBS   OBS   OBS    OBSERVE
```

**CV0/CV00 — hold out Fold 1 AND remove EnvC:**

```
          EnvA  EnvB  EnvC   role
 V1(F1)   hide  hide  TEST   EnvC → CV00 (new variety, new env); A/B hidden to keep V1 "untested"
 V2(F1)   hide  hide  TEST   EnvC → CV00
 V3(F2)   FIT   FIT   TEST   EnvC → CV0  (known variety, new env)
 V4(F2)   FIT   FIT   TEST   EnvC → CV0
 X1(—)    OBS   OBS   hide   OBSERVE in A/B; EnvC hidden but not scored
 X2(—)    OBS   OBS   hide   OBSERVE in A/B; EnvC hidden but not scored
```

Note the masking is broader than the scored set: V1/V2 are hidden in A/B so
they look genuinely untested everywhere (otherwise they wouldn't be "new"),
but only the held-out-env cells are graded.

## How the folds are constructed (with concrete numbers)

The worked example used "Fold 1 / Fold 2" abstractly. Here is how those fold
numbers are actually assigned, and what the counts are on the real G2F data.

### The dataset, in numbers

| Quantity | Count | Notes |
| -------- | ----- | ----- |
| Environments (`Env.Year`) | **19** | 12 locations × 2 years (2020–2021); some locations are single-year |
| Distinct varieties (maternal line) | **407** | the `Pedigree` substring before the first `/`, lowercased (`female_of`) |
| **Common** varieties (in **all 19** envs) | **223** | the foldable reference panel → become FIT/PREDICT |
| Non-common varieties | **184** | appear in a subset of envs → always OBSERVE (no fold number) |

The split is sharply bimodal: 223 varieties appear in every environment,
while a large block (~154) appears in only ~12. Only the 223 common ones get
fold numbers; everyone else is OBSERVE background by construction.

### Step 1 — build the fold map (shared by `cv_2_1` and `cv_0_00`)

The fold map assigns each of the **223 common varieties** a fold number
**1–5** (`NativeFoldSource.build`, `cv/schemes/fold_source.py`):

1. **Sort** the 223 common varieties into a fixed order.
2. **Lay down balanced labels** `[1,1,…,2,2,…,5,5,…]` with ~`223/5 ≈ 45`
   of each (`np.repeat`).
3. **Shuffle** with a seeded permutation (`cv_seed` controls the shuffle),
   then assign variety *i* → label *i*.

Two properties matter:

- **Folds group *varieties*, not rows.** A variety in fold 3 moves with *all*
  its samples across *every* environment at once — a **grouped** k-fold, so the
  same genotype never sits in train and test simultaneously. That is what makes
  "unseen variety" a clean test.
- **Balanced, k = 5.** Per seed the five folds hold ≈ 45 / 45 / 45 / 45 / 43
  varieties (223 total).

The assignment is **frozen to a file** for a fair, paired DL-vs-baseline
comparison: `cv/data/female_folds.csv` (columns `Seed_Num, Female, Fold`)
stores it for **10 seeds** → 10 × 223 = **2230 rows** (per-fold totals across
all seeds: 450 / 450 / 450 / 450 / 430). Both pipelines read this same file
via `cv_spec.fold_source=r_csv`, so they hold out identical varieties.
`cv_seed` selects which seed's partition to use; extra seeds give independent
re-partitions for stability. (`RCsvFoldSource` re-validates the CSV against
the data's common set at load.)

**The committed table is the R reference's.** `cv/data/female_folds.csv` is the
exact fold table produced by the R reference pipeline (`set.seed(seed_num)` +
`sample` over the sorted 223 common females; identifiers upper-case, matched
case-insensitively) and is the launcher default (`submit_cv.sh`), so the TNP
splits are identical to the kernel-model DAP/AGDD runs. `make_female_folds.py`
can generate an alternative (native-RNG) realization — same 223 females,
different partition. Override with `G2F_FEMALE_FOLDS_CSV=<path>`.

### Step 2 — how each scheme *uses* the map

The fold map is built once; what differs is what each scheme rotates over.

| Scheme | Uses fold map? | "Fold" loop variable | Runs (per seed) |
| ------ | -------------- | -------------------- | --------------- |
| `env_year_loo` | **No** (`requires_fold = False`) | environment | **19** |
| `cv_2_1` | Yes (5 variety-groups) | fold `k` ∈ 1…5 | **5** |
| `cv_0_00` | Yes (5 variety-groups) | (fold `k`, env `e`) | up to 5 × 19 = **95** |

- **`env_year_loo`** ignores the fold map entirely. A "fold" is one
  environment: hide that whole column → `test`, everything else → FIT.
  19 environments → **19 runs**, no variety-folding.
- **`cv_2_1`** loops fold `k` ∈ 1…5. Group-`k` varieties are hidden in
  *every* environment (PREDICT → CV1); the other common varieties are FIT
  (in-sample CV2); non-common are OBSERVE. **5 runs/seed**; environment is
  never looped (every env keeps training data).
- **`cv_0_00`** loops *(fold `k`, held-out env `e`)*. It hides the entire `e`
  column **plus** group-`k` varieties everywhere; training never sees env `e`.
  Only column `e` is scored, split by fold: varieties not in `k` → CV0
  (seen variety, new env), varieties in `k` → CV00 (new variety, new env).
  Up to **95 runs/seed** — by far the heaviest scheme.

## Per-modality fit scope (what "calibration" actually sees)

- **VI** (`vi_fpca`): basis fit on **FIT rows only**, all else projected. The
  only processor that consults the row-level `fit_mask` (it has no `fit_scope`
  key — the FIT rows are selected by `fit_mask`, not a `fit_scope: train`
  setting). **Caveat — OOD validation carves leak through this basis:** a DL
  eval-streams `carve` stream pulls val rows that are still inside the FIT
  `fit_mask`, so the basis is partly fitted on the held-out val rows. This is
  guarded (a `carve` + `vi_fpca` config is a hard error at the VALIDATE gate)
  and deferred; see
  [`utils/data/processing/README.md`](../utils/data/processing/README.md)
  §"`fit_mask` and the OOD-validation caveat" for the full treatment.
- **Weather**: environment-level. The `weather_fpca` score basis and
  z-stats are fit on train envs (`fit_scope: train`), held-out env
  projected; the env-level `weather_kernel` (K_W) is `fit_scope: all` in
  the transductive BGLR/GBLUP composers. The held-out env's weather is still
  used as input.
- **Genomic** (GRM / dosage): `fit_scope: all` — fit transductively over
  **all** pedigrees (genotype is observed pre-planting, deemed leakage-safe).

## FPCA/BGLR vs DL

The FIT/OBSERVE distinction is meaningful only because of the separate
unsupervised calibration step (step 1).

- **FPCA/BGLR**: the VI FPCA basis is fit on FIT rows, then frozen and
  projected onto OBSERVE/PREDICT; the downstream BGLR/GBLUP kernels are
  transductive (`fit_scope: all` — they span all rows, with held-out
  yields masked in the fit; the inductive `ols` variant keeps its kernels
  train-scoped). OBSERVE rows contribute yields to BGLR but not to the
  basis.
- **DL (end-to-end)**: the standard DL models use a **raw VI set encoder**
  learned end-to-end (no frozen basis), genomic is transductive, and weather
  is env-level — **none consult `fit_mask`**. So FIT and OBSERVE become
  operationally identical: both are ordinary training rows
  (`train = FIT ∪ OBSERVE`), both contribute gradients, and `val` (DL early
  stopping) is carved from their union. OBSERVE has no special effect unless a
  DL config explicitly enables the `vi_fpca` processor.

## Naming in the R reference scripts

This codebase's role vocabulary (`FIT` / `OBSERVE` / `PREDICT`, "common /
non-common") is **our own**; the R reference pipeline (`DAP_CV_Prediction_*`)
never uses those words. The concepts line up as follows:

| This codebase        | R reference scripts                              | Definition |
| -------------------- | ------------------------------------------ | ---------- |
| **FIT**              | `train_ids` (`DAP_CV_Prediction_Setup_V2.R`) | common females not in the held-out fold (and, for CV_0_00, not in the held-out env); fits the VI FPCA basis **and** stays in the BGLR fit with observed yield |
| **OBSERVE**          | *(no name — implicit)*                      | rows with `Fold = NA`: a Female that is **not** a common female. Falls out of `train_ids` automatically and is never masked, so it sits in the BGLR fit but not the FPCA basis |
| **PREDICT** / test   | masked rows via `mask_yields` (`DAP_CV_Prediction_V1.R`) | held-out fold and/or held-out env; `y <- NA`; scored |
| **common** variety   | `common_females` (`DAP_CV_Metadata_V1.R`)  | females present in **all** environments (`n_envs == total_envs`); only these are assigned fold numbers (`make_fold_map`) |
| CV0/00/1/2 labels    | same (`evaluate_metrics`)                   | unchanged |

Key point: **"OBSERVE" is our name for a group R leaves unnamed.** Because
`make_fold_map` assigns folds only to common females, every non-common-female
row gets `Fold = NA`; the R Setup script's `train_ids = is_common & Fold != k`
silently drops them from the FPCA basis, while the Prediction script's
`mask_yields` (which only masks `Fold == k` or the held-out env) leaves their
yields in the BGLR fit. So the R reference *does* use these rows — there is just no term
for them. When comparing against the R scripts, refer to them as "the
non-common-female rows" rather than "OBSERVE."

## Pointers

- Role assignment: `cv/schemes/cv_2_1.py`, `cv/schemes/cv_0_00.py`,
  `cv/schemes/env_year_loo.py`; masks in `utils/data/splitter.py`
  (`assign_roles`).
- Fold construction: common-set in `cv/schemes/common_females.py`; fold-map
  backends in `cv/schemes/fold_source.py` (`NativeFoldSource` / `RCsvFoldSource`);
  pinned assignment in `cv/data/female_folds.csv` (the R reference's table;
  `scripts/cv/make_female_folds.py` generates a native alternative); wired
  together in `cv/schemes/resolve.py`.
- Scheme selection: `conf/cv_spec/{cv_2_1,cv_0_00,env_year_loo}.yaml`.
- Scoring labels (CV0/00/1/2): `cv/scoring.py`; `cv/README.md`.
- R reference (parity target): `DAP_CV_Prediction_Setup_V2.R` (`train_ids`),
  `DAP_CV_Prediction_V1.R` (`mask_yields`, `evaluate_metrics`),
  `DAP_CV_Weather_Setup_V3.R` (ALL vs LOEO weather).
