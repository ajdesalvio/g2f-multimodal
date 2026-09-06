#!/bin/bash
# =============================================================================
# CV_0_00_2_1 — UNIFIED train/test submission for ALL model classes
# =============================================================================
# ONE launcher fans out every model class — neural_process (induced-set TNP)
# and fpca (BGLR / GBLUP / sklearn baselines) — over BOTH cv schemes
# (cv_0_00 + cv_2_1), one SLURM job per split
# (class × scheme × cv_seed × fold × [heldout_env] × seed).
#
# UNIFIED ARTIFACT PATH — a single root, diverging first on model class:
#   artifacts/cv/cv_0_00_2_1/<MODEL_CLASS>/<scheme>/<token>/<slug>/seed=<n>
#                            └ neural_process | fpca
#
# HOW TO EDIT — everything you tune is in the "MODEL SPECS" section right below.
# Each class is ONE block: its knobs (engine / gres / time / epochs / patience /
# project) apply to the whole family, then its CONFIGS list. Add a model = add a
# line to that block's CONFIGS. The spec is shared across every config in the
# block; if a specific model needs different resources, bump the block's knobs
# (or split it into a second block) — that's on you. Everything below the specs
# is machinery you shouldn't need to touch.
#
# Two engines, chosen by a block's ENGINE:
#   * dl   -> dl_job.job   (GPU; const LR, no early-stop, best+frozen+last,
#                           curves; curve.done marker)
#   * fpca -> fpca_job.job (CPU + R; endpoint BGLR/GBLUP; metrics.json marker)
#
# PARITY: every class shares cv_seed + fold + FOLD_CSV (r_csv) + val_seed=0, so
# the held-out test sets are IDENTICAL and metrics are directly comparable.
#
# *** SUBMITS jobs. Preview first:  DRY_RUN=1 bash <this>                        ***
# Run (preview):        DRY_RUN=1 bash scripts/jobs/cv/cv_0_00_2_1/submit_cv.sh
# Run (smoke 1 split):  ONLY_CLASS=neural_process ONLY_SCHEME=cv_0_00 ONLY_FOLD=1 \
#                         ONLY_ENV=DEH1.2020 EPOCHS=3 SMOKE_N=512 \
#                         ARTIFACT_BASE=artifacts/cv/cv_0_00_2_1-smoke \
#                         bash scripts/jobs/cv/cv_0_00_2_1/submit_cv.sh
# Run (one class):      ONLY_CLASS=fpca bash scripts/jobs/cv/cv_0_00_2_1/submit_cv.sh
# Run (ONE seed pair):  ONLY_PAIR=1:1 bash scripts/jobs/cv/cv_0_00_2_1/submit_cv.sh
# Run (full submit):    bash scripts/jobs/cv/cv_0_00_2_1/submit_cv.sh
#
# *** STAGE REPLICATES ONE PAIR AT A TIME. *** A bare submit fans out every entry
# in SEED_PAIRS (5 pairs = 1500 jobs), which exceeds MaxSubmitJobs. The intended
# loop is: ONLY_PAIR=<pair> submit -> wait for the queue to drain and all splits
# to write curve.done -> ONLY_PAIR=<next pair> submit. Re-running the SAME pair is
# idempotent (splits with curve.done are skipped), so it doubles as gap backfill.
# =============================================================================
# shellcheck disable=SC2034  # NP_*/FPCA_* spec vars are read indirectly (run_class: ${!v})
# shellcheck disable=SC2206  # ONLY_* filters intentionally word-split into arrays
set -euo pipefail

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MODEL SPECS — the only section you edit.                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ── Shared split grid (the parity invariants — SAME for every class) ──────────
SCHEMES=(cv_0_00 cv_2_1)
FOLDS=(1 2 3 4 5)
HELDOUT_ENVS=(                                  # the 19 G2F env-years (cv_0_00 only)
  DEH1.2020 IAH4.2021 MIH1.2020 MNH1.2020 MNH1.2021 MOH1.2020 NEH1.2021
  TXH1.2020 TXH1.2021 TXH2.2020 TXH2.2021 TXH3.2020 TXH3.2021
  WIH1.2020 WIH1.2021 WIH2.2020 WIH2.2021 WIH3.2020 WIH3.2021
)
# Paired replicates for robustness: each entry is cv_seed:model_seed, iterated
# as PAIRS (not a cross-product) -> N replicates, not NxM. cv_seed sets the fold
# partition (R Seed_Num; SHARED across classes so the paired DL-vs-FPCA
# comparison holds out the same genotypes); model_seed sets misc.seed (DL
# weight-init / BGLR set.seed).
SEED_PAIRS=(
  "1:1"
  "2:2"
  "3:3"
  "4:4"
  "5:5"
)

# ── neural_process (induced-set TNP) ─────────────────────────────── A100, TNP ─
NP_ENGINE=dl
NP_GRES=gpu:a100:1
NP_TIME=12:00:00   # full-scale campaign: mean 3.3 h, max 7.5 h per split
NP_EPOCHS=250
NP_PATIENCE=25
NP_PROJECT=G2F-NP
NP_CONFIGS=(
  "istnp_vi_wthr_ga_gd__grm-rows__vienc-tfm__wthrenc-tfm__aug-vi-wthr__nll"   # VI + weather + genomic
  "istnp_vi_ga_gd__grm-rows__vienc-tfm__aug-vi__nll"                          # weather-free (VI + genomic)
  "istnp_vi__vienc-tfm__aug-vi__nll"                                          # VI-only (no weather, no genomic)
)

# ── fpca (BGLR / GBLUP / sklearn baselines) ─────────────────── CPU + R, Grace ─
FPCA_ENGINE=fpca
FPCA_TIME=24:00:00  # FPC scores are recomputed from scratch per split
FPCA_CONFIGS=(
  "fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse"                    # env-GxE  (DAP)
  "fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse__t-agdd"           # env-GxE  (AGDD)
  "fpca_ga_gd_vi_bglr_mse"                                            # G+P main-effects  (DAP)
  "fpca_ga_gd_vi_bglr_mse__t-agdd"                                    # G+P main-effects  (AGDD)
  "fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_bglr_mse"           # weather-GxE (DAP)
  "fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_bglr_mse__t-agdd"   # weather-GxE (AGDD)
  "fpca_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_bglr_mse"               # weather-GxE (DAP)  env-free — the reported weather model
  "fpca_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_bglr_mse__t-agdd"       # weather-GxE (AGDD) env-free — the reported weather model
  "fpca_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse"                       # env-GxE  (DAP)  env-free — the reported weather-free model
  "fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_gblup_mse"                  # env-GxE  (DAP) GBLUP
  "fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_gblup_mse__t-agdd"          # env-GxE  (AGDD) GBLUP
  "fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_gblup_mse"          # weather-GxE (DAP) GBLUP
  "fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_gblup_mse__t-agdd"  # weather-GxE (AGDD) GBLUP
  "fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_ols_mse"            # weather-GxE (DAP) OLS
)

# Classes to run, in order (prefix:class_name). Comment a line to skip a class.
CLASS_ORDER=(
  "NP:neural_process"
  # "FPCA:fpca"     # disabled — TNP only for now
)

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MACHINERY — you shouldn't need to edit below here.                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Run infra (env-overridable; not per-model).
ARTIFACT_BASE="${ARTIFACT_BASE:-artifacts/cv/cv_0_00_2_1}"   # unified root
# Fold table for r_csv parity ("" = native). The committed default is the R
# reference pipeline's own set.seed+sample table, so TNP splits are IDENTICAL
# (same 223 common females, same fold per Seed_Num) to the kernel-model
# DAP/AGDD runs. See docs/cv_schemes.md.
FOLD_CSV="${G2F_FEMALE_FOLDS_CSV:-cv/data/female_folds.csv}"
CLUSTER="${CLUSTER:-GRACE}"; SLURM_MAIL_USER="your-email@example.com"
# Default = GRACE allocation; override via env for another cluster, e.g.
# ACCOUNT=YOUR_FASTER_ACCOUNT (FASTER). NB: fpca needs R -> Grace only.
ACCOUNT="${ACCOUNT:-YOUR_GRACE_ACCOUNT}"

# Optional filters (smoke / targeted backfill).
[[ -n "${ONLY_SCHEME:-}" ]] && SCHEMES=(${ONLY_SCHEME})
[[ -n "${ONLY_FOLD:-}"   ]] && FOLDS=(${ONLY_FOLD})
[[ -n "${ONLY_ENV:-}"    ]] && HELDOUT_ENVS=(${ONLY_ENV})
# ONLY_PAIR stages ONE replicate at a time (e.g. ONLY_PAIR=1:1). Submitting all
# of SEED_PAIRS at once is 5x the grid and exceeds MaxSubmitJobs, so pairs are
# run in sequence: submit a pair, wait for it to finish, then submit the next.
[[ -n "${ONLY_PAIR:-}"   ]] && SEED_PAIRS=(${ONLY_PAIR})
ONLY_CLASS="${ONLY_CLASS:-}"       # neural_process | fpca
ONLY_CONFIG="${ONLY_CONFIG:-}"     # exact config name (targeted backfill)
EPOCHS_OVERRIDE="${EPOCHS:-}"      # override every dl block's epochs (smoke)
SMOKE_N="${SMOKE_N:-}"
DRY_RUN="${DRY_RUN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DL_JOB="${SCRIPT_DIR}/dl_job.job"
FPCA_JOB="${SCRIPT_DIR}/fpca_job.job"
LOG_DIR="${SCRIPT_DIR}/logs"

# ── Preflight ─────────────────────────────────────────────────────────────────
[[ -z "${ACCOUNT}" ]] && { echo "Error: no SLURM account set."; exit 1; }
[[ -f "${DL_JOB}" ]]   || { echo "Error: engine not found: ${DL_JOB}"; exit 1; }
[[ -f "${FPCA_JOB}" ]] || { echo "Error: engine not found: ${FPCA_JOB}"; exit 1; }

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

# Cluster env — loaded whenever present (Grace login node), no-op elsewhere (a
# Mac dry-run falls back to system python). Loaded in DRY_RUN too: the plan
# resolves slugs/tokens via python, so previewing on Grace needs it on PATH.
if ! command -v module &>/dev/null && [[ -f /etc/profile.d/hprc_profile.sh ]]; then
  set +eu; source /etc/profile.d/hprc_profile.sh; set -eu
fi
if command -v module &>/dev/null; then module purge; module load GCCcore/13.2.0 Python/3.11.5; fi
ENV_PATH="/scratch/user/$USER/neural-processes/env/bin/activate"
[[ -f "${ENV_PATH}" ]] && source "${ENV_PATH}"

if [[ -n "${FOLD_CSV}" && ! -f "${FOLD_CSV}" ]]; then
  echo "Error: FOLD_CSV='${FOLD_CSV}' not found."; exit 1
fi
[[ -z "${FOLD_CSV}" ]] && echo "*** WARNING: native folds break cross-class pairing ***"

# ── Slug resolution ───────────────────────────────────────────────────────────
# DL: compose with the SAME identity-bearing overrides the engine applies
# (sched_name=const + the scheme's eval_streams) so OUT_DIR slug == runtime
# meta.model_slug (incl. the _sched=const + _val=gv30 tags). FPCA: plain slug.
resolve_dl_slug() {   # $1=config $2=scheme
  python - "$1" "$2" <<'PY'
import sys
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import conf  # noqa: F401  -- register resolvers + schemas
exp, es = sys.argv[1], sys.argv[2]
conf_dir = str((Path.cwd() / "conf").resolve())
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
    cfg = compose(config_name="config",
                  overrides=[f"+experiment=dl/{exp}", "sched_name=const", f"+eval_streams={es}"])
slug = OmegaConf.to_container(OmegaConf.create({"v": cfg.misc.model_slug}), resolve=True)["v"]
assert "${" not in slug, f"unresolved interpolation: {slug!r}"
print(slug)
PY
}
resolve_fpca_slug() {  # $1=config
  python -c "from cv.slug_util import resolve_slug; print(resolve_slug('fpca/$1'))"
}

# Per-config extra Hydra overrides for FPCA (threaded to fpca_job.job as
# EXTRA_OVERRIDES). main-effects pins vi_fpca.fit_vars=null so it reuses the full-37
# VI-FPCA cache (byte-identical key) instead of an R re-fit.
fpca_extra_overrides() {  # $1=config -> echoes overrides or ""
  case "$1" in
    fpca_ga_gd_vi_bglr_mse|fpca_ga_gd_vi_bglr_mse__t-agdd)
      echo "dataset.processing.vi_fpca.fit_vars=null" ;;
    *) echo "" ;;
  esac
}

submit_with_retry() {
  local rc
  for attempt in 1 2 3; do
    set +e; sbatch "$@"; rc=$?; set -e
    [[ $rc -eq 0 ]] && return 0
    echo "  sbatch failed (rc=$rc); retry ${attempt}/3 in $((attempt*5))s"
    sleep $((attempt * 5))
  done
  echo "  Giving up after 3 retries."; return 1
}

fold_token() {  # $1=cv_seed $2=fold $3=scheme $4=env(or __none__)
  if [[ "$3" == "cv_0_00" ]]; then
    python -c "from cv.slug_util import fold_dir_token; print(fold_dir_token(cv_seed=$1, fold=$2, heldout_env='$4'))"
  else
    python -c "from cv.slug_util import fold_dir_token; print(fold_dir_token(cv_seed=$1, fold=$2))"
  fi
}

# ── submit_class: fan the loaded class (CLASS/ENGINE/TIME/CONFIGS[+GRES/EPOCHS/
# PATIENCE/PROJECT]) out over the shared grid. Engine infra (cpus/mem/partition/
# wandb) is uniform and lives here, NOT in the per-class specs. ────────────────
submit_class() {
  [[ -n "${ONLY_CLASS}" && "${CLASS}" != "${ONLY_CLASS}" ]] && return 0
  [[ ${#CONFIGS[@]} -eq 0 || -z "${CONFIGS[0]:-}" ]] && return 0
  echo "════════════ ${CLASS}  (engine=${ENGINE}, ${#CONFIGS[@]} config[s]) ════════════"

  # Uniform engine infra (rare to change).
  local CPUS=2 MEM="50G" PARTITION="gpu"
  local WANDB; [[ "${ENGINE}" == "dl" ]] && WANDB=true || WANDB=false

  local config scheme slug pair cv_seed model_seed fold env token out marker job jname epochs extra EXPORTS
  local -a envs res
  for config in "${CONFIGS[@]}"; do
    [[ -n "${ONLY_CONFIG}" && "${config}" != "${ONLY_CONFIG}" ]] && continue
    for scheme in "${SCHEMES[@]}"; do
      if [[ "${ENGINE}" == "dl" ]]; then slug=$(resolve_dl_slug "${config}" "${scheme}")
      else                                slug=$(resolve_fpca_slug "${config}"); fi
      echo ">> ${CLASS}/${scheme}: ${config}  slug=${slug}"

      if [[ "${scheme}" == "cv_0_00" ]]; then envs=("${HELDOUT_ENVS[@]}"); else envs=("__none__"); fi

      for pair in "${SEED_PAIRS[@]}"; do
        cv_seed="${pair%%:*}"; model_seed="${pair##*:}"
        for fold in "${FOLDS[@]}"; do
          for env in "${envs[@]}"; do
            token=$(fold_token "${cv_seed}" "${fold}" "${scheme}" "${env}")
            out="${ARTIFACT_BASE}/${CLASS}/${scheme}/${token}/${slug}/seed=${model_seed}"
            if [[ "${ENGINE}" == "dl" ]]; then marker="${out}/curve.done"; else marker="${out}/metrics.json"; fi
            if [[ -f "${marker}" ]]; then
              echo "SKIP   ${CLASS}/${scheme}/${token}/seed=${model_seed}"; (( ++SKIPPED )); continue
            fi

            EXPORTS="SCHEME=${scheme},CV_SEED=${cv_seed},FOLD=${fold},CONFIG=${config},SLUG=${slug},SEED=${model_seed},OUT_DIR=${out}"
            [[ "${scheme}" == "cv_0_00" ]] && EXPORTS="${EXPORTS},HELDOUT_ENV=${env}"
            [[ -n "${FOLD_CSV}" ]]         && EXPORTS="${EXPORTS},FOLD_CSV=${FOLD_CSV}"
            [[ -n "${SMOKE_N}" ]]          && EXPORTS="${EXPORTS},SMOKE_N=${SMOKE_N}"

            if [[ "${ENGINE}" == "dl" ]]; then
              epochs="${EPOCHS_OVERRIDE:-${EPOCHS}}"
              EXPORTS="${EXPORTS},MODEL_CLASS=${CLASS},EVAL_STREAMS=${scheme},EPOCHS=${epochs},FREEZE_PATIENCE=${PATIENCE},WANDB_PROJECT=${PROJECT}-${scheme},WANDB_LOGGING=${WANDB}"
              job="${DL_JOB}"
              res=(--time="${TIME}" --cpus-per-task="${CPUS}" --mem="${MEM}" --gres="${GRES}" --partition="${PARTITION}")
              jname="dl_${CLASS%%_*}_${scheme}_${token}_${slug}_s${model_seed}"
            else
              extra=$(fpca_extra_overrides "${config}")
              [[ -n "${extra}" ]] && EXPORTS="${EXPORTS},EXTRA_OVERRIDES=${extra}"
              EXPORTS="${EXPORTS},WANDB_LOGGING=${WANDB}"
              job="${FPCA_JOB}"
              res=(--time="${TIME}" --cpus-per-task="${CPUS}" --mem="${MEM}")
              jname="fpca_${scheme}_${token}_${slug}_s${model_seed}"
            fi

            if [[ -n "${DRY_RUN}" ]]; then
              echo "DRY    ${CLASS}/${scheme}/${token}/${slug}/seed=${model_seed}  (${res[*]})"
              (( ++SUBMITTED )); continue
            fi
            if submit_with_retry \
                --job-name="${jname}" \
                --account="${ACCOUNT}" \
                --nodes=1 --ntasks-per-node=1 \
                "${res[@]}" \
                --output="${LOG_DIR}/%x.txt" \
                --mail-type=FAIL --mail-user="${SLURM_MAIL_USER}" \
                --export="${EXPORTS}" \
                "${job}"; then
              echo "SUBMIT ${CLASS}/${scheme}/${token}/${slug}/seed=${model_seed}"; (( ++SUBMITTED ))
            else
              echo "FAILED ${CLASS}/${scheme}/${token}/seed=${model_seed}"; (( ++FAILED ))
            fi
            sleep 0.2
          done
        done
      done
    done
  done
}

# ── run_class: load a spec block (by prefix) into the working vars, then run ──
run_class() {  # $1=prefix (NP/FPCA)  $2=class_name
  local p="$1" v cref
  CLASS="$2"
  v="${p}_ENGINE";   ENGINE="${!v}"
  v="${p}_TIME";     TIME="${!v}"
  v="${p}_GRES";     GRES="${!v:-}"
  v="${p}_EPOCHS";   EPOCHS="${!v:-}"
  v="${p}_PATIENCE"; PATIENCE="${!v:-}"
  v="${p}_PROJECT";  PROJECT="${!v:-}"
  cref="${p}_CONFIGS[@]"; CONFIGS=("${!cref}")
  submit_class
}

# ── Plan banner ───────────────────────────────────────────────────────────────
echo "============================================================"
echo "CV_0_00_2_1 — UNIFIED train/test submission"
echo "============================================================"
echo "Root:      ${ARTIFACT_BASE}/<class>/<scheme>/<token>/<slug>/seed=<n>"
echo "Schemes:   ${SCHEMES[*]}    folds: ${FOLDS[*]}   seed_pairs(cv:model): ${SEED_PAIRS[*]}"
echo "Fold src:  $([[ -n "${FOLD_CSV}" ]] && echo "r_csv (${FOLD_CSV})" || echo native)"
echo "Account:   ${ACCOUNT}"
[[ -n "${ONLY_CLASS}"  ]] && echo "*** FILTER class=${ONLY_CLASS} ***"
[[ -n "${ONLY_CONFIG}" ]] && echo "*** FILTER config=${ONLY_CONFIG} ***"
[[ -n "${SMOKE_N}"     ]] && echo "*** SMOKE: smoke_n=${SMOKE_N} ***"
[[ -n "${DRY_RUN}"     ]] && echo "*** DRY RUN — no sbatch will be issued ***"
echo "------------------------------------------------------------"

[[ -z "${DRY_RUN}" ]] && { exec > >(stdbuf -oL tee -a "${LOG_DIR}/submit_$(date +%Y%m%dT%H%M%S)_$$.log") 2>&1; }

# ── Fan-out (drive each class block in CLASS_ORDER) ───────────────────────────
SUBMITTED=0; SKIPPED=0; FAILED=0
for entry in "${CLASS_ORDER[@]}"; do
  run_class "${entry%%:*}" "${entry##*:}"
done

echo "------------------------------------------------------------"
echo "$([[ -n "${DRY_RUN}" ]] && echo Planned || echo Submitted): ${SUBMITTED}  Skipped: ${SKIPPED}  Failed: ${FAILED}"
[[ ${FAILED} -gt 0 ]] && exit 1 || exit 0
