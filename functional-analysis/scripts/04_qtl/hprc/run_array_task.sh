#!/usr/bin/env bash
set -euo pipefail

ANALYSIS=${1:?Usage: run_array_task.sh ANALYSIS scan|downstream}
STAGE=${2:?Usage: run_array_task.sh ANALYSIS scan|downstream}
case "${ANALYSIS}" in
  dap|agdd|flowering_yield) ;;
  *) echo "Unsupported analysis: ${ANALYSIS}" >&2; exit 2 ;;
esac
case "${STAGE}" in
  scan) R_SCRIPT="hprc_scan.R" ;;
  downstream) R_SCRIPT="hprc_downstream.R" ;;
  *) echo "Unsupported stage: ${STAGE}" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export G2F_PROJECT_DIR=${G2F_PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
ENV_TESTER_FILE=${G2F_QTL_ENV_TESTER_FILE:-${SCRIPT_DIR}/env_testers.txt}
TASK_ID=${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}
ENV_TESTER=$(sed -n "${TASK_ID}p" "${ENV_TESTER_FILE}")
if [[ -z "${ENV_TESTER}" ]]; then
  echo "No environment-tester at row ${TASK_ID} of ${ENV_TESTER_FILE}" >&2
  exit 2
fi

if [[ -z "${G2F_QTL_ROOT:-}" && -z "${SCRATCH:-}" &&
      ( -z "${G2F_QTL_INPUT_DIR:-}" || -z "${G2F_QTL_OUTPUT_DIR:-}" ) ]]; then
  echo "Set SCRATCH, G2F_QTL_ROOT, or both G2F_QTL_INPUT_DIR and G2F_QTL_OUTPUT_DIR." >&2
  exit 2
fi
QTL_ROOT=${G2F_QTL_ROOT:-${SCRATCH:-}/g2f-multimodal-qtl}
export G2F_QTL_INPUT_DIR=${G2F_QTL_INPUT_DIR:-${QTL_ROOT}/${ANALYSIS}/input}
export G2F_QTL_OUTPUT_DIR=${G2F_QTL_OUTPUT_DIR:-${QTL_ROOT}/${ANALYSIS}/output}
mkdir -p "${G2F_QTL_INPUT_DIR}" "${G2F_QTL_OUTPUT_DIR}"

module purge
module load GCC/13.3.0 OpenMPI/5.0.3 R_tamu/4.4.2
R_ENV_ARGS=()
if [[ -n "${G2F_R_ENVIRONMENT:-}" ]]; then
  R_ENV_ARGS+=("--rtamuenvs=${G2F_R_ENVIRONMENT}")
fi

echo "Slurm task ${TASK_ID}: ${STAGE} ${ANALYSIS} QTL for ${ENV_TESTER}"
Rscript "${R_ENV_ARGS[@]}" \
  "${G2F_PROJECT_DIR}/scripts/04_qtl/${ANALYSIS}/${R_SCRIPT}" \
  "${ENV_TESTER}"
