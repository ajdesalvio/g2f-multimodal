#!/usr/bin/env bash
#SBATCH --job-name=g2f_qtl_dap_downstream
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=short
#SBATCH --array=1-29%29
#SBATCH --output=%x.out.%A_%a
#SBATCH --error=%x.err.%A_%a

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "${SCRIPT_DIR}/../run_array_task.sh" dap downstream
