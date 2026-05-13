#!/bin/bash
# =======================================================================
# submit_elf.sh
# SOTA baseline graph + ELF features on atom AND bond nodes
# atom features: 13 topology + 5 ELF = 18
# bond features: 7 topology + 5 ELF = 12
# Run after submit_preprocess_elf.sh completes
# =======================================================================
#SBATCH -A ihj@h100
#SBATCH -p gpu_p6
#SBATCH -C h100
#SBATCH -J train_sota_elf_corrected
#SBATCH -t 48:00:00
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH -o /lustre/fsn1/projects/rech/ihj/%u/qtaim_embed_private/logs_suba/%x_%j.out
#SBATCH -e /lustre/fsn1/projects/rech/ihj/%u/qtaim_embed_private/logs_suba/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

set -eo pipefail

module purge
module load miniforge/24.11.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/env/qtaim_embed

touch /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/qtaim_embed/__init__.py
export PYTHONPATH=/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private:$PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DGL_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
export PYTORCH_SHOW_WORKER_ERRORS=1
export PYTHONNOUSERSITE=1
export TMPDIR="${JOBSCRATCH:-/tmp}"

cd /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private

PERSIST_RUN_DIR="/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/logs_suba/train_43k_elf_corrected${SLURM_JOB_ID}"
LOG_DIR="${PERSIST_RUN_DIR}/logs"
WANDB_DIR="${PERSIST_RUN_DIR}/wandb"
mkdir -p "$LOG_DIR" "$WANDB_DIR"

export WANDB_MODE=offline
export WANDB_DIR

CFG="/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/configs_suba/settings_qm9_nonlmdb_43k_withval_baseline_elf_atom_bond.json"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found at $CFG" >&2; exit 1
fi

echo "========================================"
echo "Job:     $SLURM_JOB_ID"
echo "Config:  $CFG"
echo "Option:  C — ELF on atom + bond nodes (atom=18, bond=12)"
echo "========================================"

srun --ntasks-per-node=1 --cpu-bind=none python -u \
  /lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/qtaim_embed/scripts/train/train_qtaim_graph_suba.py \
  -config "${CFG}" \
  -project_name "qm9_43k_elf_atom_bond" \
  -log_save_dir "${LOG_DIR}" \
  -dataset_test_loc "/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/data_suba/filtered_qtaim_fullqm9_elf_corrected/test_43k_elf.pkl" \
  -model_name "elf" \
  -wandb_entity "subashini-kennedy"