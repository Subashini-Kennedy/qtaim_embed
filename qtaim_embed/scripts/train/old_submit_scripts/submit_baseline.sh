#!/bin/bash
#SBATCH -A ihj@h100
#SBATCH -p gpu_p6
#SBATCH -C h100
#SBATCH -J train_qm9_baseline
#SBATCH -t 18:00:00
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

#  path to your env
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

# FIXED: working directory where your script and configs live
cd /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private

# Run outputs
PERSIST_RUN_DIR="/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/logs_suba/train_qm9_baseline_${SLURM_JOB_ID}"
LOG_DIR="${PERSIST_RUN_DIR}/logs"
WANDB_DIR="${PERSIST_RUN_DIR}/wandb"
mkdir -p "$LOG_DIR" "$WANDB_DIR"

export WANDB_MODE=offline
export WANDB_DIR

# Config file
CFG="/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/configs_suba/settings_qm9_nonlmdb_43k_withval_baseline.json"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found at $CFG" >&2; exit 1
fi

# FIXED: call your script directly, not as a module
srun --ntasks-per-node=1 --cpu-bind=none python -u \
  /lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/qtaim_embed/scripts/train/train_qtaim_graph_suba.py \
  -config "${CFG}" \
  -project_name "qm9_43k_baseline" \
  -log_save_dir "${LOG_DIR}" \
  -dataset_test_loc "/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/data_suba/filtered_qtaim_fullqm9/test_43k.pkl" \
  -wandb_entity "subashini-kennedy"
