#!/bin/bash
# =======================================================================
# submit_hpo_sota.sh
# Optuna HPO for SOTA atom-bond-global graph — 3 modes in parallel:
#   task 0 → baseline (atom=13, bond=7)
#   task 1 → qtaim    (atom=31, bond=26)
#   task 2 → elf      (atom=18, bond=12)
# =======================================================================
#SBATCH -A ihj@h100
#SBATCH -p gpu_p6
#SBATCH -C h100
#SBATCH -J hpo_sota_graph_corrected
#SBATCH -t 48:00:00
#SBATCH --array=0-2
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH -o /lustre/fsn1/projects/rech/ihj/%u/qtaim_embed_private/logs_suba/%x_%A_%a.out
#SBATCH -e /lustre/fsn1/projects/rech/ihj/%u/qtaim_embed_private/logs_suba/%x_%A_%a.err
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --export=ALL


set -eo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true

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
export WANDB_MODE=offline

SOTA_ROOT=/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private
SCRIPT=$SOTA_ROOT/qtaim_embed/scripts/train/hpo_sota_graph.py
RESULTS=$SOTA_ROOT/results/hpo_sota

MODES=("baseline" "qtaim" "elf")
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
OUTDIR=$RESULTS/$MODE

mkdir -p "$OUTDIR"

cd "$SOTA_ROOT"

echo "========================================"
echo "Job array task: $SLURM_ARRAY_TASK_ID"
echo "mode:           $MODE"
echo "Output:         $OUTDIR"
echo "========================================"

python "$SCRIPT" \
    --mode      "$MODE" \
    --sota_root "$SOTA_ROOT" \
    --outdir    "$OUTDIR" \
    --n_trials  60

echo "HPO done: $OUTDIR/best.json"