#!/bin/bash
#SBATCH -A ihj@h100
#SBATCH -p gpu_p6
#SBATCH -C h100
#SBATCH -J split_qtaim_43k_fullqm9
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH -o /lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/logs_suba/%x_%j.out
#SBATCH -e /lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/logs_suba/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

set -eo pipefail

module purge
module load miniforge/24.11.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/env/qtaim_embed

PROJECT=/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private
DATA=$PROJECT/data_suba
SCRIPT=$PROJECT/preprocess_43K_suba/filter_qm9_43k_train_test_val.py
OUTDIR=$DATA/filtered_qtaim_fullqm9
LOGDIR=$PROJECT/logs_suba

mkdir -p "$LOGDIR"
mkdir -p "$OUTDIR"

echo "Job started on $(hostname) at $(date)"
echo "Python: $(which python)"

python "$SCRIPT" \
  --train-pkl "$DATA/train_qm9_qtaim_1205_labelled_corrected.pkl" \
  --test-pkl "$DATA/test_qm9_qtaim_1205_labelled_corrected.pkl" \
  --split-csv "$DATA/qm9_43k_clean_with_val.csv" \
  --pickle-id-field names \
  --csv-id-field GDB_Index \
  --split-field split \
  --outdir "$OUTDIR" \
  --suffix _43k \
  --write-csv

echo "Job finished at $(date)"