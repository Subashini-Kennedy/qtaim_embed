#!/bin/bash
#SBATCH -A ihj@h100
#SBATCH -p gpu_p6
#SBATCH -C h100
#SBATCH -J lmdb_qm9_multi
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=1
#SBATCH -o /lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/logs_suba/%x_%j.out
#SBATCH -e /lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/logs_suba/%x_%j.err

set -eo pipefail

module purge
module load miniforge/24.11.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_proj/env/qtaim_embed
source /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_proj/env/activate_qtaim.sh


# Make sure sitecustomize.py (with the pymatgen patch) is found
export PYTHONNOUSERSITE=1
export PYTHONPATH=/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_proj/code:$PYTHONPATH


# keep native libs tame; avoid MKL var error
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DGL_NUM_THREADS=1
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64,GNU}
export TMPDIR="${JOBSCRATCH:-/tmp}"
ulimit -n 4096

# paths
CFG="/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/configs_suba/settings_qm9_multi.json"
TRAIN_PKL="/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/data_suba/train_qm9_qtaim_1205_labelled_corrected.pkl"
TEST_PKL="/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/data_suba/test_qm9_qtaim_1205_labelled_corrected.pkl"
OUT_ROOT="/lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/data_suba/lmdb/qm9_lmdb_multi"

mkdir -p "$OUT_ROOT/train" "$OUT_ROOT/test"

echo ">>> train LMDB (HOMO,LUMO,GAP)…"
python -u "$(which mol2lmdb.py)" \
  -dataset_loc "$TRAIN_PKL" \
  -config "$CFG" \
  -lmdb_dir "$OUT_ROOT/train/"

echo ">>> test LMDB (HOMO,LUMO,GAP)…"
python -u "$(which mol2lmdb.py)" \
  -dataset_loc "$TEST_PKL" \
  -config "$CFG" \
  -lmdb_dir "$OUT_ROOT/test/"

echo ">>> LMDB contents:"
find "$OUT_ROOT" -maxdepth 2 -type f -name 'molecule.lmdb*' -printf '%p\t%k KB\n' | sort

echo "DONE."
