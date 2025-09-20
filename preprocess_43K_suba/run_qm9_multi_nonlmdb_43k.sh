#!/bin/bash
#SBATCH -A ihj@h100
#SBATCH -p gpu_p6
#SBATCH -C h100
#SBATCH -J train_qm9_multi_nonlmdb_43k
#SBATCH -t 18:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH -o /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/logs_suba/%x_%j.out
#SBATCH -e /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/logs_suba/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

set -eo pipefail

module purge
module load miniforge/24.11.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_proj/env/qtaim_embed

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
export PYTHONPATH=/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private:$PYTHONPATH

# Outputs
export PERSIST_RUN_DIR="/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/logs_suba/train_qm9_multi_nonlmdb_43k_${SLURM_JOB_ID}"
export LOG_DIR="${PERSIST_RUN_DIR}/logs"
export WANDB_DIR="${PERSIST_RUN_DIR}/wandb"
mkdir -p "$LOG_DIR" "$WANDB_DIR"

export WANDB_MODE=offline
unset WANDB_DISABLED
export WANDB_DIR

# Config
SRC_CFG="/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/configs_suba/settings_qm9_nonlmdb_43k.json"
if [[ ! -f "$SRC_CFG" ]]; then
  echo "ERROR: $SRC_CFG not found" >&2; exit 1
fi

# Optional overrides via sbatch --export=ALL,OVERRIDE_TRAIN=...,OVERRIDE_TEST=...
OVERRIDE_TRAIN="${OVERRIDE_TRAIN:-}"
OVERRIDE_TEST="${OVERRIDE_TEST:-}"

# Create runtime-local cfg (adjust log path and optional overrides)
LOC_CFG="${JOBSCRATCH:-/tmp}/${USER}/configs_suba/settings_qm9_nonlmdb_43k_local.json"
mkdir -p "$(dirname "$LOC_CFG")"

python - <<'PY'
import os, json
src  = os.environ["SRC_CFG"]
dst  = os.environ["LOC_CFG"]
logd = os.environ["LOG_DIR"]
over_train = os.environ.get("OVERRIDE_TRAIN","")
over_test  = os.environ.get("OVERRIDE_TEST","")

cfg = json.load(open(src))
ds = cfg.setdefault("dataset", {})
# ensure non-LMDB
for k in ("train_lmdb","val_lmdb","test_lmdb"): ds.pop(k, None)
if over_train: ds["train_dataset_loc"] = over_train
if over_test:  ds["test_dataset_loc"]  = over_test
# disable internal test split; use external
ds["test_prop"] = 0.0
ds.setdefault("val_prop", 0.10)
ds["log_save_dir"] = logd

opt = cfg.setdefault("optim", {})
if opt.get("precision") in ("16","32"):
    opt["precision"] = int(opt["precision"])

os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(cfg, open(dst, "w"), indent=2)
print("Using config:", dst)
print("train_dataset_loc:", ds.get("train_dataset_loc"))
print("test_dataset_loc :", ds.get("test_dataset_loc"))
print("val_prop/test_prop:", ds.get("val_prop"), ds.get("test_prop"))
print("log_save_dir     :", ds.get("log_save_dir"))
PY

# Run (no --use_lmdb). Pass dataset paths explicitly for clarity.
TRAIN_PATH=$(jq -r '.dataset.train_dataset_loc' "${LOC_CFG}")
TEST_PATH=$(jq -r '.dataset.test_dataset_loc'  "${LOC_CFG}")

srun --cpu-bind=none python -u -m qtaim_embed.scripts.train.train_qtaim_graph_nonlmdb_43k \
  -config "${LOC_CFG}" \
  -project_name "qm9_qtaim_multi_nonlmdb_43k" \
  -log_save_dir "${LOG_DIR}" \
  -dataset_loc "${TRAIN_PATH}" \
  -dataset_test_loc "${TEST_PATH}"