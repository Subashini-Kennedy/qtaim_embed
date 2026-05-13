#!/bin/bash
#SBATCH -A ihj@h100
#SBATCH -p gpu_p6
#SBATCH -C h100
#SBATCH -J train_qm9_multi_43k
#SBATCH -t 18:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH -o /lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/logs_suba/%x_%j.out
#SBATCH -e /lustre/fsn1/projects/rech/ihj/urb54jd/qtaim_embed_private/logs_suba/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

set -eo pipefail

# ---------------- Env ----------------
module purge
module load miniforge/24.11.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_proj/env/qtaim_embed

# Tame BLAS threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DGL_NUM_THREADS=1
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64,GNU}
export PYTHONFAULTHANDLER=1
export PYTORCH_SHOW_WORKER_ERRORS=1
export PYTHONNOUSERSITE=1
export TMPDIR="${JOBSCRATCH:-/tmp}"

# ---------------- Paths ----------------
cd /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private
export PYTHONPATH=/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private:$PYTHONPATH

# Scratch (node-local) for fast I/O of LMDB only
export SCRATCH_ROOT="${JOBSCRATCH:-/tmp}/${USER}"
export SCRATCH_LMDB="${SCRATCH_ROOT}/qm9_lmdb_multi"
export LOC_CFG="${SCRATCH_ROOT}/configs_suba/settings_qm9_multi_local.json"

# Persistent outputs — EVERYTHING goes under logs_suba (no /runs)
export PERSIST_RUN_DIR="/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/logs_suba/train_qm9_multi_${SLURM_JOB_ID}"
export LOG_DIR="${PERSIST_RUN_DIR}/logs"
export CKPT_DIR="${PERSIST_RUN_DIR}/checkpoints"
export WANDB_DIR="${PERSIST_RUN_DIR}/wandb"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$WANDB_DIR"

# Weights & Biases (still offline, but stored under logs_suba)
export WANDB_MODE=offline
unset WANDB_DISABLED
export WANDB_DIR

echo "PYTHON      : $(command -v python)"
echo "SCRATCH_ROOT: ${SCRATCH_ROOT}"
echo "PERSIST_RUN_DIR : ${PERSIST_RUN_DIR}"
echo "LOG_DIR          : ${LOG_DIR}"
echo "CKPT_DIR         : ${CKPT_DIR}"
echo "WANDB_DIR        : ${WANDB_DIR}"
python - <<'PY'
import torch
print("CUDA avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

# ---------------- Config selection ----------------
FSN1_CFG="/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/configs_suba/settings_qm9_multi.json"
if [[ -f "$FSWORK_CFG" ]]; then
  export SRC_CFG="$FSWORK_CFG"
elif [[ -f "$FSN1_CFG" ]]; then
  export SRC_CFG="$FSN1_CFG"
else
  echo "ERROR: settings_qm9_multi.json not found." >&2
  exit 1
fi
echo "Using SRC_CFG: $SRC_CFG"

# Ensure scratch dirs exist for LMDB staging + local cfg
mkdir -p "$(dirname "$LOC_CFG")" "$SCRATCH_LMDB"

# -------- Stage LMDBs to scratch for speed --------
rsync -a --delete \
  "/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/data_suba/lmdb/qm9_lmdb_multi_43k/" \
  "${SCRATCH_LMDB}/"

# Optional resume: pick a checkpoint under logs_suba first
CKPT=$(ls -t \
  "${CKPT_DIR}"/*.ckpt \
  /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/checkpoints_suba/*.ckpt \
  2>/dev/null | head -n1 || true)
[[ -n "$CKPT" ]] && echo "Seeding resume checkpoint from: $CKPT"

# -------- Rewrite cfg: LMDBs -> scratch; logs/ckpts -> logs_suba --------
python - <<'PY'
import os, json
src  = os.environ["SRC_CFG"]
dst  = os.environ["LOC_CFG"]
root = os.environ["SCRATCH_LMDB"]
log_dir = os.environ["LOG_DIR"]
ckpt_dir = os.environ["CKPT_DIR"]
ckpt = os.environ.get("CKPT", "")

cfg = json.load(open(src))

def to_scratch(path):
    # path like .../train/molecule.lmdb or .../test/molecule.lmdb
    split_dir = os.path.basename(os.path.dirname(path))   # 'train' or 'test'
    return os.path.join(root, split_dir, "molecule.lmdb")

for k in ("train_lmdb","val_lmdb","test_lmdb"):
    cfg["dataset"][k] = to_scratch(cfg["dataset"][k])

# Persist outputs directly under logs_suba (no /runs)
cfg["dataset"]["log_save_dir"] = log_dir
cfg["checkpoint_dir"]          = ckpt_dir

cfg.setdefault("model", {})
cfg["model"].setdefault("target_dict", {"global": ["homo","lumo","gap"]})
cfg["model"].setdefault("ntasks", 3)

if ckpt:
    cfg["model"]["restore"] = True
    cfg["model"]["restore_path"] = ckpt

os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(cfg, open(dst, "w"), indent=2)
print("Using config:", dst)
print("train_lmdb :", cfg["dataset"]["train_lmdb"])
print("val_lmdb   :", cfg["dataset"]["val_lmdb"])
print("test_lmdb  :", cfg["dataset"]["test_lmdb"])
print("log_save_dir:", cfg["dataset"]["log_save_dir"])
print("checkpoint_dir:", cfg["checkpoint_dir"])
PY

# ---------------- Run ----------------
srun --cpu-bind=none python -u -m qtaim_embed.scripts.train.train_qtaim_graph_43k \
  -config "${LOC_CFG}" \
  -project_name "qm9_qtaim_multi_43k" \
  -log_save_dir "${LOG_DIR}" \
  --use_lmdb
