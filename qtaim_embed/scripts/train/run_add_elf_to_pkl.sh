#!/bin/bash
# =======================================================================
# submit_preprocess_elf.sh
# Step 1: Add ELF features to SOTA PKL files — Option C only
#         (ELF on atom nodes + bond nodes)
#
# Runs verify first to print the full mapping for one sample molecule,
# then processes all splits (train/val/test).
#
# Output: data_suba/filtered_qtaim_fullqm9_elf_corrected/elf/
# =======================================================================
#SBATCH -A ihj@h100
#SBATCH -p gpu_p6
#SBATCH -C h100
#SBATCH -J add_elf_to_pkl
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=0
#SBATCH -o /lustre/fsn1/projects/rech/ihj/%u/qtaim_embed_private/logs_suba/%x_%j.out
#SBATCH -e /lustre/fsn1/projects/rech/ihj/%u/qtaim_embed_private/logs_suba/%x_%j.err

set -eo pipefail

module purge
module load miniforge/24.11.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/env/qtaim_embed

export PYTHONNOUSERSITE=1
export PYTHONPATH=/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private:$PYTHONPATH

PKL_DIR=/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/data_suba/filtered_qtaim_fullqm9_corrected
JSON_DIR=/lustre/fsn1/projects/rech/ihj/$USER/gnn/control_and_critical_points_GNNs/data/criticalpoints_jsonfiles
ELF_CSV=/lustre/fsn1/projects/rech/ihj/$USER/gnn/control_and_critical_points_GNNs/data/qm9_43k_clean_with_val.csv
OUTDIR=/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/data_suba/filtered_qtaim_fullqm9_elf_corrected
SCRIPT=/lustre/fsn1/projects/rech/ihj/$USER/qtaim_embed_private/qtaim_embed/scripts/train/add_elf_to_pkl.py

mkdir -p "$OUTDIR"

ARGS="--pkl_dir $PKL_DIR --json_dir $JSON_DIR --elf_csv $ELF_CSV --outdir $OUTDIR"

echo "========================================"
echo "Job:    $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Output: $OUTDIR/elf/"
echo "========================================"

echo ""
echo ">>> STEP 1: VERIFY — printing full mapping for one sample molecule"
echo "========================================"
python "$SCRIPT" --mode verify $ARGS

echo ""
echo ">>> STEP 2: PROCESS — enriching all PKL splits"
echo "========================================"
python "$SCRIPT" --mode process $ARGS

echo "========================================"
echo "Done."
echo "Next: sbatch submit_elf.sh"
echo "========================================"
