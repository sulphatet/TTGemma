#!/usr/bin/env bash
#SBATCH -J tiny_gemma_rsync
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /home2/%u/logs/%x.%j.out
#SBATCH -e /home2/%u/logs/%x.%j.err

set -euo pipefail

echo "=========================================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-NA}"
echo "SLURM_NODELIST = ${SLURM_NODELIST:-NA}"
echo "SLURM_GPUS     = ${SLURM_JOB_GPUS:-NA}"
echo "CUDA_VISIBLE_DEVICES (pre) = ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

# ---- User paths
USER_DIR="/home2/$USER/LMA_SLM"
PY="$USER_DIR/scripts/pretrain_tiny_gemma.py"
SPM="$USER_DIR/tokenizers/sp_unigram_64000.model"

# Data source on the login host
SRC_SPLITS="ada:/share1/$USER/LMA_SLM/data/splits/"

# Node-local working dir (prefer /scratch, else /tmp)
NODE_SCRATCH_BASE="/scratch/$USER"
[[ -d "$NODE_SCRATCH_BASE" ]] || NODE_SCRATCH_BASE="/tmp/$USER"
WORKDIR="$NODE_SCRATCH_BASE/LMA_SLM/${SLURM_JOB_ID:-manual}"

SAVE_ROOT="$USER_DIR/checkpoints"
LEDGER="$USER_DIR/training/token_budget.json"
LOG_DIR="/home2/$USER/logs"
mkdir -p "$WORKDIR" "$SAVE_ROOT" "$LOG_DIR" || true

echo "[paths] WORKDIR   = $WORKDIR"
echo "[paths] SAVE_ROOT = $SAVE_ROOT"
echo "[paths] LEDGER    = $LEDGER"
echo "[paths] SPM       = $SPM"

# ---- Environment
# ---- Environment
# Make sure no site CUDA modules leak in
module purge            # <— important: removes cuda/12.x etc.
module load rsync 2>/dev/null || true

source ~/.bashrc
conda activate lma

# extra safety: don’t let system CUDA override PyTorch’s CUDA
unset CUDA_HOME CUDA_PATH
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -E '/cuda-|/cudnn|/nvidia|/nvhpc' | paste -sd:)
fi

# Show what’s loaded now (should be basically empty)
module list || true


# ---- Stage-in
DEST_SPLITS="$WORKDIR/data/splits"
mkdir -p "$DEST_SPLITS"
echo "[rsync] pulling splits from $SRC_SPLITS -> $DEST_SPLITS"
time rsync -avh --delete --partial --inplace --info=stats2,progress2 \
  "$SRC_SPLITS" "$DEST_SPLITS/"
du -sh "$DEST_SPLITS"/* 2>/dev/null || true

# Tokenizer local copy
mkdir -p "$WORKDIR/tokenizers"
cp -n "$SPM" "$WORKDIR/tokenizers/" || true
SPM_LOCAL="$WORKDIR/tokenizers/$(basename "$SPM")"

# ---- Quick GPU diagnostics (BEFORE training)
echo "----- GPU diagnostics -----"
echo "CUDA_VISIBLE_DEVICES (post) = ${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
python - <<'PY'
import torch, os
print("[torch] version:", torch.__version__)
print("[torch.cuda.is_available]:", torch.cuda.is_available())
print("[torch.cuda.device_count]:", torch.cuda.device_count())
print("[env CUDA_VISIBLE_DEVICES]:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("[torch] device name:", torch.cuda.get_device_name(0))
PY
echo "---------------------------"

# If no GPU visible, fail early with a helpful message
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("[FATAL] No GPU visible inside job. This is usually one of:")
    print("  - payload not launched with `srun` so cgroups weren’t applied")
    print("  - CPU-only node, or old/incompatible NVIDIA driver for your PyTorch CUDA build")
    print("  - mis-set CUDA_VISIBLE_DEVICES")
    sys.exit(42)
PY

# ---- Hyperparams (overridable via --export)
RUN_NAME="${RUN_NAME:-tiny-1p5M-sbatch}"
MODEL_PRESET="${MODEL_PRESET:-1p5M}"
SEQ_LEN="${SEQ_LEN:-512}"
GBTOK="${GBTOK:-16384}"
MICRO_BSZ="${MICRO_BSZ:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-0}"
MAX_STEPS="${MAX_STEPS:-200}"
VAL_INT="${VAL_INT:-50}"
CKPT_INT="${CKPT_INT:-50}"
AMP="${AMP:-fp16}"

# ---- Launch training WITH srun so GPU bindings apply
# --gpu-bind=single:1 is a nice extra hint on some setups.
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     --gres=gpu:1 --cpu-bind=none \
  python "$PY" \
    --spm_model_path "$SPM_LOCAL" \
    --splits_root   "$DEST_SPLITS" \
    --run_name      "$RUN_NAME" \
    --model_preset  "$MODEL_PRESET" \
    --seq_len       "$SEQ_LEN" \
    --global_batch_tokens "$GBTOK" \
    --micro_bsz     "$MICRO_BSZ" \
    --grad_accum    "$GRAD_ACCUM" \
    --max_steps     "$MAX_STEPS" \
    --val_interval  "$VAL_INT" \
    --ckpt_interval "$CKPT_INT" \
    --amp_dtype     "$AMP" \
    --save_to       "$SAVE_ROOT" \
    --token_budget_ledger "$LEDGER"

echo "[done] Training finished."
