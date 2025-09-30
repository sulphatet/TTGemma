#!/usr/bin/env bash
#SBATCH -J gemma150M_3B
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --exclude=gnode008
#SBATCH --constraint=2080ti
#SBATCH --mem=64G
#SBATCH -t 2-20:00:00
#SBATCH -o /home2/%u/logs/%x.%j.out
#SBATCH -e /home2/%u/logs/%x.%j.err

set -euo pipefail

echo "=========================================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-NA}"
echo "SLURM_NODELIST = ${SLURM_NODELIST:-NA}"
echo "CUDA_VISIBLE_DEVICES (pre) = ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

# ---- Paths
USER_DIR="/home2/$USER/LMA_SLM"
PY="$USER_DIR/scripts/train_gemma_150m_ddp.py"
SPM="$USER_DIR/tokenizers/sp_unigram_64000.model"
SRC_SPLITS="ada:/share1/$USER/LMA_SLM/data/splits/"

NODE_SCRATCH_BASE="/scratch/$USER"
[[ -d "$NODE_SCRATCH_BASE" ]] || NODE_SCRATCH_BASE="/tmp/$USER"
WORKDIR="$NODE_SCRATCH_BASE/LMA_SLM/${SLURM_JOB_ID:-manual}"

SRC_PACKED="ada:/share1/$USER/LMA_SLM/packed/seq1024/"
DEST_PACKED="$WORKDIR/packed/seq1024"
mkdir -p "$DEST_PACKED"
rsync -avh --delete --partial --inplace --info=stats2,progress2 \
  "$SRC_PACKED" "$DEST_PACKED/"

SAVE_ROOT="$USER_DIR/checkpoints"
LEDGER="$USER_DIR/training/token_budget.json"
LOG_DIR="/home2/$USER/logs"
mkdir -p "$WORKDIR" "$SAVE_ROOT" "$LOG_DIR" || true

echo "[paths] WORKDIR   = $WORKDIR"
echo "[paths] SAVE_ROOT = $SAVE_ROOT"
echo "[paths] LEDGER    = $LEDGER"
echo "[paths] SPM       = $SPM"

# ---- Environment
module purge
module load rsync 2>/dev/null || true

source ~/.bashrc
conda activate lma

# Avoid toolchain conflicts but keep NVML available
unset CUDA_HOME CUDA_PATH
# DO NOT scrub LD_LIBRARY_PATH anymore (NVML needed for nvidia-smi/torch queries)

# NCCL single-node safety
export NCCL_DEBUG=VERSION
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
# If IB issues appear, uncomment:
# export NCCL_IB_DISABLE=1

# CPU threads – avoid oversubscription
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# NCCL: single-node sanity; IB often flaky/slow on mixed nodes
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_MIN_NRINGS=4

# PyTorch SDPA: force math path on Pascal/Turing if you see kernel thrash
export PYTORCH_SDP_KERNEL=math
# (Remove that line on 2080 Ti if you measure better with default; keep if stable throughput is more important.)


# ---- Stage-in to node-local
DEST_SPLITS="$WORKDIR/data/splits"
mkdir -p "$DEST_SPLITS"
echo "[rsync] pulling splits from $SRC_SPLITS -> $DEST_SPLITS"
time rsync -avh --delete --partial --inplace --info=stats2,progress2 \
  "$SRC_SPLITS" "$DEST_SPLITS/"
du -sh "$DEST_SPLITS"/* 2>/dev/null || true

mkdir -p "$WORKDIR/tokenizers"
cp -n "$SPM" "$WORKDIR/tokenizers/" || true
SPM_LOCAL="$WORKDIR/tokenizers/$(basename "$SPM")"

# ---- GPU detection (skip broken adapters)
AVAILABLE_GPU_IDX=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//')
if [[ -z "${AVAILABLE_GPU_IDX}" ]]; then
  echo "[FATAL] nvidia-smi returned no GPUs."; exit 42
fi
export CUDA_VISIBLE_DEVICES="${AVAILABLE_GPU_IDX}"

# Quick diag
echo "----- GPU diagnostics -----"
echo "CUDA_VISIBLE_DEVICES (post) = ${CUDA_VISIBLE_DEVICES}"
nvidia-smi || true
python - <<'PY'
import torch, os
print("[torch] version:", torch.__version__)
print("[torch.cuda.is_available]:", torch.cuda.is_available())
print("[torch.cuda.device_count]:", torch.cuda.device_count())
print("[env CUDA_VISIBLE_DEVICES]:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[torch] GPU{i}:", torch.cuda.get_device_name(i))
PY
echo "---------------------------"

# Decide how many processes to launch
NUM_GPUS=$(python - <<'PY'
import torch, sys
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "[FATAL] No usable GPU visible inside job after remap."; exit 42
fi
echo "[launch] Using ${NUM_GPUS} GPU(s): ${CUDA_VISIBLE_DEVICES}"

# ---- Hyperparams (overridable via --export)
RUN_NAME="${RUN_NAME:-gemma150M_3Btok}"
SEQ_LEN="${SEQ_LEN:-1024}"
MICRO_BSZ="${MICRO_BSZ:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-64}"
AMP="${AMP:-fp16}"
VAL_INT="${VAL_INT:-5000}"
CKPT_INT="${CKPT_INT:-5000}"
EVAL_TOK="${EVAL_TOK:-500000}"
EPOCH_ENG="${EPOCH_ENG:-1200000000}"
EPOCH_HIN="${EPOCH_HIN:-1100000000}"
EPOCH_NEP="${EPOCH_NEP:-700000000}"
LR_MAX="${LR_MAX:-2e-4}"
LR_MIN="${LR_MIN:-1e-5}"
WARMUP="${WARMUP:-1500}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
CKPT_DTYPE="${CKPT_DTYPE:-fp16}"
KEEP_LAST="${KEEP_LAST:-3}"
SAVE_OPT="${SAVE_OPT:-true}"

# ---- Launch under srun so cgroups apply; torchrun handles DDP
# Using --ntasks=1 (torchrun will spawn ${NUM_GPUS} workers on this node)
srun --ntasks=1 --gres=gpu:${NUM_GPUS} --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=none \
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" "$PY" \
    --spm_model_path "$SPM_LOCAL" \
    --splits_root   "$DEST_SPLITS" \
    --run_name      "$RUN_NAME" \
    --seq_len       "$SEQ_LEN" \
    --micro_bsz     "$MICRO_BSZ" \
    --grad_accum    "$GRAD_ACCUM" \
    --amp_dtype     "$AMP" \
    --val_interval  "$VAL_INT" \
    --ckpt_interval "$CKPT_INT" \
    --eval_tokens_per_lang "$EVAL_TOK" \
    --epoch_tokens_eng "$EPOCH_ENG" \
    --epoch_tokens_hin "$EPOCH_HIN" \
    --epoch_tokens_nep "$EPOCH_NEP" \
    --lr_max "$LR_MAX" \
    --lr_min "$LR_MIN" \
    --warmup_steps "$WARMUP" \
    --weight_decay "$WEIGHT_DECAY" \
    --grad_clip "$GRAD_CLIP" \
    --ckpt_dtype "$CKPT_DTYPE" \
    --keep_last_k "$KEEP_LAST" \
    --save_optimizer "$SAVE_OPT" \
    --save_to       "$SAVE_ROOT" \
    --packed_root "$DEST_PACKED" \
    --token_budget_ledger "$LEDGER"

echo "[done] Training finished."
