#!/usr/bin/env bash
#SBATCH -J shard_pack
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /home2/%u/logs/%x.%j.out
#SBATCH -e /home2/%u/logs/%x.%j.err

set -euo pipefail
IFS=$'\n\t'

echo "=========================================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-NA}"
echo "SLURM_NODELIST = ${SLURM_NODELIST:-NA}"
echo "=========================================="

# ---- Paths
USER_DIR="/home2/$USER/LMA_SLM"
PY_PACK="$USER_DIR/scripts/pack_lma_shards.py"   # your packer
SRC_ROOT="ada:/share1/$USER/LMA_SLM"             # login host view
SRC_SPLITS="$SRC_ROOT/data/splits/"
DST_ROOT_LOCAL_BASE="/scratch/$USER"
[[ -d "$DST_ROOT_LOCAL_BASE" ]] || DST_ROOT_LOCAL_BASE="/tmp/$USER"
WORKDIR="$DST_ROOT_LOCAL_BASE/LMA_SLM/${SLURM_JOB_ID:-manual}"

TOK="$USER_DIR/tokenizers/sp_unigram_64000.model"

# Where to write updated shards on ADA:/share1 (push-back target)
PUSH_BACK_DIR="$SRC_ROOT/data/splits_packed"

mkdir -p "$WORKDIR" /home2/$USER/logs || true

echo "[paths] WORKDIR      = $WORKDIR"
echo "[paths] SRC_SPLITS   = $SRC_SPLITS"
echo "[paths] PUSH_BACK    = $PUSH_BACK_DIR"

# ---- Environment (no system CUDA; just rsync + conda env)
module purge
module load rsync 2>/dev/null || true
source ~/.bashrc
conda activate lma

# ---- Stage-in from ada:/share1 -> node scratch
DEST_SPLITS="$WORKDIR/data/splits"
mkdir -p "$DEST_SPLITS"
echo "[rsync] PULL  ada:/share1 -> node local"
time rsync -avh --delete --partial --inplace --info=stats2,progress2 \
  "$SRC_SPLITS" "$DEST_SPLITS/"

du -sh "$DEST_SPLITS"/* 2>/dev/null || true

# ---- Run the packer (CPU-only)
echo "[python] packing shards ..."
# Only pass --max_docs if user actually set it
MAX_DOCS_ARG=()
if [[ -n "${MAX_DOCS:-}" ]]; then
  MAX_DOCS_ARG=(--max_docs "$MAX_DOCS")
fi

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=none \
  python -u "$PY_PACK" \
    --spm_model_path "$TOK" \
    --splits_root    "$DEST_SPLITS" \
    --out_root       "$WORKDIR/data/splits_packed" \
    --seq_len        "${SEQ_LEN:-1024}" \
    --rows_per_shard "${ROWS_PER_SHARD:-200000}" \
    ${MAX_DOCS_ARG:+${MAX_DOCS_ARG[@]}}

echo "[python] pack complete."


echo "[python] pack complete."
du -sh "$WORKDIR/data/splits_packed"/* 2>/dev/null || true

# ---- Push back to ada:/share1
echo "[rsync] PUSH node local -> ada:/share1 (splits_packed)"
time rsync -avh --delete --partial --inplace --info=stats2,progress2 \
  "$WORKDIR/data/splits_packed/" "$PUSH_BACK_DIR/"

echo "[done] All good."
