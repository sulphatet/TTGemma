#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH -o /home2/%u/split_jsonl.%j.out

set -euo pipefail

IN_DIR="/share1/$USER/LMA_SLM/data/raw"
OUT_DIR="/share1/$USER/LMA_SLM/data/splits"

echo "[env] whoami=$(whoami)"
echo "[env] host=$(hostname)"
echo "[env] IN_DIR=$IN_DIR"
echo "[env] OUT_DIR=$OUT_DIR"

mkdir -p "$OUT_DIR"

python3 /home2/$USER/LMA_SLM/scripts/hpc_make_splits_hash.py \
  --in_dir "$IN_DIR" \
  --out_dir "$OUT_DIR" \
  --langs eng hin nep \
  --train_ratio 0.98 \
  --val_ratio 0.01 \
  --seed 42

echo "[done] splits written under $OUT_DIR/<lang>/{train,val,test}.jsonl"
