#!/usr/bin/env bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH -o /home2/%u/split_jsonl.%j.out
#SBATCH --nodelist=gnode004   # <<< IMPORTANT: data is on /scratch of gnode004

set -Eeuo pipefail

echo "=========================================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID:-}"
echo "SLURM_NODELIST = ${SLURM_NODELIST:-}"
echo "=========================================="
echo "[env] whoami=$(whoami)"
echo "[env] host=$(hostname)"

# Make 'module' available (just in case)
/etc/profile.d/modules.sh >/dev/null 2>&1 && source /etc/profile.d/modules.sh || true

# Try to use your conda env 'lma' if it exists; otherwise use cluster module python.
USE_CONDA=0
if [ -d "$HOME/miniconda3" ] || [ -d "$HOME/miniconda" ] || [ -d "/share1/$USER/miniconda3" ]; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate lma && USE_CONDA=1
  elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    conda activate lma && USE_CONDA=1
  elif [ -f "/share1/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/share1/$USER/miniconda3/etc/profile.d/conda.sh"
    conda activate lma && USE_CONDA=1
  fi
fi

if [ "$USE_CONDA" -eq 1 ]; then
  echo "[python] using conda env: $(python -V) ($(which python))"
else
  echo "[python] conda env not found; using cluster python module"
  module purge || true
  module load u18/python/3.10.2
  python3 -V || true
fi

IN_DIR="/scratch/$USER/LMA_SLM/data/raw"
OUT_DIR="/scratch/$USER/LMA_SLM/data/splits"
echo "[paths] IN_DIR=$IN_DIR"
echo "[paths] OUT_DIR=$OUT_DIR"

if [ ! -d "$IN_DIR" ]; then
  echo "[ERROR] $IN_DIR does not exist on $(hostname)."
  echo "        Your raw .jsonl files live on the /scratch of the node where you downloaded them."
  echo "        Either pin this job to that node with:   #SBATCH --nodelist=$(hostname)"
  echo "        or re-download/copy raw files onto this node's /scratch."
  exit 2
fi

shopt -s nullglob
files=("$IN_DIR"/*.jsonl)
shopt -u nullglob
if [ "${#files[@]}" -eq 0 ]; then
  echo "[ERROR] No .jsonl files found in $IN_DIR on $(hostname)."
  exit 2
fi

echo "[info] found raw files:"
ls -lh "$IN_DIR"/*.jsonl || true

mkdir -p "$OUT_DIR"

python3 /home2/$USER/LMA_SLM/scripts/hpc_make_splits_hash.py \
  --in_dir "$IN_DIR" \
  --out_dir "$OUT_DIR" \
  --langs eng hin nep \
  --train_ratio 0.98 \
  --val_ratio 0.01 \
  --seed 42

echo "[done] splits written under $OUT_DIR/<lang>/{train,val,test}.jsonl"
