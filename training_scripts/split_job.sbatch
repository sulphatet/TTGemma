#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J split_jsonl
#SBATCH -o %x.%j.out

set -euo pipefail
RAW="/scratch/$USER/LMA_SLM/data/raw"
SPLITS="/scratch/$USER/LMA_SLM/data/splits"
python3 "$HOME/LMA_SLM/scripts/split_jsonl.py" --raw_dir "$RAW" --splits_dir "$SPLITS" --seed 42 --train 0.98 --val 0.01
