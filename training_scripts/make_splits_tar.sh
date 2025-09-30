#!/usr/bin/env bash
set -euo pipefail
SRC="/share1/$USER/LMA_SLM/data/splits"    # existing splits dir
OUT="/share1/$USER/splits.tar.zst"         # tarball used by sbcast

if [[ ! -d "$SRC" ]]; then
  echo "[ERR] $SRC does not exist"; exit 2
fi
module load zstd 2>/dev/null || true
echo "[info] Packing $SRC -> $OUT (parallel zstd -19)…"
tar -I 'pzstd -19' -cf "$OUT" -C "$(dirname "$SRC")" "$(basename "$SRC")"
ls -lh "$OUT"
echo "[ok] Ready: $OUT"
