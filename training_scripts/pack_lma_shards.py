# -*- coding: utf-8 -*-
"""
pack_lma_shards.py — Pretokenize + pack JSONL -> memmap shards for fast training

- Reads JSONL at: {splits_root}/{eng,hin,nep}/{train,val,test}.jsonl ({"text": "..."} per line)
- Uses SentencePiece tokenizer (with <eng>/<hin>/<nep> tags) to encode
- Packs contiguous tokens into fixed-length sequences of (seq_len+1) for next-token prediction
- Writes .npy int32 shards: {out_root}/{lang}/{split}/shard_{00000..}.npy with shape [rows_per_shard, seq_len+1]
- Also writes a small metadata.json next to each split

Tested with Python 3.10, numpy, sentencepiece. No torch runtime required.
"""

from __future__ import annotations
import os, io, json, argparse, math
from typing import Iterator, List, Optional
import numpy as np

try:
    import sentencepiece as spm
except Exception as e:
    raise RuntimeError("Install sentencepiece: pip install sentencepiece") from e

def jsonl_iter_texts(path: str) -> Iterator[str]:
    import gzip, lzma
    if not os.path.exists(path):
        return
    opener = open
    p = path.lower()
    if p.endswith(".gz"): opener = gzip.open
    elif p.endswith(".xz"): opener = lzma.open
    with opener(path, "rb") as fb:
        with io.TextIOWrapper(fb, encoding="utf-8", errors="replace", newline="") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    t = obj.get("text", "")
                    if t: yield t
                except Exception:
                    continue

class SPMTokenizer:
    def __init__(self, spm_model_path: str):
        assert os.path.exists(spm_model_path), f"SPM not found: {spm_model_path}"
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_model_path)
        self.vocab_size = int(self.sp.GetPieceSize())
        def pid(piece: str) -> Optional[int]:
            try:
                i = self.sp.PieceToId(piece)
                return int(i) if i >= 0 else None
            except Exception:
                return None
        self.id_eos = pid("</s>")
        self.lang_tags = {"eng": pid("<eng>"), "hin": pid("<hin>"), "nep": pid("<nep>")}

    def encode(self, text: str, lang: Optional[str]=None, add_eos=True) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        if lang and self.lang_tags.get(lang) is not None:
            ids = [self.lang_tags[lang]] + ids
        if add_eos and self.id_eos is not None:
            ids = ids + [self.id_eos]
        return ids

from numpy.lib.format import open_memmap  # add this import at top

def pack_split_for_lang(
    splits_root: str, out_root: str, lang: str, split: str, tok: SPMTokenizer,
    seq_len: int, rows_per_shard: int, max_docs: Optional[int]=None
) -> dict:
    in_path = os.path.join(splits_root, lang, f"{split}.jsonl")
    assert os.path.exists(in_path), f"missing: {in_path}"

    out_dir = os.path.join(out_root, lang, split)
    os.makedirs(out_dir, exist_ok=True)

    seq_total = 0
    shard_idx = 0
    row_idx = 0
    shard = None  # numpy.memmap created via open_memmap (true .npy)
    shard_path = None

    buffer: List[int] = []
    docs = 0
    printed = 0

    def _alloc_new_shard():
        nonlocal shard, shard_path, shard_idx, row_idx
        # close previous shard cleanly
        if shard is not None:
            try:
                shard.flush()
            except Exception:
                pass
            # drop handle to ensure file is closed on some filesystems
            del shard
            shard = None
        shard_path = os.path.join(out_dir, f"shard_{shard_idx:05d}.npy")
        # Create a *real* .npy memmap with header
        shard = open_memmap(
            shard_path, mode="w+", dtype=np.int32, shape=(rows_per_shard, seq_len + 1)
        )
        row_idx = 0
        shard_idx += 1

    _alloc_new_shard()

    for text in jsonl_iter_texts(in_path):
        ids = tok.encode(text, lang=lang, add_eos=True)
        if not ids:
            continue
        buffer.extend(ids)
        docs += 1
        printed += 1
        if printed % 10000 == 0:
            print(f"[{lang}/{split}] {printed} docs ... total seq={seq_total:,}")

        while len(buffer) >= (seq_len + 1):
            seq = buffer[: seq_len + 1]
            del buffer[: seq_len + 1]
            shard[row_idx, :] = np.asarray(seq, dtype=np.int32)
            row_idx += 1
            seq_total += 1
            if row_idx >= rows_per_shard:
                _alloc_new_shard()

        if max_docs is not None and docs >= max_docs:
            break

    # Drain any remaining buffer
    while len(buffer) >= (seq_len + 1):
        seq = buffer[: seq_len + 1]
        del buffer[: seq_len + 1]
        shard[row_idx, :] = np.asarray(seq, dtype=np.int32)
        row_idx += 1
        seq_total += 1
        if row_idx >= rows_per_shard:
            _alloc_new_shard()

    # Finalize last shard
    # last_rows = row_idx
    # try:
    #     shard.flush()
    # except Exception:
    #     pass
    # # release memmap handle
    # del shard
    # shard = None

    # # If the last shard is *not* full, rewrite it compactly and atomically replace
    # if last_rows < rows_per_shard:
    #     # Load only valid rows (avoid mmap handle)
    #     arr = np.load(shard_path, mmap_mode="r")[:last_rows].astype(np.int32, copy=True)
    #     tmp_path = shard_path + ".compact_tmp"
    #     np.save(tmp_path, arr)
    #     # Replace the oversize shard by the compact one
    #     os.replace(tmp_path, shard_path)
        # Finalize last shard
    last_rows = row_idx
    try:
        shard.flush()
    except Exception:
        pass
    # release memmap handle (important on some FS)
    del shard
    shard = None

    # If last shard had 0 rows, remove it entirely
    if last_rows == 0:
        try:
            os.remove(shard_path)
        except FileNotFoundError:
            pass
    else:
        # If the last shard is not full, rewrite it compactly and atomically replace
        if last_rows < rows_per_shard:
            # Load only valid rows; make a compact array
            arr = np.load(shard_path, mmap_mode="r")[:last_rows].astype(np.int32, copy=True)
            # IMPORTANT: ensure tmp path ends with .npy so np.save does not change it
            tmp_path = shard_path + ".compact_tmp.npy"
            np.save(tmp_path, arr)
            # Replace the oversized shard by the compact one
            os.replace(tmp_path, shard_path)

    # metadata
    meta = {
        "lang": lang,
        "split": split,
        "seq_len_plus1": seq_len + 1,
        "rows_per_shard": rows_per_shard,
        "num_shards": shard_idx,
        "total_sequences": seq_total,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done {lang}/{split}] seq={seq_total:,} shards={shard_idx}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm_model_path", required=True)
    ap.add_argument("--splits_root", required=True)
    ap.add_argument("--out_root", required=True, help="Where to write packed shards")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--rows_per_shard", type=int, default=65536, help="rows per .npy (each row has seq_len+1)")
    ap.add_argument("--max_docs", type=int, default=None, help="for smoke tests")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    tok = SPMTokenizer(args.spm_model_path)

    summary = {}
    for lang in ("eng","hin","nep"):
        for split in ("train","val","test"):
            in_path = os.path.join(args.splits_root, lang, f"{split}.jsonl")
            if not os.path.exists(in_path):
                print(f"[skip] {lang}/{split}: {in_path} missing.")
                continue
            meta = pack_split_for_lang(
                splits_root=args.splits_root,
                out_root=args.out_root,
                lang=lang, split=split, tok=tok,
                seq_len=args.seq_len, rows_per_shard=args.rows_per_shard,
                max_docs=args.max_docs
            )
            summary[f"{lang}/{split}"] = meta

    with open(os.path.join(args.out_root, "PACKING_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[ALL DONE]")

if __name__ == "__main__":
    main()
