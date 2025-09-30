#!/usr/bin/env python3
import os, sys, json, random, argparse
from pathlib import Path

def split_file(src, out_dir, seed=42, train=0.98, val=0.01):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    writers = {
        "train": open(Path(out_dir)/"train.jsonl", "w", encoding="utf-8"),
        "val":   open(Path(out_dir)/"val.jsonl",   "w", encoding="utf-8"),
        "test":  open(Path(out_dir)/"test.jsonl",  "w", encoding="utf-8"),
    }
    rng = random.Random(seed)
    counts = {k:0 for k in writers}
    with open(src, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            r = rng.random()
            bucket = "train" if r < train else ("val" if r < train+val else "test")
            writers[bucket].write(json.dumps(rec, ensure_ascii=False)+"\n")
            counts[bucket] += 1
            if i % 200000 == 0:
                print(f"[{os.path.basename(src)}] processed={i:,} train={counts['train']:,} val={counts['val']:,} test={counts['test']:,}", flush=True)
    for w in writers.values(): w.close()
    print(f"[done:{os.path.basename(src)}] train={counts['train']:,} val={counts['val']:,} test={counts['test']:,}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.98)
    ap.add_argument("--val", type=float, default=0.01)
    args = ap.parse_args()

    langs = ["eng","hin","nep"]
    for lang in langs:
        src = os.path.join(args.raw_dir, f"{lang}.jsonl")
        out_dir = os.path.join(args.splits_dir, lang)
        print(f"[split] {lang} -> {out_dir}")
        split_file(src, out_dir, seed=args.seed, train=args.train, val=args.val)

if __name__ == "__main__":
    main()
