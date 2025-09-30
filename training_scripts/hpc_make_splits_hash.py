#!/usr/bin/env python3
"""
Streaming splitter: splits each <lang>.jsonl into train/val/test without
loading everything in memory. Deterministic by hashing doc_id (or text).
Writes to OUT_DIR/<lang>/{train,val,test}.jsonl
"""
import os, sys, json, argparse, hashlib

def safe_hash(s: str) -> int:
    h = hashlib.sha1(s.encode('utf-8')).hexdigest()
    return int(h[:8], 16)  # 32-bit bucket

def decide_bucket(doc, train_ratio, val_ratio):
    # stable key: prefer doc_id, fallback to text
    key = doc.get("doc_id") or doc.get("id") or doc.get("text","")
    if not key:
        key = json.dumps(doc, sort_keys=True)
    b = safe_hash(key) / 0xFFFFFFFF  # [0,1)
    if b < train_ratio:
        return "train"
    elif b < train_ratio + val_ratio:
        return "val"
    else:
        return "test"

def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--train_ratio", type=float, default=0.98)
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)  # unused (deterministic hash)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for lang in args.langs:
        in_path = os.path.join(args.in_dir, f"{lang}.jsonl")
        if not os.path.exists(in_path):
            print(f"[ERROR] missing {in_path}", file=sys.stderr)
            sys.exit(2)

        lang_out = os.path.join(args.out_dir, lang)
        os.makedirs(lang_out, exist_ok=True)
        f_train = open(os.path.join(lang_out, "train.jsonl"), "w", encoding="utf-8")
        f_val   = open(os.path.join(lang_out, "val.jsonl"),   "w", encoding="utf-8")
        f_test  = open(os.path.join(lang_out, "test.jsonl"),  "w", encoding="utf-8")

        c_train = c_val = c_test = 0
        print(f"[split:{lang}] reading {in_path}")
        for rec in stream_jsonl(in_path):
            bucket = decide_bucket(rec, args.train_ratio, args.val_ratio)
            if bucket == "train":
                f_train.write(json.dumps(rec, ensure_ascii=False)+"\n"); c_train += 1
            elif bucket == "val":
                f_val.write(json.dumps(rec, ensure_ascii=False)+"\n"); c_val += 1
            else:
                f_test.write(json.dumps(rec, ensure_ascii=False)+"\n"); c_test += 1

        for fh in (f_train, f_val, f_test):
            fh.flush(); fh.close()

        total = c_train + c_val + c_test
        print(f"[done:{lang}] total={total:,} train={c_train:,} val={c_val:,} test={c_test:,} -> {lang_out}")

    print("[done] all languages finished.")
if __name__ == "__main__":
    main()
