#!/usr/bin/env python3
import os, sys, json, hashlib, time
from typing import Dict
from dataclasses import dataclass, asdict
from datasets import load_dataset
import sentencepiece as spm

def _user():
    u = os.environ.get("USER") or os.popen("whoami").read().strip()
    return u or "user"

@dataclass
class CFG:
    user: str = _user()
    # persistent things in HOME
    base_home: str = os.path.expanduser(f"~/LMA_SLM")
    # large data on per-node scratch
    base_scratch: str = f"/scratch/{_user()}/LMA_SLM"
    spm_path: str = None
    langs: tuple = ("eng", "hin", "nep")
    source_subset: str = "verified"
    token_targets: Dict[str,int] = None
    print_every: int = 50_000

    def __post_init__(self):
        if self.spm_path is None:
            self.spm_path = os.path.join(self.base_home, "tokenizers", "sp_unigram_64000.model")
        if self.token_targets is None:
            # ≈ 3B total (tweak if you want)
            self.token_targets = {"eng": 1_200_000_000, "hin": 1_100_000_000, "nep": 700_000_000}

    # outputs (big) on scratch
    @property
    def raw_dir(self): return os.path.join(self.base_scratch, "data", "raw")
    @property
    def reports_dir(self): return os.path.join(self.base_scratch, "reports")
    # small state in HOME (persistent)
    @property
    def state_dir(self): return os.path.join(self.base_home, "state")

def ensure_dirs(cfg: CFG):
    for d in (cfg.raw_dir, cfg.reports_dir, cfg.state_dir):
        os.makedirs(d, exist_ok=True)

def load_spm(path: str) -> spm.SentencePieceProcessor:
    if not os.path.exists(path):
        print(f"[warn] SPM not found at {path}. Proceeding without token counts.", file=sys.stderr)
        return None
    sp = spm.SentencePieceProcessor(); sp.Load(path); return sp

def state_path(cfg: CFG, lang: str): return os.path.join(cfg.state_dir, f"download_{lang}.json")
def read_state(cfg: CFG, lang: str) -> Dict:
    p = state_path(cfg, lang)
    if os.path.exists(p):
        try: return json.load(open(p, "r"))
        except Exception: pass
    return {"lang": lang, "docs": 0, "tokens": 0, "finished": False, "started_at": int(time.time())}
def write_state(cfg: CFG, lang: str, st: Dict):
    tmp = state_path(cfg, lang) + ".tmp"
    with open(tmp, "w") as f: json.dump(st, f, indent=2)
    os.replace(tmp, state_path(cfg, lang))

def count_tokens(sp, text: str) -> int:
    if sp is None:  # fallback: rough char-based proxy if spm missing
        return max(1, len(text)//4)
    return len(sp.EncodeAsIds(text))

def stream_lang(cfg: CFG, lang: str, sp):
    out_path = os.path.join(cfg.raw_dir, f"{lang}.jsonl")
    target = int(cfg.token_targets.get(lang, 0))
    if target <= 0:
        print(f"[{lang}] target 0, skipping."); return
    st = read_state(cfg, lang)
    if st.get("finished"):
        print(f"[{lang}] already finished (tokens={st['tokens']:,}), skipping."); return
    f_out = open(out_path, "a", encoding="utf-8")
    print(f"[{lang}] -> {out_path} (append). Target tokens: {target:,}")

    ds_name = "ai4bharat/sangraha"
    data_dir = f"{cfg.source_subset}/{lang}"
    try:
        ds = load_dataset(ds_name, data_dir=data_dir, streaming=True, split="train")
    except Exception as e:
        print(f"[{lang}] ERROR opening dataset {ds_name} ({data_dir}): {e}", file=sys.stderr)
        f_out.close(); return

    t0 = time.time()
    last_print_docs = st["docs"]
    for ex in ds:
        text = ex.get("text", "") if isinstance(ex, dict) else ""
        if not text: continue
        ntok = count_tokens(sp, text)
        doc_id = ex.get("doc_id") or hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        f_out.write(json.dumps({"doc_id": str(doc_id), "subset": cfg.source_subset, "lang": lang, "text": text}, ensure_ascii=False) + "\n")

        st["docs"] += 1; st["tokens"] += ntok
        if (st["docs"] - last_print_docs) >= cfg.print_every:
            last_print_docs = st["docs"]
            dt = max(1e-6, time.time()-t0)
            print(f"[{lang}] docs={st['docs']:,} tokens={st['tokens']:,} ({int(st['tokens']/dt):,} tok/s)")

        if st["tokens"] >= target:
            st["finished"] = True; write_state(cfg, lang, st)
            f_out.flush(); f_out.close()
            print(f"[{lang}] ✅ reached {st['tokens']:,} tokens (docs={st['docs']:,})")
            return

        if (st["docs"] % 10000) == 0:
            write_state(cfg, lang, st)

    write_state(cfg, lang, st); f_out.flush(); f_out.close()
    print(f"[{lang}] ⚠️ dataset stream ended early. tokens={st['tokens']:,} docs={st['docs']:,}")

def main():
    cfg = CFG(); ensure_dirs(cfg)
    print("[cfg]", json.dumps(asdict(cfg), indent=2))
    sp = load_spm(cfg.spm_path)
    if sp: print("[spm] vocab_size:", sp.GetPieceSize())
    tot_target = sum(cfg.token_targets.get(l,0) for l in cfg.langs)
    print(f"[plan] target total ≈ {tot_target:,} tokens")
    for lang in cfg.langs: stream_lang(cfg, lang, sp)
    tot = 0
    for lang in cfg.langs:
        st = read_state(cfg, lang)
        print(f"[final:{lang}] tokens={st['tokens']:,} docs={st['docs']:,} finished={st['finished']}")
        tot += st["tokens"]
    print(f"[final] total_tokens={tot:,}")

if __name__ == "__main__":
    main()
