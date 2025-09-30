# -*- coding: utf-8 -*-
"""
Clickbait Detection (EN/HI) — LoRA fine-tuning on your custom Gemma-style LM

- Reads CSV/TSV robustly (auto delimiter, skips malformed lines)
- Prompts:
    EN: Clickbait Detection: Is the headline "<headline>" likely clickbait? Answer:
    HI: क्लिकबेट डिटेक्शन: क्या हेडलाइन "<headline>" क्लिकबेट होने की संभावना है? उत्तर:
  Target text: "Yes"/"No" (English) or "हाँ"/"नहीं" (Hindi)

- Saves adapter to:
  /content/drive/MyDrive/LMA_SLM/finetune/clickbait/<lang>/<run_name>/lora_adapter

- Avoids adapter overrides by using a run_name subfolder you can set.
"""

import os, re, csv, argparse, time
from pathlib import Path
import torch

# ---------- Colab-friendly deps ----------
try:
    import transformers, datasets, peft, sentencepiece, bitsandbytes, pandas  # noqa
    _has_bnb_import = True
except Exception:
    os.system("pip install -q peft datasets transformers sentencepiece pandas bitsandbytes")
    try:
        import transformers, datasets, peft, sentencepiece, bitsandbytes, pandas  # noqa
        _has_bnb_import = True
    except Exception:
        import transformers, datasets, peft, sentencepiece, pandas  # noqa
        _has_bnb_import = False

import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import TrainingArguments, Trainer

# -----------------------------
#   Custom Model + Tokenizer
# -----------------------------

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__(); self.weight = torch.nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x): return self.weight * (x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps))

class SwiGLU(torch.nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__(); self.w1 = torch.nn.Linear(dim, hidden, bias=False); self.w2 = torch.nn.Linear(dim, hidden, bias=False)
        self.proj = torch.nn.Linear(hidden, dim, bias=False); self.drop = torch.nn.Dropout(drop)
    def forward(self, x): return self.drop(self.proj(self.w1(x) * torch.nn.functional.silu(self.w2(x))))

def _rope_freqs(hd, base, max_pos, device, dtype):
    half=hd//2; inv=1.0/(base**(torch.arange(0,half,device=device,dtype=dtype)/half)); pos=torch.arange(0,max_pos,device=device,dtype=dtype)
    f=torch.outer(pos,inv); return torch.cos(f), torch.sin(f)

def _apply_rope(q,k,cos,sin):
    Dh=q.shape[-1]; half=Dh//2
    def rot(x):
        x1,x2=x[...,:half],x[...,half:]; T=x.size(2)
        return torch.cat([x1*cos[:T]-x2*sin[:T], x2*cos[:T]+x1*sin[:T]],dim=-1)
    return rot(q),rot(k)

class DecoderBlock(torch.nn.Module):
    def __init__(self,d_model,n_heads,d_ff,drop=0.0,rope_base=10000.0,max_position=4096):
        super().__init__(); assert d_model%n_heads==0
        self.n_heads=n_heads; self.d_head=d_model//n_heads
        self.norm1=RMSNorm(d_model); self.qkv=torch.nn.Linear(d_model,3*d_model,bias=False); self.proj=torch.nn.Linear(d_model,d_model,bias=False)
        self.norm2=RMSNorm(d_model); self.mlp=SwiGLU(d_model,d_ff,drop); self.drop=torch.nn.Dropout(drop)
        self.rope_base=rope_base; self.max_position=max_position
        self.register_buffer("_cos",None,persistent=False); self.register_buffer("_sin",None,persistent=False)
    def _ensure_rope(self,device,dtype):
        if self._cos is None or self._cos.device!=device or self._cos.dtype!=dtype:
            self._cos,self._sin=_rope_freqs(self.d_head,self.rope_base,self.max_position,device,dtype)
    def forward(self,x):
        B,T,C=x.shape; h=self.norm1(x); qkv=self.qkv(h); q,k,v=qkv.chunk(3,dim=-1)
        q=q.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        k=k.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        v=v.view(B,T,self.n_heads,self.d_head).transpose(1,2)
        self._ensure_rope(q.device,q.dtype); q,k=_apply_rope(q,k,self._cos,self._sin)
        attn=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=True)
        attn=attn.transpose(1,2).contiguous().view(B,T,C)
        x=x+self.drop(self.proj(attn)); h=self.norm2(x); x=x+self.drop(self.mlp(h)); return x

class SimpleConfig:
    def __init__(self, vocab_size:int, eos_token_id:int|None=None, pad_token_id:int|None=None,
                 is_encoder_decoder:bool=False, tie_word_embeddings:bool=True,
                 hidden_size:int|None=None, num_hidden_layers:int|None=None,
                 num_attention_heads:int|None=None, max_position_embeddings:int|None=None,
                 model_type:str="gemma", **kwargs):
        self.vocab_size=int(vocab_size); self.eos_token_id=eos_token_id; self.pad_token_id=pad_token_id
        self.is_encoder_decoder=is_encoder_decoder; self.tie_word_embeddings=tie_word_embeddings
        self.hidden_size=hidden_size; self.num_hidden_layers=num_hidden_layers; self.num_attention_heads=num_attention_heads
        self.max_position_embeddings=max_position_embeddings; self.model_type=model_type
        for k,v in kwargs.items(): setattr(self,k,v)
    def get(self,k,default=None): return getattr(self,k,default)
    def __getitem__(self,k): return getattr(self,k)
    def to_dict(self): return self.__dict__.copy()

def _checkpoint_block(module, x):
    def fn(inp): return module(inp)
    try: return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
    except TypeError: return torch.utils.checkpoint.checkpoint(fn, x)

class GemmaLM(torch.nn.Module):
    def __init__(self,V,d_model,n_layers,n_heads,d_ff,max_position,drop=0.0,tie_weights=True):
        super().__init__()
        self.tok_emb=torch.nn.Embedding(V,d_model); self.drop=torch.nn.Dropout(drop)
        self.blocks=torch.nn.ModuleList([DecoderBlock(d_model,n_heads,d_ff,drop,10000.0,max_position) for _ in range(n_layers)])
        self.norm_f=RMSNorm(d_model); self.lm_head=torch.nn.Linear(d_model,V,bias=False)
        if tie_weights: self.lm_head.weight=self.tok_emb.weight
        self.config=SimpleConfig(vocab_size=V,eos_token_id=None,pad_token_id=None,is_encoder_decoder=False,
                                 tie_word_embeddings=bool(tie_weights),hidden_size=d_model,
                                 num_hidden_layers=n_layers,num_attention_heads=n_heads,
                                 max_position_embeddings=max_position,model_type="gemma")
        self._gc_enabled=False
    def gradient_checkpointing_enable(self, **kwargs): self._gc_enabled=True
    def gradient_checkpointing_disable(self): self._gc_enabled=False
    def forward(self,input_ids,labels=None,attention_mask=None,**kwargs):
        x=self.drop(self.tok_emb(input_ids))
        for b in self.blocks:
            x=_checkpoint_block(b,x) if (self.training and self._gc_enabled) else b(x)
        x=self.norm_f(x); logits=self.lm_head(x)
        if labels is not None:
            shift_logits=logits[:, :-1, :].contiguous(); shift_labels=labels[:, 1:].contiguous()
            loss=torch.nn.functional.cross_entropy(shift_logits.reshape(-1,shift_logits.size(-1)),
                                                   shift_labels.reshape(-1),ignore_index=-100)
            return {"loss":loss,"logits":logits}
        return {"logits":logits}
    def prepare_inputs_for_generation(self,input_ids,**kwargs): return {"input_ids":input_ids}

class SPTokenizer:
    def __init__(self, model_path: Path):
        from sentencepiece import SentencePieceProcessor
        self.sp=SentencePieceProcessor(); self.sp.load(str(model_path))
        try: pid=self.sp.pad_id()
        except Exception: pid=-1
        self.pad_token_id = pid if (isinstance(pid,int) and pid>=0) else 0
        try: self.eos_token_id=int(self.sp.eos_id())
        except Exception: self.eos_token_id=None
        try: self.vocab_size=int(self.sp.get_piece_size())
        except Exception:
            try: self.vocab_size=int(self.sp.GetPieceSize())
            except Exception: self.vocab_size=None
    def __call__(self,text):
        ids=self.sp.encode_as_ids(text); return {"input_ids":[ids]}
    def batch_decode(self, sequences, skip_special_tokens=True):
        decoded=[]
        try: unk_id=self.sp.unk_id(); bos_id=self.sp.bos_id(); eos_id=self.sp.eos_id(); pad_id=self.sp.pad_id()
        except Exception: unk_id=bos_id=eos_id=pad_id=None
        for seq in sequences:
            seq2=seq
            if skip_special_tokens and None not in (unk_id,bos_id,eos_id,pad_id):
                seq2=[t for t in seq if t not in [unk_id,bos_id,eos_id,pad_id]]
            decoded.append(self.sp.decode_ids(seq2))
        return decoded

# --- Helpers for base checkpoint ---
def _cast_state_dict_to_dtype(sd,dtype):
    out={}
    for k,v in sd.items():
        out[k]=v.to(dtype=dtype) if (isinstance(v,torch.Tensor) and v.is_floating_point()) else v
    return out

PRESETS={"150M_plan":dict(d_model=640,n_layers=16,n_heads=10,d_ff=2688,tie_weights=True)}

def _infer_arch_from_state_dict(sd):
    if "tok_emb.weight" not in sd: raise RuntimeError("tok_emb.weight missing")
    V,d_model=sd["tok_emb.weight"].shape
    max_idx=-1
    for k in sd.keys():
        m=re.search(r"^blocks\.(\d+)\.",k)
        if m: max_idx=max(max_idx,int(m.group(1)))
    n_layers=max_idx+1 if max_idx>=0 else 0
    key="blocks.0.mlp.proj.weight"
    if key not in sd: raise RuntimeError("blocks.0.mlp.proj.weight missing")
    _,hidden=sd[key].shape; d_ff=hidden; n_heads=10
    return dict(V=V,d_model=d_model,n_layers=n_layers,n_heads=n_heads,d_ff=d_ff)

# --- Simple collator (pads + -100 labels) ---
class SimpleCausalCollator:
    def __init__(self,pad_id:int=0): self.pad_id=int(pad_id)
    def __call__(self,features):
        seqs=[torch.tensor(f["input_ids"],dtype=torch.long) for f in features]
        maxlen=max(s.size(0) for s in seqs); bsz=len(seqs)
        batch=torch.full((bsz,maxlen),self.pad_id,dtype=torch.long)
        for i,s in enumerate(seqs): batch[i,:s.size(0)]=s
        attention_mask=(batch!=self.pad_id).long()
        labels=batch.clone(); labels[labels==self.pad_id]=-100
        return {"input_ids":batch,"labels":labels,"attention_mask":attention_mask}

# -----------------------------
#   CSV/TSV reading (robust)
# -----------------------------
def _guess_delimiter(path: Path, fallback: str | None = None) -> str:
    try:
        sample = path.open("rb").read(4096).decode("utf-8", errors="ignore")
        tabs = sample.count("\t"); commas = sample.count(",")
        if tabs > commas: return "\t"
        if commas > 0:    return ","
    except Exception: pass
    return fallback or ","

def _load_clickbait_csv(path: Path, delimiter: str | None = None) -> pd.DataFrame:
    sep = delimiter or _guess_delimiter(path)
    df = pd.read_csv(
        path, sep=sep, engine="python", dtype=str,
        quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar="\\",
        on_bad_lines="skip", keep_default_na=False
    )
    # Normalize columns to lower
    df.columns = [c.strip().lower() for c in df.columns]
    # Map plausible column names → standardized
    rename_map = {
        "headline":"headline", "title":"headline", "text":"headline",
        "clickbait":"clickbait", "label":"clickbait", "is_clickbait":"clickbait", "target":"clickbait"
    }
    cols={}
    for want in ("headline","clickbait"):
        if want in df.columns: cols[want]=want
        else:
            for k,v in rename_map.items():
                if v==want and k in df.columns: cols[want]=k; break
    missing=[c for c in ("headline","clickbait") if c not in cols]
    if missing: raise ValueError(f"Missing required column(s): {missing}. Found={list(df.columns)}")
    df = df[[cols["headline"], cols["clickbait"]]].rename(columns={cols["headline"]:"headline", cols["clickbait"]:"clickbait"})
    # Clean & coerce labels to {0,1}
    df["headline"] = df["headline"].astype(str).map(lambda s: re.sub(r"\s+"," ", s.strip()))
    y = df["clickbait"].astype(str).str.strip().str.lower()
    y = y.map({"1":1,"0":0,"yes":1,"no":0,"true":1,"false":0})
    df = df.assign(clickbait=y)
    df = df[(df["headline"]!="") & (df["clickbait"].isin([0,1]))]
    return df

# -----------------------------
#            MAIN
# -----------------------------
def main(args):
    # --------- Output dir: avoid overriding existing adapters ----------
    base_out = Path("/content/drive/MyDrive/LMA_SLM/finetune") / "clickbait" / args.lang / args.run_name
    base_out.mkdir(parents=True, exist_ok=True)

    # --------- Tokenizer ----------
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tok = SPTokenizer(Path(args.tokenizer_path))
    if tok.vocab_size is None: print("❌ ERROR: Could not determine tokenizer vocab size."); return
    V_from_tok = tok.vocab_size

    # --------- Base checkpoint ----------
    print(f"Loading pre-trained model checkpoint from: {args.checkpoint_path}")
    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists(): print("❌ ERROR: Checkpoint not found."); return
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {}); sd = ckpt["model"]

    # Arch
    if "model_preset" in cfg and cfg["model_preset"] in PRESETS:
        mp = PRESETS[cfg["model_preset"]]
        d_model, n_layers, n_heads, d_ff = mp["d_model"], mp["n_layers"], mp["n_heads"], mp["d_ff"]
        tie_weights = mp.get("tie_weights", True)
        seq_len = int(cfg.get("seq_len", 4096)); V = V_from_tok
    else:
        print("[warn] model_preset missing — inferring arch from state_dict.")
        arch = _infer_arch_from_state_dict(sd)
        d_model, n_layers, n_heads, d_ff = arch["d_model"], arch["n_layers"], arch["n_heads"], arch["d_ff"]
        tie_weights = True; seq_len = int(cfg.get("seq_len", 4096)); V = V_from_tok

    print("Instantiating model architecture...")
    base_model = GemmaLM(V, d_model, n_layers, n_heads, d_ff, max_position=seq_len, tie_weights=tie_weights)
    base_model.config.eos_token_id = tok.eos_token_id; base_model.config.pad_token_id = tok.pad_token_id
    base_model.load_state_dict(_cast_state_dict_to_dtype(sd, next(base_model.parameters()).dtype), strict=False)
    print("✅ Model loaded.")

    # --------- Load dataset ----------
    data_path = Path(args.dataset_path)
    if not data_path.exists(): print("❌ ERROR: Dataset file not found."); return
    try:
        df = _load_clickbait_csv(data_path, delimiter=args.delimiter)
    except Exception as e:
        print(f"❌ ERROR while reading dataset: {e}"); return
    print(f"Using {len(df):,} rows from {data_path.name}")
    ds = Dataset.from_pandas(df, preserve_index=False)

    # --------- Prompt formatting ----------
    def format_row(ex):
        headline = str(ex["headline"]).strip()
        y = int(ex["clickbait"])
        if args.lang == "hi":
            prompt = f'क्लिकबेट डिटेक्शन: क्या हेडलाइन "{headline}" क्लिकबेट होने की संभावना है? उत्तर: '
            label_text = "हाँ" if y == 1 else "नहीं"
        else:
            prompt = f'Clickbait Detection: Is the headline "{headline}" likely clickbait? Answer: '
            label_text = "Yes" if y == 1 else "No"
        full_text = prompt + label_text
        ids = tok(full_text)["input_ids"][0]
        if tok.eos_token_id is not None: ids = ids + [tok.eos_token_id]
        return {"input_ids": ids}

    ds = ds.map(format_row, remove_columns=ds.column_names)
    ds = ds.filter(lambda ex: len(ex["input_ids"]) > 0)
    print("✅ Dataset processed.")

    # --------- LoRA (attn + MLP) ----------
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.08,
        target_modules=["qkv","proj","w1","w2"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    try: model.print_trainable_parameters()
    except Exception: pass

    # --------- Optimizer choice ----------
    def _can_use_bnb8bit():
        if not _has_bnb_import or not torch.cuda.is_available(): return False
        try:
            import bitsandbytes as bnb  # noqa
            return hasattr(bnb.optim, "Adam8bit")
        except Exception:
            return False
    chosen_optim = "adamw_bnb_8bit" if _can_use_bnb8bit() else "adamw_torch"
    if chosen_optim != "adamw_bnb_8bit":
        print("[info] Using adamw_torch (bitsandbytes 8-bit optimizer unavailable).")

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(base_out),
        auto_find_batch_size=True,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=20,
        save_steps=200,
        report_to="none",
        fp16=bool(use_cuda),
        gradient_checkpointing=True,
        optim=chosen_optim,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        dataloader_pin_memory=bool(use_cuda),
        save_total_limit=5,
    )

    collator = SimpleCausalCollator(pad_id=tok.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    print(f"\n--- 🚀 Starting Fine-Tuning for {args.lang.upper()} Clickbait ---")
    trainer.train()
    print("--- ✅ Fine-Tuning Complete ---")

    # --------- Save LoRA adapter (task-specific path) ----------
    peft_dir = base_out / "lora_adapter"
    trainer.model.save_pretrained(str(peft_dir))
    print(f"LoRA adapter saved to: {peft_dir}")

    # --------- Tiny inference demo (fresh base + adapter) ----------
    print("\n--- 🔍 Inference Demo ---")
    # Rebuild a fresh base to avoid double-adapter stacking
    fresh_base = GemmaLM(V, d_model, n_layers, n_heads, d_ff, max_position=seq_len, tie_weights=tie_weights)
    fresh_base.load_state_dict(_cast_state_dict_to_dtype(sd, next(fresh_base.parameters()).dtype), strict=False)

    clf = PeftModel.from_pretrained(fresh_base, str(peft_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf.to(device).eval()

    demo_headline = "You Won't BELIEVE What Happened Next!" if args.lang == "en" else "आप विश्वास नहीं करेंगे कि आगे क्या हुआ!"
    if args.lang == "hi":
        prompt = f'क्लिकबेट डिटेक्शन: क्या हेडलाइन "{demo_headline}" क्लिकबेट होने की संभावना है? उत्तर: '
        pos, neg = "हाँ", "नहीं"
    else:
        prompt = f'Clickbait Detection: Is the headline "{demo_headline}" likely clickbait? Answer: '
        pos, neg = "Yes", "No"

    enc = tok(prompt)["input_ids"]
    ids = torch.tensor(enc, device=device)

    @torch.inference_mode()
    def greedy_decode(m, ids, max_new=3, eos_id=None):
        out_ids = ids.clone()
        for _ in range(max_new):
            out = m(input_ids=out_ids)
            logits = out["logits"]
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            out_ids = torch.cat([out_ids, nxt], dim=1)
            if eos_id is not None and int(nxt.item()) == int(eos_id): break
        return out_ids

    gen = greedy_decode(clf, ids, max_new=3, eos_id=tok.eos_token_id)
    text = tok.batch_decode([gen[0].tolist()], skip_special_tokens=True)[0]
    pred = text.split("Answer:")[-1].split("उत्तर:")[-1].strip()
    # very light normalization
    pred_norm = pred.lower()
    if args.lang == "hi":
        final = pos if ("हाँ" in pred_norm or "han" in pred_norm) else (neg if ("नहीं" in pred_norm or "nahi" in pred_norm) else pred)
    else:
        final = "Yes" if "yes" in pred_norm else ("No" if "no" in pred_norm else pred)

    print(f'Headline: {demo_headline}\nModel Prediction: {final}')

# -----------------------------
#             CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune Gemma-style LM for Clickbait Detection (EN/HI) with LoRA.")
    ap.add_argument("--lang", type=str, required=True, choices=["en","hi"], help="Language code.")
    ap.add_argument("--checkpoint_path", type=Path, required=True, help="Path to base .pt checkpoint.")
    ap.add_argument("--tokenizer_path", type=Path, required=True, help="Path to SentencePiece .model.")
    ap.add_argument("--dataset_path", type=Path, required=True, help="Path to CSV/TSV with headline + label.")
    ap.add_argument("--delimiter", type=str, default=None, help="Override autodetect: ',' or '\\t'")
    ap.add_argument("--run_name", type=str, default=lambda: time.strftime("run_%Y%m%d_%H%M%S"),
                    help="Folder name under /clickbait/<lang>/ to avoid overriding (default: timestamp).")
    args = ap.parse_args()
    # argparse treats callable defaults awkwardly; normalize run_name:
    if not isinstance(args.run_name, str):
        args.run_name = time.strftime("run_%Y%m%d_%H%M%S")
    main(args)
