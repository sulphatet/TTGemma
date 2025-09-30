# -*- coding: utf-8 -*-
"""
Gemma-Style LM Trainer (~150M params) — Single-Node Multi-GPU (DDP), 3B-token epoch
Final: 2025-09-10
"""
from __future__ import annotations
import os, io, json, math, time, glob, random, argparse, csv, signal, shutil
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Iterator

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast as cuda_autocast, GradScaler
from datetime import timedelta

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# ---------- DDP helpers ----------
def is_distributed() -> bool: return dist.is_available() and dist.is_initialized()
def get_dist_info():
    if not is_distributed(): return 0,1,0,True
    r=dist.get_rank(); w=dist.get_world_size(); lr=int(os.environ.get("LOCAL_RANK",0)); return r,w,lr,(r==0)
def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend="nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(seconds=int(os.environ.get("DDP_TIMEOUT", 1800))))
        if torch.cuda.is_available(): torch.cuda.set_device(int(os.environ.get("LOCAL_RANK",0)))
def ddp_cleanup():
    if is_distributed(): dist.barrier(); dist.destroy_process_group()
def barrier(): 
    if is_distributed(): dist.barrier()
def bcast_small_int(x:int, device)->int:
    t=torch.tensor([x],device=device,dtype=torch.int64)
    if is_distributed(): dist.broadcast(t,src=0)
    return int(t.item())
def allreduce_sum_int(x:int, device)->int:
    t=torch.tensor([x],device=device,dtype=torch.int64)
    if is_distributed(): dist.all_reduce(t,op=dist.ReduceOp.SUM)
    return int(t.item())

# ---------- Tokenizer ----------
try:
    import sentencepiece as spm
except Exception as e:
    raise RuntimeError("Please install sentencepiece: pip install sentencepiece") from e

class SPMTokenizer:
    def __init__(self, spm_model_path: str, verbose: bool=True):
        assert os.path.exists(spm_model_path), f"SPM not found: {spm_model_path}"
        if verbose: print(f"[tokenizer] Loading: {spm_model_path}")
        self.sp = spm.SentencePieceProcessor(); self.sp.Load(spm_model_path)
        self.vocab_size = int(self.sp.GetPieceSize())
        def pid(p): 
            try: i=self.sp.PieceToId(p); return int(i) if i>=0 else None
            except: return None
        self.id_eos = pid("</s>")
        self.lang_tags = {"eng": pid("<eng>"), "hin": pid("<hin>"), "nep": pid("<nep>")}
        if verbose: print(f"[tokenizer] vocab_size={self.vocab_size} | tags={self.lang_tags}")
    def encode(self, text: str, lang: Optional[str]=None, add_eos: bool=True) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        if lang and self.lang_tags.get(lang) is not None: ids = [self.lang_tags[lang]] + ids
        if add_eos and self.id_eos is not None: ids = ids + [self.id_eos]
        return ids

# ---------- JSONL stream ----------
def jsonl_iter_texts(path: str):
    import gzip, lzma
    if not os.path.exists(path): return
    op=open; p=path.lower()
    if p.endswith(".gz"): op=gzip.open
    elif p.endswith(".xz"): op=lzma.open
    with op(path,"rb") as fb:
        with io.TextIOWrapper(fb,encoding="utf-8",errors="replace",newline="") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    rec=json.loads(line); t=rec.get("text","")
                    if t: yield t
                except: continue

def list_files_for_lang(splits_root: str, split: str, lang: str) -> List[str]:
    p=os.path.join(splits_root,lang,f"{split}.jsonl")
    return [p] if os.path.exists(p) else []

class LangPackedStream:
    """Infinite packed (seq_len+1) chunks; per-rank sharded by chunk index."""
    def __init__(self, files, tok, lang, seq_len, shard_rank=0, shard_world=1, print_every_docs=5000):
        assert files, f"No files for {lang}"
        self.files=files[:]; self.tok=tok; self.lang=lang; self.seq_len=seq_len
        self.print_every_docs=print_every_docs; self._fi=0; self._printed=0; self._buf=[]; self._gidx=0
        self.shard_rank=shard_rank; self.shard_world=max(1,shard_world)
    def _next_file(self): p=self.files[self._fi%len(self.files)]; self._fi+=1; return p
    def __iter__(self):
        while True:
            path=self._next_file()
            for text in jsonl_iter_texts(path):
                ids=self.tok.encode(text, lang=self.lang, add_eos=True)
                if not ids: continue
                self._buf.extend(ids); self._printed+=1
                if self.print_every_docs and (self._printed%self.print_every_docs==0):
                    print(f"[data:{self.lang}] streamed {self._printed} docs ({os.path.basename(path)})")
                while len(self._buf) >= (self.seq_len+1):
                    chunk=self._buf[:self.seq_len+1]; self._buf=self._buf[self.seq_len+1:]
                    if (self._gidx % self.shard_world) == self.shard_rank:
                        yield torch.tensor(chunk,dtype=torch.long)
                    self._gidx+=1

# ---- FAST PACKED READER (sequential, pinned) -------------------------------
import numpy as np

class ShardedMemmap:
    """Read-only multi-shard [rows, T] with contiguous cursors per shard."""
    def __init__(self, split_dir: str):
        assert os.path.isdir(split_dir), f"missing dir: {split_dir}"
        files = sorted(f for f in os.listdir(split_dir)
                       if f.startswith("shard_") and f.endswith(".npy"))
        assert files, f"no shard_*.npy in {split_dir}"
        self.paths = [os.path.join(split_dir, f) for f in files]
        self.arrs, self.rows, self.cursors = [], [], []
        self.T = None
        for p in self.paths:
            a = np.load(p, mmap_mode="r")
            if self.T is None: self.T = int(a.shape[1])
            else: assert int(a.shape[1]) == self.T
            self.arrs.append(a)
            self.rows.append(int(a.shape[0]))
            self.cursors.append(0)
        self.nshard = len(self.arrs)
        self.total_rows = int(sum(self.rows))

    def next_block(self, shard_id: int, n: int) -> np.ndarray:
        """Return n contiguous rows from shard_id (wrap-around)."""
        a = self.arrs[shard_id]; R = self.rows[shard_id]
        c = self.cursors[shard_id]
        if n <= R - c:
            out = a[c:c+n]
            self.cursors[shard_id] = (c + n) % R
            return out
        # wrap: split read
        first = a[c:R]
        rest  = a[0:(n - (R - c))]
        self.cursors[shard_id] = (n - (R - c)) % R
        return np.concatenate([first, rest], axis=0)

class PackedLangCycler:
    """
    Per-language cycler that:
      - assigns a preferred shard per rank (good NUMA/IO locality)
      - draws contiguous blocks -> cache-friendly
      - returns pinned LongTensor for fast non_blocking .to(cuda)
    """
    def __init__(self, packed_root: str, seq_len: int, shard_rank: int = 0, shard_world: int = 1):
        self.langs = {}
        for L in ("eng","hin","nep"):
            mm = ShardedMemmap(os.path.join(packed_root, L, "train"))
            assert mm.T == seq_len + 1, f"{L} shard T={mm.T} != {seq_len+1}"
            self.langs[L] = mm
        self.rank = int(shard_rank)
        self.world = max(1, int(shard_world))
        # pick a stable shard per (lang, rank)
        self.pref_shard = {}
        for L, mm in self.langs.items():
            self.pref_shard[L] = (self.rank % mm.nshard)

    def make_batch(self, lang: str, micro_bsz: int) -> torch.Tensor:
        mm = self.langs[lang]
        sid = self.pref_shard[lang]
        # grab contiguous block; convert once; pin it
        block = mm.next_block(sid, micro_bsz)          # np[int32] [B, T]
        if block.dtype != np.int64:
            # keep ids as 32-bit on disk, upcast on the fly for torch embedding index
            t = torch.from_numpy(block.astype(np.int64, copy=False))
        else:
            t = torch.from_numpy(block)
        return t.pin_memory()                           # critical for fast H2D
# ---------------------------------------------------------------------------


# ---------- Model ----------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5): super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x): return self.weight*(x*torch.rsqrt(torch.mean(x*x,dim=-1,keepdim=True)+self.eps))

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden, drop=0.0): super().__init__(); self.w1=nn.Linear(dim,hidden,bias=False); self.w2=nn.Linear(dim,hidden,bias=False); self.proj=nn.Linear(hidden,dim,bias=False); self.drop=nn.Dropout(drop)
    def forward(self,x): return self.drop(self.proj(self.w1(x)*F.silu(self.w2(x))))

def _rope_freqs(hd, base, max_pos, device, dtype):
    half=hd//2; inv=1.0/(base**(torch.arange(0,half,device=device,dtype=dtype)/half)); pos=torch.arange(0,max_pos,device=device,dtype=dtype)
    f=torch.outer(pos,inv); return torch.cos(f), torch.sin(f)

def _apply_rope(q,k,cos,sin):
    Dh=q.shape[-1]; half=Dh//2
    def rot(x):
        x1,x2=x[...,:half],x[...,half:]; T=x.size(2)
        return torch.cat([x1*cos[:T]-x2*sin[:T], x2*cos[:T]+x1*sin[:T]],dim=-1)
    return rot(q),rot(k)

class DecoderBlock(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,drop=0.0,rope_base=10000.0,max_position=4096):
        super().__init__(); assert d_model%n_heads==0
        self.n_heads=n_heads; self.d_head=d_model//n_heads
        self.norm1=RMSNorm(d_model); self.qkv=nn.Linear(d_model,3*d_model,bias=False); self.proj=nn.Linear(d_model,d_model,bias=False)
        self.norm2=RMSNorm(d_model); self.mlp=SwiGLU(d_model,d_ff,drop); self.drop=nn.Dropout(drop)
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
        attn=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=True)
        attn=attn.transpose(1,2).contiguous().view(B,T,C)
        x=x+self.drop(self.proj(attn)); h=self.norm2(x); x=x+self.drop(self.mlp(h)); return x

class GemmaLM(nn.Module):
    def __init__(self,V,d_model,n_layers,n_heads,d_ff,max_position,drop=0.0,tie_weights=True):
        super().__init__()
        self.tok_emb=nn.Embedding(V,d_model); self.drop=nn.Dropout(drop)
        self.blocks=nn.ModuleList([DecoderBlock(d_model,n_heads,d_ff,drop,10000.0,max_position) for _ in range(n_layers)])
        self.norm_f=RMSNorm(d_model); self.lm_head=nn.Linear(d_model,V,bias=False)
        if tie_weights: self.lm_head.weight=self.tok_emb.weight
    def forward(self,idx):
        x=self.drop(self.tok_emb(idx))
        for b in self.blocks: x=b(x)
        x=self.norm_f(x); return self.lm_head(x)

def count_params(m): return sum(p.numel() for p in m.parameters())

PRESETS={"150M_plan":dict(d_model=640,n_layers=16,n_heads=10,d_ff=2688,tie_weights=True)}

# ---------- Config ----------
@dataclass
class TrainConfig:
    spm_model_path: str
    splits_root: str
    run_name: str = "gemma-150M"
    save_to: str = "/home2/USER/LMA_SLM/checkpoints"
    token_budget_ledger: str = "/home2/USER/LMA_SLM/training/token_budget.json"
    token_budget_total: int = 12_000_000_000

    model_preset: str = "150M_plan"
    seq_len: int = 1024
    dropout: float = 0.05
    amp_dtype: str = "fp16"  # 1080/2080 -> fp16 path

    packed_root: str = ""   # empty => use JSONL path

    micro_bsz: int = 1       # per-rank
    grad_accum: int = 128

    warmup_steps: int = 1500
    lr_max: float = 2e-4
    lr_min: float = 1e-5
    weight_decay: float = 0.05
    grad_clip: float = 1.0

    val_interval: int = 5000     # tightened to 5k
    ckpt_interval: int = 5000    # tightened to 5k

    eval_tokens_per_lang: int = 500_000

    epoch_tokens_eng: int = 1_200_000_000
    epoch_tokens_hin: int = 1_100_000_000
    epoch_tokens_nep: int =   700_000_000   # sum=3.0B

    resume: bool = True
    session_max_tokens: Optional[int] = None
    seed: int = 1337

    # ----- NEW: checkpoint size/retention -----
    ckpt_dtype: str = "fp16"     # fp16|bf16|fp32 — saved model weights dtype
    keep_last_k: int = 3         # keep only latest K checkpoints
    save_optimizer: bool = True  # include optimizer/scaler for robust resume

# ---------- misc utils ----------
def set_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def cosine_lr(it,max_it,warmup,lr_max,lr_min):
    if it<warmup: return lr_max*(it+1)/max(1,warmup)
    if it>=max_it: return lr_min
    r=(it-warmup)/max(1,(max_it-warmup)); return lr_min+0.5*(1.0+math.cos(math.pi*r))*(lr_max-lr_min)
def read_token_budget(path,total):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    if os.path.exists(path):
        try:
            j=json.load(open(path,"r")); j.setdefault("budget_total",total); j.setdefault("budget_used",0); return j
        except: pass
    j={"budget_total":total,"budget_used":0}; json.dump(j,open(path,"w"),indent=2); return j
def _atomic_write_json(path,obj):
    tmp=path+".tmp"; json.dump(obj,open(tmp,"w"),indent=2)
    try: os.replace(tmp,path)
    except: 
        try: shutil.copy2(tmp,path); os.remove(tmp)
        except: pass
def _cast_state_dict_params(state_dict, target_dtype):
    new={}
    for k,v in state_dict.items():
        if isinstance(v,torch.Tensor) and v.is_floating_point():
            try: new[k]=v.to(dtype=target_dtype)
            except: new[k]=v
        else: new[k]=v
    return new
def _autocast_enabled(dtype): return dtype in (torch.float16, torch.bfloat16)

# ---------- Eval ----------
@torch.inference_mode()
def evaluate_per_language(model, tok, device, splits_root, seq_len, batch, max_tokens, amp_dtype):
    results={}; ce=nn.CrossEntropyLoss(reduction="sum")
    for lang in ("eng","hin","nep"):
        files=list_files_for_lang(splits_root,"val",lang)
        if not files: print(f"[eval] {lang}: no val.jsonl, skipping."); continue
        stream=LangPackedStream(files,tok,lang,seq_len,shard_rank=0,shard_world=1)
        it=iter(stream); total_loss=0.0; total_tok=0; total_correct=0; t0=time.time()
        while total_tok<max_tokens:
            chunks=[]
            for _ in range(batch):
                try: chunks.append(next(it))
                except StopIteration:
                    it=iter(LangPackedStream(files,tok,lang,seq_len)); chunks.append(next(it))
            b=torch.stack(chunks,0); x=b[:,:-1].to(device); y=b[:,1:].to(device)
            with cuda_autocast(dtype=amp_dtype, enabled=_autocast_enabled(amp_dtype)):
                logits=model(x); loss=ce(logits.reshape(-1,logits.size(-1)), y.reshape(-1)); pred=logits.argmax(-1)
            total_loss+=float(loss); ntok=int(y.numel()); total_tok+=ntok; total_correct+=int((pred==y).sum().item())
        dt=max(1e-6,time.time()-t0); m=total_loss/max(1,total_tok); ppl=math.exp(min(20.0,m)); bpt=m/math.log(2); acc=total_correct/max(1,total_tok)
        results[lang]=dict(loss_nats=m,ppl=ppl,bpt=bpt,acc_top1=acc,tokens=total_tok,tps=total_tok/dt)
        print(f"[eval:{lang}] loss={m:.4f} ppl={ppl:.2f} bpt={bpt:.3f} acc={acc*100:.2f}% tokens={total_tok:,}")
    return results

# ---------- Train ----------
def train(cfg: TrainConfig):
    # ----- DDP init -----
    ddp_setup()
    rank, world, local_rank, is_master = get_dist_info()

    # perf toggles that exist on torch 2.1.x
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(cfg.seed + rank)
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    if is_master:
        print(f"[env] torch={torch.__version__} device={'cuda' if use_cuda else 'cpu'} world={world}")
        if use_cuda:
            print(f"[env] GPU[{local_rank}]: {torch.cuda.get_device_name(device)}  BF16? {torch.cuda.is_bf16_supported()}")

    # ----- tokenizer -----
    tok = SPMTokenizer(cfg.spm_model_path, verbose=is_master)
    V = tok.vocab_size

    # ----- model -----
    mp = PRESETS[cfg.model_preset]
    model = GemmaLM(
        V, mp["d_model"], mp["n_layers"], mp["n_heads"], mp["d_ff"],
        cfg.seq_len, drop=cfg.dropout, tie_weights=mp.get("tie_weights", True)
    )

    want_bf16 = (cfg.amp_dtype.lower() == "bf16")
    want_fp16 = (cfg.amp_dtype.lower() == "fp16")
    use_bf16  = bool(want_bf16 and use_cuda and torch.cuda.is_bf16_supported())
    use_fp16  = bool(want_fp16 and use_cuda and not use_bf16)
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    if use_fp16:
        model = model.to(device)  # FP32 master params; compute in fp16 via autocast
        if is_master: print("[amp] fp16 autocast; FP32 master params.")
    elif use_bf16:
        model = model.to(device, dtype=torch.bfloat16)
        if is_master: print("[amp] BF16 params.")
    else:
        model = model.to(device)
        if is_master: print("[amp] FP32.")

    if world > 1:
        model = DDP(
            model,
            device_ids=[device.index] if use_cuda else None,
            output_device=(device.index if use_cuda else None),
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    if is_master:
        nparams = count_params(model if world == 1 else model.module) / 1e6
        print(f"[model] params≈{nparams:.2f}M")

    # ----- optim/scaler -----
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_max, betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.weight_decay
    )
    scaler = GradScaler(enabled=use_fp16)

    # ----- DATA: packed or JSONL path -----
    use_packed = bool(getattr(cfg, "packed_root", "") or "")
    if use_packed and is_master:
        print(f"[data] using packed shards at: {cfg.packed_root}")

    if use_packed:
        # fast memmapped shards (train); eval can stay JSONL
        cycler = PackedLangCycler(cfg.packed_root, cfg.seq_len, shard_rank=rank, shard_world=world)
        # for eval JSONL we still need file lists later
        files_eng = list_files_for_lang(cfg.splits_root, "val", "eng")
        files_hin = list_files_for_lang(cfg.splits_root, "val", "hin")
        files_nep = list_files_for_lang(cfg.splits_root, "val", "nep")
    else:
        files_eng = list_files_for_lang(cfg.splits_root, "train", "eng")
        files_hin = list_files_for_lang(cfg.splits_root, "train", "hin")
        files_nep = list_files_for_lang(cfg.splits_root, "train", "nep")
        if is_master:
            assert files_eng and files_hin and files_nep, "[data] Missing train.jsonl for a language."
        barrier()
        stream_eng = LangPackedStream(files_eng, tok, "eng", cfg.seq_len, shard_rank=rank, shard_world=world)
        stream_hin = LangPackedStream(files_hin, tok, "hin", cfg.seq_len, shard_rank=rank, shard_world=world)
        stream_nep = LangPackedStream(files_nep, tok, "nep", cfg.seq_len, shard_rank=rank, shard_world=world)
        iters = {"eng": iter(stream_eng), "hin": iter(stream_hin), "nep": iter(stream_nep)}

    # ----- quotas -----
    epoch_targets  = {"eng": cfg.epoch_tokens_eng, "hin": cfg.epoch_tokens_hin, "nep": cfg.epoch_tokens_nep}
    epoch_counters = {"eng": 0, "hin": 0, "nep": 0}
    if is_master:
        eff = cfg.seq_len * cfg.micro_bsz * cfg.grad_accum * world
        print(f"[epoch plan] {epoch_targets} total={sum(epoch_targets.values()):,}  eff_tokens/step={eff:,}")

    # ----- ledger / run dir / metrics (rank0 only) -----
    if is_master:
        ledger = read_token_budget(cfg.token_budget_ledger, cfg.token_budget_total)
        print(f"[budget] total={ledger['budget_total']:,} used={ledger['budget_used']:,}")
    else:
        ledger = None

    run_dir = os.path.join(cfg.save_to, cfg.run_name)
    if is_master:
        os.makedirs(run_dir, exist_ok=True)
        json.dump(asdict(cfg), open(os.path.join(run_dir, "train_config.json"), "w"), indent=2)
        mloc = f"/tmp/{cfg.run_name}_metrics.jsonl"
        cloc = f"/tmp/{cfg.run_name}_metrics.csv"
        mdrv = os.path.join(run_dir, "metrics.jsonl")
        cdrv = os.path.join(run_dir, "metrics.csv")
        mj = open(mloc, "a", buffering=1)
        cf = open(cloc, "a", newline="")
        new = not os.path.exists(cdrv)
        cw = csv.DictWriter(cf, fieldnames=[
            "time","epoch","step","tokens_seen","eng_tok","hin_tok","nep_tok","lr","train_loss","grad_norm",
            "val_loss_eng","val_ppl_eng","val_loss_hin","val_ppl_hin","val_loss_nep","val_ppl_nep"
        ])
        if new: cw.writeheader(); cf.flush()
        def _flush():
            try:
                if os.path.exists(mloc): shutil.copy2(mloc, mdrv)
                if os.path.exists(cloc): shutil.copy2(cloc, cdrv)
            except Exception as e:
                print("[warn] metrics flush:", e)
    barrier()

    # ----- resume (rank0 finds latest, path broadcast to others) -----
    step = 0; epoch_idx = 0; tokens_seen_global = 0
    latest = None
    if is_master and cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_e*_s*.pt")))
        latest = ckpts[-1] if ckpts else None

    if is_distributed():
        has = 1 if latest else 0
        t = torch.tensor([has], device=device)
        dist.broadcast(t, src=0)
        has = int(t.item())
        if not has:
            latest = None
        else:
            if is_master:
                b = latest.encode()
                tn = torch.tensor([len(b)], device=device)
                dist.broadcast(tn, src=0)
                tb = torch.tensor(list(b), device=device, dtype=torch.uint8)
                dist.broadcast(tb, src=0)
            else:
                tn = torch.tensor([0], device=device)
                dist.broadcast(tn, src=0)
                n = int(tn.item())
                tb = torch.empty(n, device=device, dtype=torch.uint8)
                dist.broadcast(tb, src=0)
                latest = bytes(tb.tolist()).decode()

    if latest:
        if is_master: print(f"[resume] {latest}")
        ck = torch.load(latest, map_location="cpu")
        mod = model.module if isinstance(model, DDP) else model
        tgt_dtype = next(mod.parameters()).dtype
        state = _cast_state_dict_params(ck.get("model", {}), tgt_dtype)
        mod.load_state_dict(state, strict=False)
        try:
            opt.load_state_dict(ck.get("opt", {}))
        except Exception:
            if is_master: print("[resume] opt restore failed; continuing.")
        step = int(ck.get("step", 0)); epoch_idx = int(ck.get("epoch_idx", 0)); tokens_seen_global = int(ck.get("tokens_seen", 0))
        epoch_counters = ck.get("epoch_counters", epoch_counters)
        try:
            if "py_random_state" in ck: random.setstate(ck["py_random_state"])
            if "np_rng_state" in ck: np.random.set_state(ck["np_rng_state"])
            if "torch_rng_state" in ck: torch.random.set_rng_state(ck["torch_rng_state"])
            if torch.cuda.is_available() and ck.get("torch_cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(ck["torch_cuda_rng_state"])
        except Exception:
            pass
        if ck.get("scaler_state") is not None:
            try: scaler.load_state_dict(ck["scaler_state"])
            except Exception: pass
        if is_master:
            print(f"[resume] step={step} epoch={epoch_idx} tokens_seen={tokens_seen_global:,} counters={epoch_counters}")
    barrier()

    # ----- checkpoint helpers -----
    def _ckpt_tensor_dtype(s: str) -> torch.dtype:
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(s.lower(), torch.float16)

    def _list_ckpts():
        return sorted(glob.glob(os.path.join(run_dir, "ckpt_e*_s*.pt")))

    def _prune_ckpts(keep: int):
        ck = _list_ckpts()
        if len(ck) > keep:
            for p in ck[:-keep]:
                try: os.remove(p); print(f"[ckpt] pruned {os.path.basename(p)}")
                except Exception: pass

    def save_ckpt():
        if not is_master: return
        mod = model.module if isinstance(model, DDP) else model
        sd  = mod.state_dict()
        target = _ckpt_tensor_dtype(cfg.ckpt_dtype)
        sd_cast = _cast_state_dict_params(sd, target)
        payload = {
            "model": sd_cast, "step": step, "epoch_idx": epoch_idx, "tokens_seen": tokens_seen_global,
            "epoch_counters": epoch_counters, "cfg": asdict(cfg),
            "py_random_state": random.getstate(), "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        if cfg.save_optimizer:
            try:
                payload["opt"] = opt.state_dict()
                payload["scaler_state"] = scaler.state_dict()
            except Exception:
                pass
        tmp = os.path.join(run_dir, f".tmp_ckpt_e{epoch_idx}_s{step}.pt")
        fin = os.path.join(run_dir, f"ckpt_e{epoch_idx}_s{step}.pt")
        try:
            torch.save(payload, tmp); os.replace(tmp, fin)
        except Exception:
            try: torch.save(payload, fin)
            except Exception as e: print("[warn] ckpt save:", e)
        try:
            if ledger is not None: _atomic_write_json(cfg.token_budget_ledger, ledger)
        except Exception:
            pass
        try: _flush()
        except Exception: pass
        print(f"[ckpt] saved {os.path.basename(fin)}")
        _prune_ckpts(cfg.keep_last_k)

    def on_sigterm(signum, frame):
        if is_master: print("[signal] SIGTERM -> saving checkpoint...")
        save_ckpt()
        raise SystemExit(0)

    if is_master:
        signal.signal(signal.SIGTERM, on_sigterm)

    # ----- selection + batching helpers -----
    lang2id = {"eng": 0, "hin": 1, "nep": 2}
    id2lang = {v: k for k, v in lang2id.items()}

    def pick_lang():
        rem = {k: max(0, epoch_targets[k] - epoch_counters[k]) for k in epoch_targets}
        tot = sum(rem.values())
        if tot <= 0: return ""
        r = random.random() * tot; acc = 0
        for L in ("eng", "hin", "nep"):
            acc += rem[L]
            if r <= acc: return L
        return "eng"

    def make_batch(L, B):
        if use_packed:
            return cycler.make_batch(L, B)     # CPU LongTensor [B, T]
        # fallback to your JSONL loader
        it = iters[L]; chunks = []
        while len(chunks) < B:
            try: chunks.append(next(it))
            except StopIteration:
                if L == "eng": iters["eng"] = iter(LangPackedStream(files_eng, tok, "eng", cfg.seq_len, rank, world))
                elif L == "hin": iters["hin"] = iter(LangPackedStream(files_hin, tok, "hin", cfg.seq_len, rank, world))
                else: iters["nep"] = iter(LangPackedStream(files_nep, tok, "nep", cfg.seq_len, rank, world))
                it = iters[L]
        return torch.stack(chunks, 0)


    if is_master:
        eff = cfg.seq_len * cfg.micro_bsz * cfg.grad_accum * world
        print(f"[train] eff tokens/step={eff:,}")

    # ----- main loop -----
    running_loss = None
    last_metrics = time.time()
    last_ledger  = time.time()

    while True:
        # decide language on rank0 and broadcast id
        if is_master:
            L = pick_lang()
            if not L:
                print(f"[epoch {epoch_idx}] DONE quotas: {epoch_counters} / {epoch_targets}")
        else:
            L = "eng"
        lang_id = bcast_small_int(lang2id.get(L, -1), device)
        if lang_id < 0:
            barrier()
            if is_master:
                save_ckpt()
                target = (model.module if isinstance(model, DDP) else model).eval()
                print("[eval] running val...")
                ev = evaluate_per_language(
                    target, tok, device, cfg.splits_root, cfg.seq_len,
                    batch=cfg.micro_bsz, max_tokens=cfg.eval_tokens_per_lang, amp_dtype=amp_dtype
                )
                row = {
                    "time": time.time(), "epoch": epoch_idx, "step": step, "tokens_seen": tokens_seen_global,
                    "eng_tok": epoch_counters["eng"], "hin_tok": epoch_counters["hin"], "nep_tok": epoch_counters["nep"],
                    "lr": None, "train_loss": None, "grad_norm": None
                }
                for l in ("eng", "hin", "nep"):
                    if l in ev:
                        row[f"val_loss_{l}"] = ev[l]["loss_nats"]
                        row[f"val_ppl_{l}"]  = ev[l]["ppl"]
                mj.write(json.dumps({"type": "eval", "epoch": epoch_idx, "step": step, "tokens_seen": tokens_seen_global, "eval": ev}) + "\n")
                cw.writerow(row); cf.flush(); mj.flush(); _flush()
                epoch_idx += 1; epoch_counters = {"eng": 0, "hin": 0, "nep": 0}
                target.train()
            barrier()
            continue

        L = id2lang[lang_id]

        # LR schedule
        lr = cosine_lr(step, max_it=200_000, warmup=cfg.warmup_steps, lr_max=cfg.lr_max, lr_min=cfg.lr_min)
        for g in opt.param_groups: g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_tokens_rank = 0
        micro_losses = []
        t0 = time.time()

        for _ in range(cfg.grad_accum):
            b = make_batch(L, cfg.micro_bsz)        # [B, T+1] on CPU
            x = b[:, :-1].to(device, non_blocking=True)
            y = b[:, 1: ].to(device, non_blocking=True)
            step_tokens_rank += int(x.numel())
            with cuda_autocast(dtype=amp_dtype, enabled=(amp_dtype in (torch.float16, torch.bfloat16))):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / cfg.grad_accum
            if use_fp16: scaler.scale(loss).backward()
            else:        loss.backward()
            micro_losses.append(float(loss.detach().cpu()))

        if use_fp16:
            try: scaler.unscale_(opt)
            except Exception: pass
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if use_fp16: scaler.step(opt); scaler.update()
        else:        opt.step()

        # global token accounting
        step_tokens_global = allreduce_sum_int(step_tokens_rank, device)
        dt  = max(1e-6, time.time() - t0)
        tps = step_tokens_global / dt
        running_loss = (0.98 * running_loss + 0.02 * sum(micro_losses)) if running_loss is not None else sum(micro_losses)

        if is_master:
            tokens_seen_global += step_tokens_global
            epoch_counters[L]  += step_tokens_global
            if (step % 20) == 0:
                print(f"[step {step:06d} | epoch {epoch_idx} | {L}] lr={lr:.2e} loss={running_loss:.4f} "
                      f"gn={float(gn):.3f} t/s={int(tps):,} seen={tokens_seen_global:,} quotas={epoch_counters}")

        # periodic eval
        if is_master and (step % cfg.val_interval) == 0 and step > 0:
            target = (model.module if isinstance(model, DDP) else model).eval()
            ev = evaluate_per_language(
                target, tok, device, cfg.splits_root, cfg.seq_len,
                batch=cfg.micro_bsz, max_tokens=cfg.eval_tokens_per_lang, amp_dtype=amp_dtype
            )
            row = {
                "time": time.time(), "epoch": epoch_idx, "step": step, "tokens_seen": tokens_seen_global,
                "eng_tok": epoch_counters["eng"], "hin_tok": epoch_counters["hin"], "nep_tok": epoch_counters["nep"],
                "lr": lr, "train_loss": running_loss, "grad_norm": float(gn),
            }
            for l in ("eng", "hin", "nep"):
                if l in ev:
                    row[f"val_loss_{l}"] = ev[l]["loss_nats"]
                    row[f"val_ppl_{l}"]  = ev[l]["ppl"]
            mj.write(json.dumps({"type": "eval", "epoch": epoch_idx, "step": step, "tokens_seen": tokens_seen_global,
                                 "train_loss": running_loss, "lr": lr, "eval": ev}) + "\n")
            cw.writerow(row); cf.flush(); mj.flush()
            target.train()

        # periodic ckpt
        if is_master and (step % cfg.ckpt_interval) == 0 and step > 0:
            save_ckpt()

        # ledger + metrics flush
        if is_master and ledger is not None:
            now = time.time()
            ledger["budget_used"] += step_tokens_global
            if now - last_ledger > 60:
                try: _atomic_write_json(cfg.token_budget_ledger, ledger)
                except Exception: pass
                last_ledger = now
            if now - last_metrics > 300:
                try: mj.flush(); cf.flush(); _flush()
                except Exception: pass
                last_metrics = now
            remain = ledger["budget_total"] - ledger["budget_used"]
            if remain <= 0:
                print(f"[budget] EXHAUSTED ({ledger['budget_used']:,}/{ledger['budget_total']:,}). Saving & stop.")
                save_ckpt()
                break
            elif remain < 5_000_000:
                print(f"[budget] Heads-up: {remain:,} tokens remain.")

        if cfg.session_max_tokens is not None and is_master and tokens_seen_global >= cfg.session_max_tokens:
            print(f"[session] cap reached ({tokens_seen_global:,}). Saving & exit.")
            save_ckpt()
            break

        step += 1

        # end-of-epoch check (rank0 decides, broadcasts)
        if is_master:
            done = all(epoch_counters[k] >= epoch_targets[k] for k in epoch_targets)
        else:
            done = False
        td = torch.tensor([1 if done else 0], device=device, dtype=torch.int32)
        if is_distributed(): dist.broadcast(td, src=0)
        if int(td.item()) == 1:
            if is_master:
                print(f"[epoch {epoch_idx}] Quotas reached; finishing.")
                save_ckpt()
            barrier()
            break

    if is_master:
        print(f"[done] steps={step} tokens_seen={tokens_seen_global:,}")
    ddp_cleanup()

# ---------- CLI ----------
def build_argparser():
    p=argparse.ArgumentParser(description="Gemma-Style 150M Trainer (DDP) w/ 3B-token epoch & compact checkpoints")
    p.add_argument("--spm_model_path",type=str,required=True)
    p.add_argument("--splits_root",type=str,required=True)
    p.add_argument("--run_name",type=str,default="gemma-150M")
    p.add_argument("--save_to",type=str,default="/home2/USER/LMA_SLM/checkpoints")
    p.add_argument("--token_budget_ledger",type=str,default="/home2/USER/LMA_SLM/training/token_budget.json")
    p.add_argument("--token_budget_total",type=int,default=12_000_000_000)

    p.add_argument("--model_preset",type=str,default="150M_plan",choices=list(PRESETS.keys()))
    p.add_argument("--seq_len",type=int,default=1024)
    p.add_argument("--dropout",type=float,default=0.05)
    p.add_argument("--amp_dtype",type=str,default="fp16",choices=["bf16","fp16","fp32"])

    p.add_argument("--micro_bsz",type=int,default=1)
    p.add_argument("--grad_accum",type=int,default=128)

    p.add_argument("--warmup_steps",type=int,default=1500)
    p.add_argument("--lr_max",type=float,default=2e-4)
    p.add_argument("--lr_min",type=float,default=1e-5)
    p.add_argument("--weight_decay",type=float,default=0.05)
    p.add_argument("--grad_clip",type=float,default=1.0)

    p.add_argument("--val_interval",type=int,default=5000)
    p.add_argument("--ckpt_interval",type=int,default=5000)
    p.add_argument("--eval_tokens_per_lang",type=int,default=500000)

    p.add_argument("--epoch_tokens_eng",type=int,default=1200000000)
    p.add_argument("--epoch_tokens_hin",type=int,default=1100000000)
    p.add_argument("--epoch_tokens_nep",type=int,default=700000000)
    p.add_argument("--packed_root", type=str, default="", help="If set, read pretokenized shards from here")


    p.add_argument("--resume",action="store_true"); p.add_argument("--no-resume",dest="resume",action="store_false"); p.set_defaults(resume=True)
    p.add_argument("--session_max_tokens",type=int,default=None)
    p.add_argument("--seed",type=int,default=1337)

    # NEW: checkpoint knobs
    p.add_argument("--ckpt_dtype",type=str,default="fp16",choices=["fp16","bf16","fp32"])
    p.add_argument("--keep_last_k",type=int,default=3)
    p.add_argument("--save_optimizer",type=lambda s: s.lower()!="false",default=True)
    return p

if __name__=="__main__":
    a=build_argparser().parse_args()
    cfg=TrainConfig(
        spm_model_path=a.spm_model_path, splits_root=a.splits_root, run_name=a.run_name,
        save_to=a.save_to, token_budget_ledger=a.token_budget_ledger, token_budget_total=a.token_budget_total,
        model_preset=a.model_preset, seq_len=a.seq_len, dropout=a.dropout, amp_dtype=a.amp_dtype,
        micro_bsz=a.micro_bsz, grad_accum=a.grad_accum, warmup_steps=a.warmup_steps,
        lr_max=a.lr_max, lr_min=a.lr_min, weight_decay=a.weight_decay, grad_clip=a.grad_clip,
        val_interval=a.val_interval, ckpt_interval=a.ckpt_interval, eval_tokens_per_lang=a.eval_tokens_per_lang,
        epoch_tokens_eng=a.epoch_tokens_eng, epoch_tokens_hin=a.epoch_tokens_hin, epoch_tokens_nep=a.epoch_tokens_nep,
        resume=a.resume, session_max_tokens=a.session_max_tokens, seed=a.seed,
        ckpt_dtype=a.ckpt_dtype, keep_last_k=a.keep_last_k, save_optimizer=a.save_optimizer, packed_root=a.packed_root,
    )
    if int(os.environ.get("RANK","0"))==0:
        print("[config]", json.dumps(asdict(cfg), indent=2))
    train(cfg)
