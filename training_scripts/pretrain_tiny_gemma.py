#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tiny Gemma-Style LM Pretrainer (1–10M params), HPC-friendly

- Single-file, PyTorch, decoder-only, RoPE, RMSNorm, SwiGLU.
- No Colab assumptions. Defaults save/checkpoints/logs to your HOME (~ = /home2/$USER).
- Token budget ledger is optional (and safe if unwritable).
- Good for quick smoke tests on CentOS7 + CUDA 11.x + Pascal (GTX 1080 Ti).

Author: ChatGPT (updated for ADA/gnode)
Date: 2025-09-10
"""

from __future__ import annotations
import os, io, re, json, math, time, glob, signal, random, argparse, sys
from dataclasses import dataclass, asdict
from typing import Optional, List, Iterator, Tuple, Dict

# -------------------------------
# Torch / NumPy / Datasets / SPM
# -------------------------------
import numpy as np  # torch 2.1.x expects numpy 1.26.x; on this cluster pin numpy<2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

try:
    import sentencepiece as spm
except Exception as e:
    raise RuntimeError("Please install sentencepiece: `pip install sentencepiece`") from e

try:
    from datasets import load_dataset  # only for dummy data path (not required normally)
except Exception:
    load_dataset = None

# -------------------------------
# Utility: seed + simple file logger
# -------------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TeeLogger:
    """
    Mirror prints to a log file. Use like:
      logger = TeeLogger(run_dir/'train.log')
      logger.print("hello")
    """
    def __init__(self, logfile: str):
        self.logfile = logfile
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        self._fh = open(logfile, "a", buffering=1, encoding="utf-8")
    def print(self, *a, **k):
        msg = " ".join(str(x) for x in a)
        print(msg, **k)
        try:
            self._fh.write(msg + ("\n" if not msg.endswith("\n") else ""))
        except Exception:
            pass
    def close(self):
        try: self._fh.close()
        except Exception: pass

# -------------------------------
# Tokenizer wrapper (SentencePiece)
# -------------------------------
class SPMTokenizer:
    def __init__(self, spm_model_path: str, verbose: bool=True):
        assert os.path.exists(spm_model_path), f"SPM model not found: {spm_model_path}"
        if verbose: print(f"[tokenizer] Loading SentencePiece model: {spm_model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_model_path)
        self.vocab_size = self.sp.GetPieceSize()

        def pid(piece: str) -> Optional[int]:
            try:
                i = self.sp.PieceToId(piece)
                return int(i) if i >= 0 else None
            except Exception:
                return None

        self.id_unk = pid("<unk>")
        self.id_bos = pid("<s>")
        self.id_eos = pid("</s>")
        self.lang_tags = {"eng": pid("<eng>"), "hin": pid("<hin>"), "nep": pid("<nep>")}
        if verbose:
            print(f"[tokenizer] vocab_size={self.vocab_size}, <unk>={self.id_unk}, <s>={self.id_bos}, </s>={self.id_eos}")
            print(f"[tokenizer] language tags: {self.lang_tags}")

    def encode(self, text: str, add_bos: bool=False, add_eos: bool=False, lang: Optional[str]=None) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        if lang and lang in self.lang_tags and self.lang_tags[lang] is not None:
            ids = [self.lang_tags[lang]] + ids
        if add_bos and self.id_bos is not None:
            ids = [self.id_bos] + ids
        if add_eos and self.id_eos is not None:
            ids = ids + [self.id_eos]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.sp.DecodeIds(list(map(int, ids)))

# -------------------------------
# Data: JSONL streaming + packing
# -------------------------------
def jsonl_iter_texts(path: str) -> Iterator[str]:
    """Stream 'text' field from JSONL with robust decoding."""
    import gzip, lzma
    if not os.path.exists(path):
        return
    open_fn = open
    lower = path.lower()
    if lower.endswith(".gz"): open_fn = gzip.open
    elif lower.endswith(".xz"): open_fn = lzma.open
    with open_fn(path, "rb") as fb:
        with io.TextIOWrapper(fb, encoding="utf-8", errors="replace", newline="") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                    t = rec.get("text", "")
                    if t: yield t
                except json.JSONDecodeError:
                    continue

def list_split_files(splits_root: str, split: str) -> List[Tuple[str, str]]:
    pairs = []
    for lang in ("eng", "hin", "nep"):
        p = os.path.join(splits_root, lang, f"{split}.jsonl")
        if os.path.exists(p):
            pairs.append((p, lang))
    return pairs

class PackedIterableDataset(IterableDataset):
    def __init__(
        self,
        files_with_lang: List[Tuple[str, str]],
        tokenizer: SPMTokenizer,
        seq_len: int,
        max_docs_per_lang: Optional[int]=None,
        add_bos: bool=False,
        add_eos: bool=True,
        shuffle_files_each_epoch: bool=True,
        print_every_docs: int=1000,
    ):
        super().__init__()
        self.files_with_lang = list(files_with_lang)
        self.tok = tokenizer
        self.seq_len = int(seq_len)
        self.max_docs_per_lang = max_docs_per_lang
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.shuffle_files_each_epoch = shuffle_files_each_epoch
        self.print_every_docs = print_every_docs

    def _doc_iter(self) -> Iterator[Tuple[List[int], str]]:
        files = self.files_with_lang[:]
        if self.shuffle_files_each_epoch:
            random.shuffle(files)
        per_lang = {}
        printed = 0
        for path, lang in files:
            per_lang.setdefault(lang, 0)
            for text in jsonl_iter_texts(path):
                if self.max_docs_per_lang and per_lang[lang] >= self.max_docs_per_lang:
                    break
                ids = self.tok.encode(text, add_bos=self.add_bos, add_eos=self.add_eos, lang=lang)
                if ids:
                    per_lang[lang] += 1
                    yield ids, lang
                    printed += 1
                    if self.print_every_docs and (printed % self.print_every_docs == 0):
                        print(f"[data] streamed {printed} docs so far (lang={lang})")

    def __iter__(self) -> Iterator[torch.Tensor]:
        buffer: List[int] = []
        for ids, _lang in self._doc_iter():
            buffer.extend(ids)
            while len(buffer) >= (self.seq_len + 1):
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                yield torch.tensor(chunk, dtype=torch.long)
        if len(buffer) >= (self.seq_len + 1):
            chunk = buffer[: self.seq_len + 1]
            yield torch.tensor(chunk, dtype=torch.long)

# -------------------------------
# Model bits: RMSNorm, SwiGLU, RoPE
# -------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_normed

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.proj(self.w1(x) * F.silu(self.w2(x))))

def precompute_rope_frequencies(head_dim: int, base: float=10000.0, device=None, dtype=None, max_position: int=8192):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    positions = torch.arange(0, max_position, device=device, dtype=dtype)
    freqs = torch.outer(positions, inv_freq)  # [T, half]
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    B, H, T, D = q.shape
    half = D // 2
    def rot(x):
        x1 = x[..., :half]; x2 = x[..., half:]
        return torch.cat([x1 * cos[:T] - x2 * sin[:T], x2 * cos[:T] + x1 * sin[:T]], dim=-1)
    return rot(q), rot(k)

# -------------------------------
# Factorized embeddings (optional)
# -------------------------------
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab: int, d_model: int, e_dim: int):
        super().__init__()
        self.E = nn.Embedding(vocab, e_dim)
        self.P = nn.Linear(e_dim, d_model, bias=False)
    def forward(self, idx):
        return self.P(self.E(idx))

# -------------------------------
# Tiny Gemma-style Decoder Block
# -------------------------------
class TinyGemmaBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, attn_dropout: float=0.0, resid_dropout: float=0.0, rope_base: float=10000.0, max_position: int=8192):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
        self.resid_drop = nn.Dropout(resid_dropout)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.rope_base = rope_base
        self.max_position = max_position
        self.register_buffer("_rope_cos", None, persistent=False)
        self.register_buffer("_rope_sin", None, persistent=False)

    def maybe_build_rope(self, device, dtype):
        if self._rope_cos is None or self._rope_cos.device != device or self._rope_cos.dtype != dtype:
            cos, sin = precompute_rope_frequencies(self.d_head, base=self.rope_base, device=device, dtype=dtype, max_position=self.max_position)
            self._rope_cos = cos
            self._rope_sin = sin

    def forward(self, x, is_causal: bool=True):
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        self.maybe_build_rope(device=q.device, dtype=q.dtype)
        q, k = apply_rope(q, k, self._rope_cos, self._rope_sin)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        attn = self.attn_drop(self.out(attn))
        x = x + self.resid_drop(attn)

        h = self.norm2(x)
        x = x + self.resid_drop(self.mlp(h))
        return x

# -------------------------------
# Full Tiny Gemma Decoder
# -------------------------------
class TinyGemmaLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_position: int = 8192,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        factorized_emb: bool = False,
        factor_e_dim: int = 64,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_position = max_position

        if factorized_emb:
            self.tok_emb = FactorizedEmbedding(vocab_size, d_model, e_dim=factor_e_dim)
            self._uses_factorized = True
        else:
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self._uses_factorized = False

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TinyGemmaBlock(d_model, n_heads, d_ff, attn_dropout=dropout, resid_dropout=dropout, rope_base=rope_base, max_position=max_position)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights and not self._uses_factorized:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        x = self.tok_emb(idx)  # [B, T, C]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, is_causal=True)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# -------------------------------
# Presets (~param targets with 64k vocab + factorized embeddings)
# -------------------------------
PRESETS = {
    "1p5M": dict(d_model=160, n_layers=8,  n_heads=4, d_ff=384, factorized_emb=True, factor_e_dim=64, tie_weights=False),
    "3p0M": dict(d_model=192, n_layers=10, n_heads=6, d_ff=512, factorized_emb=True, factor_e_dim=64, tie_weights=False),
    "5p5M": dict(d_model=224, n_layers=12, n_heads=7, d_ff=640, factorized_emb=True, factor_e_dim=80, tie_weights=False),
    "7p5M": dict(d_model=256, n_layers=12, n_heads=8, d_ff=768, factorized_emb=True, factor_e_dim=96, tie_weights=False),
    "9p8M": dict(d_model=288, n_layers=12, n_heads=8, d_ff=896, factorized_emb=True, factor_e_dim=96, tie_weights=False),
}

# -------------------------------
# Training utilities
# -------------------------------
@dataclass
class TrainConfig:
    spm_model_path: str
    splits_root: str
    run_name: str = "tiny-run"

    # HOME defaults (works on /home2/$USER)
    save_to: str = os.path.join(os.path.expanduser("~"), "LMA_SLM/checkpoints")
    token_budget_ledger: str = os.path.join(os.path.expanduser("~"), "LMA_SLM/training/token_budget.json")
    token_budget_total: int = 3_000_000_000
    disable_budget: bool = False

    model_preset: str = "5p5M"
    seq_len: int = 512
    global_batch_tokens: int = 16384   # ≈ micro_bsz * grad_accum * seq_len
    micro_bsz: int = 8
    grad_accum: int = 1                # if 0 → computed
    max_steps: int = 1000
    warmup_steps: int = 100
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0
    val_interval: int = 200
    ckpt_interval: int = 200
    eval_batches: int = 50
    max_docs_per_lang_train: Optional[int] = None
    max_docs_per_lang_val: Optional[int] = 400
    amp_dtype: str = "bf16"            # "bf16" | "fp16" | "fp32"
    resume: bool = True
    seed: int = 1337
    num_workers: int = 2
    pin_memory: bool = True

def cosine_lr(it, max_it, warmup, lr_max, lr_min):
    if it < warmup: return lr_max * (it + 1) / max(1, warmup)
    if it >= max_it: return lr_min
    ratio = (it - warmup) / max(1, (max_it - warmup))
    return lr_min + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr_max - lr_min)

def save_checkpoint(path: str, model: nn.Module, opt: torch.optim.Optimizer, step: int, tokens_seen: int, cfg: TrainConfig, logger: TeeLogger):
    ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "tokens_seen": tokens_seen, "cfg": asdict(cfg)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    logger.print(f"[ckpt] saved: {path}")

def load_latest_checkpoint(run_dir: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(run_dir, "ckpt_step_*.pt")))
    return paths[-1] if paths else None

def load_checkpoint(path: str, model: nn.Module, opt: torch.optim.Optimizer, logger: TeeLogger) -> Tuple[int, int, Optional[TrainConfig]]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    step = int(ckpt.get("step", 0))
    tokens_seen = int(ckpt.get("tokens_seen", 0))
    cfg_dict = ckpt.get("cfg", {})
    cfg = TrainConfig(**cfg_dict) if cfg_dict else None
    logger.print(f"[ckpt] loaded from {path} (step={step}, tokens_seen={tokens_seen})")
    return step, tokens_seen, cfg

def read_token_budget(ledger_path: str, default_total: int, disable: bool, logger: TeeLogger) -> Dict[str, int]:
    if disable:
        logger.print("[budget] disabled by flag.")
        return {"budget_total": default_total, "budget_used": 0}
    try:
        if not ledger_path:
            logger.print("[budget] no ledger path; disabled.")
            return {"budget_total": default_total, "budget_used": 0}
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
        if os.path.exists(ledger_path):
            with open(ledger_path, "r") as f:
                j = json.load(f)
            if not isinstance(j, dict): raise ValueError
            j.setdefault("budget_total", default_total)
            j.setdefault("budget_used", 0)
            return j
        j = {"budget_total": default_total, "budget_used": 0}
        with open(ledger_path, "w") as f:
            json.dump(j, f, indent=2)
        return j
    except Exception as e:
        logger.print(f"[budget] disabled (reason: {e})")
        return {"budget_total": default_total, "budget_used": 0}

def write_token_budget(ledger_path: str, j: Dict[str, int], logger: TeeLogger):
    try:
        if not ledger_path: return
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
        with open(ledger_path, "w") as f:
            json.dump(j, f, indent=2)
    except Exception as e:
        logger.print(f"[budget] write skipped: {e}")

# -------------------------------
# Training loop (single GPU)
# -------------------------------
def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    # ---- device / env ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Run directory + logger
    run_dir = os.path.join(cfg.save_to, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = TeeLogger(os.path.join(run_dir, "train.log"))
    logger.print("[config]", json.dumps(asdict(cfg), indent=2))

    logger.print(f"[env] torch={torch.__version__}  device={device}")
    if device == "cuda":
        logger.print(f"[env] GPU name: {torch.cuda.get_device_name(0)}  BF16 supported? {torch.cuda.is_bf16_supported()}")
        logger.print(f"[env] CUDA capability: {torch.cuda.get_device_capability(0)}")
        logger.print(f"[env] Total VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1e9,2)} GB")

    # ---- tokenizer ----
    tok = SPMTokenizer(cfg.spm_model_path, verbose=True)
    vocab_size = tok.vocab_size

    # ---- model preset ----
    assert cfg.model_preset in PRESETS, f"Unknown model_preset {cfg.model_preset}. Options: {list(PRESETS.keys())}"
    mp = PRESETS[cfg.model_preset]
    model = TinyGemmaLM(
        vocab_size=vocab_size,
        d_model=mp["d_model"], n_layers=mp["n_layers"], n_heads=mp["n_heads"], d_ff=mp["d_ff"],
        max_position=cfg.seq_len, dropout=cfg.dropout, rope_base=10000.0,
        factorized_emb=mp.get("factorized_emb", False), factor_e_dim=mp.get("factor_e_dim", 64),
        tie_weights=mp.get("tie_weights", False),
    )
    n_params = count_params(model)
    logger.print(f"[model] preset={cfg.model_preset}  params={n_params/1e6:.3f}M "
                 f"(d_model={mp['d_model']} layers={mp['n_layers']} heads={mp['n_heads']} ff={mp['d_ff']} "
                 f"factorized_emb={mp.get('factorized_emb', False)} e_dim={mp.get('factor_e_dim', None)})")

    # ---- AMP dtype ----
    amp_dtype = (torch.bfloat16 if (cfg.amp_dtype.lower()=="bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                 else torch.float16 if (cfg.amp_dtype.lower()=="fp16" and torch.cuda.is_available())
                 else torch.float32)
    logger.print(f"[amp] using dtype={amp_dtype}")
    model = model.to(device=device, dtype=torch.float32)  # keep params in FP32

    # ---- optimizer ----
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_max, betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.weight_decay)

    # ---- data ----
    train_files = list_split_files(cfg.splits_root, "train")
    val_files   = list_split_files(cfg.splits_root, "val")
    if not train_files:
        logger.print("[warn] No train files found. Using tiny dummy dataset for smoke test.")
        dummy_texts = ["<eng> hello world " * 50, "<eng> this is a tiny test " * 50, "<eng> नमस्ते दुनिया " * 50]
        class DummyDataset(IterableDataset):
            def __iter__(self):
                for t in dummy_texts * 100:
                    ids = tok.encode(t, add_bos=False, add_eos=True, lang=None)
                    if len(ids) >= (cfg.seq_len + 1):
                        for i in range(0, len(ids) - (cfg.seq_len + 1), cfg.seq_len + 1):
                            yield torch.tensor(ids[i : i + cfg.seq_len + 1], dtype=torch.long)
        train_ds = DummyDataset()
        val_ds   = DummyDataset()
    else:
        train_ds = PackedIterableDataset(
            files_with_lang=train_files, tokenizer=tok, seq_len=cfg.seq_len,
            max_docs_per_lang=cfg.max_docs_per_lang_train, add_bos=False, add_eos=True,
            shuffle_files_each_epoch=True, print_every_docs=2000
        )
        val_ds = PackedIterableDataset(
            files_with_lang=(val_files if val_files else train_files), tokenizer=tok, seq_len=cfg.seq_len,
            max_docs_per_lang=cfg.max_docs_per_lang_val, add_bos=False, add_eos=True,
            shuffle_files_each_epoch=False, print_every_docs=0
        )

    # Compute grad_accum to match global batch tokens if not set
    if cfg.grad_accum <= 0:
        cfg.grad_accum = max(1, cfg.global_batch_tokens // (cfg.micro_bsz * cfg.seq_len))
    eff_tokens_per_step = cfg.micro_bsz * cfg.grad_accum * cfg.seq_len
    logger.print(f"[train] seq_len={cfg.seq_len} micro_bsz={cfg.micro_bsz} grad_accum={cfg.grad_accum} "
                 f"-> effective tokens/step={eff_tokens_per_step}")

    # DataLoaders (tweak if you see hangs on CentOS7)
    train_loader = DataLoader(train_ds, batch_size=cfg.micro_bsz, pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.micro_bsz, pin_memory=cfg.pin_memory, num_workers=max(0, min(2, cfg.num_workers)))

    # ---- token budget ledger ----
    ledger = read_token_budget(cfg.token_budget_ledger, cfg.token_budget_total, cfg.disable_budget, logger)
    logger.print(f"[budget] total={ledger['budget_total']:,} used={ledger['budget_used']:,} "
                 f"remaining={ledger['budget_total']-ledger['budget_used']:,}")

    # ---- resume ----
    latest = load_latest_checkpoint(run_dir) if cfg.resume else None
    start_step = 0
    tokens_seen = 0
    if latest:
        try:
            start_step, tokens_seen, _ = load_checkpoint(latest, model, opt, logger)
        except Exception as e:
            logger.print("[warn] Failed to load checkpoint; starting fresh.", e)

    # Save config snapshot
    with open(os.path.join(run_dir, "train_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype==torch.float16))
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    # Graceful save on interrupt
    def _handle_sigterm(signum, frame):
        path = os.path.join(run_dir, f"ckpt_step_{start_step}.pt")
        try:
            save_checkpoint(path, model, opt, start_step, tokens_seen, cfg, logger)
        finally:
            logger.print("[signal] SIGTERM received. Checkpointed and exiting.")
            raise SystemExit(0)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    # ----------------------------------
    # Training loop
    # ----------------------------------
    model.train()
    t0 = time.time()
    step = start_step
    running_loss = 0.0

    train_iter = iter(train_loader)
    def next_batch(loader, it):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        return b, it

    while step < cfg.max_steps:
        model.train()
        lr = cosine_lr(step, cfg.max_steps, cfg.warmup_steps, cfg.lr_max, cfg.lr_min)
        for g in opt.param_groups: g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        micro_losses = []
        tokens_this_step = 0
        step_start = time.time()

        for micro in range(cfg.grad_accum):
            batch, train_iter = next_batch(train_loader, train_iter)  # [B, T+1]
            x = batch[:, :-1].to(device, non_blocking=True)
            y = batch[:, 1: ].to(device, non_blocking=True)
            tokens_this_step += x.numel()

            with torch.cuda.amp.autocast(enabled=(amp_dtype in (torch.bfloat16, torch.float16)), dtype=(amp_dtype if amp_dtype!=torch.float32 else None)):
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / cfg.grad_accum

            scaler.scale(loss).backward()
            micro_losses.append(loss.detach().float().item())

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt); scaler.update()

        step_time = time.time() - step_start
        tokens_seen += tokens_this_step
        running_loss = 0.98*running_loss + 0.02*(sum(micro_losses)) if step>0 else sum(micro_losses)

        tps = tokens_this_step / max(1e-6, step_time)
        est_tflops = (6.0 * count_params(model) * tps) / 1e12  # rough

        if (step % 10) == 0:
            logger.print(f"[step {step:05d}] lr={lr:.2e} loss={running_loss:.4f} "
                         f"gn={float(gn):.3f} tokens/s={int(tps):,} est_TFLOPs={est_tflops:.2f} "
                         f"tokens_seen={tokens_seen:,}")

        # Validation
        if (step % cfg.val_interval == 0 and step > 0) or (step == cfg.max_steps - 1):
            model.eval()
            with torch.no_grad():
                valloss = []
                itv = iter(val_loader)
                for _ in range(cfg.eval_batches):
                    try: vb = next(itv)
                    except StopIteration: break
                    vx = vb[:, :-1].to(device); vy = vb[:, 1: ].to(device)
                    with torch.cuda.amp.autocast(enabled=(amp_dtype in (torch.bfloat16, torch.float16)), dtype=(amp_dtype if amp_dtype!=torch.float32 else None)):
                        vout = model(vx)
                        vloss = loss_fn(vout.reshape(-1, vout.size(-1)), vy.reshape(-1))
                    valloss.append(float(vloss))
                if valloss:
                    vmean = sum(valloss)/len(valloss)
                    ppl = math.exp(min(20.0, vmean))
                    logger.print(f"[eval ] step={step} val_loss={vmean:.4f} ppl≈{ppl:.2f}")

        # Checkpoint
        if (step % cfg.ckpt_interval == 0 and step > start_step) or (step == cfg.max_steps - 1):
            ckpt_path = os.path.join(run_dir, f"ckpt_step_{step}.pt")
            save_checkpoint(ckpt_path, model, opt, step, tokens_seen, cfg, logger)

        # Budget update
        ledger["budget_used"] += tokens_this_step
        write_token_budget(cfg.token_budget_ledger if not cfg.disable_budget else "", ledger, logger)
        remaining = ledger["budget_total"] - ledger["budget_used"]
        if remaining <= 0:
            logger.print(f"[budget] WARNING: Token budget exhausted (used={ledger['budget_used']:,} / total={ledger['budget_total']:,}). Stopping.")
            break
        elif remaining < 5_000_000:
            logger.print(f"[budget] Heads-up: Only {remaining:,} tokens remaining in your {ledger['budget_total']:,} budget.")

        step += 1

    elapsed = time.time() - t0
    logger.print(f"[done] steps={step} tokens_seen={tokens_seen:,} time={elapsed/60:.1f} min "
                 f"avg tokens/sec≈{int(tokens_seen/max(1.0, elapsed)):,}")
    logger.close()

# -------------------------------
# CLI
# -------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Tiny Gemma-Style LM Pretrainer (1–10M params) w/ SPM tokenizer (HPC-friendly)")
    p.add_argument("--spm_model_path", type=str, required=True, help="Path to SentencePiece .model")
    p.add_argument("--splits_root", type=str, required=True, help="Root dir with {eng,hin,nep}/{train,val}.jsonl")
    p.add_argument("--run_name", type=str, default="tiny-run")

    p.add_argument("--save_to", type=str, default=os.path.join(os.path.expanduser("~"), "LMA_SLM/checkpoints"))
    p.add_argument("--token_budget_ledger", type=str, default=os.path.join(os.path.expanduser("~"), "LMA_SLM/training/token_budget.json"))
    p.add_argument("--token_budget_total", type=int, default=3_000_000_000)
    p.add_argument("--disable_budget", action="store_true")

    p.add_argument("--model_preset", type=str, default="5p5M", choices=list(PRESETS.keys()))
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--global_batch_tokens", type=int, default=16384)
    p.add_argument("--micro_bsz", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=0, help="If 0, computed as global_batch_tokens/(micro_bsz*seq_len)")

    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--val_interval", type=int, default=200)
    p.add_argument("--ckpt_interval", type=int, default=200)
    p.add_argument("--eval_batches", type=int, default=50)

    p.add_argument("--max_docs_per_lang_train", type=int, default=None)
    p.add_argument("--max_docs_per_lang_val", type=int, default=400)

    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--no-pin_memory", dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=True)
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        spm_model_path=args.spm_model_path,
        splits_root=args.splits_root,
        run_name=args.run_name,

        save_to=args.save_to,
        token_budget_ledger=args.token_budget_ledger,
        token_budget_total=args.token_budget_total,
        disable_budget=args.disable_budget,

        model_preset=args.model_preset,
        seq_len=args.seq_len,
        global_batch_tokens=args.global_batch_tokens,
        micro_bsz=args.micro_bsz,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        val_interval=args.val_interval,
        ckpt_interval=args.ckpt_interval,
        eval_batches=args.eval_batches,
        max_docs_per_lang_train=args.max_docs_per_lang_train,
        max_docs_per_lang_val=args.max_docs_per_lang_val,
        amp_dtype=args.amp_dtype,
        resume=args.resume,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    train(cfg)
