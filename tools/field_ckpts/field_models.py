"""Unified loaders for the public HRM / TRM(alphaXiv) / EqR Sudoku-Extreme checkpoints (2026-09-05).
One process per family (their top-level packages collide: HRM and EqR both use `models/`).
    m = load("hrm" | "trm" | "eqr", device, dtype)     -> FieldModel
    m.tokens(puzzles9)                                  -> their input tokens (blank 1, digits 2..10) with the prefix ids
    st = m.init_state(B, mode="fixed"|"random", std=1.0, gen=None)
    st, logits9, q_halt = m.step(st, inputs)            -> ONE outer step (their inner forward); logits9 (B,81,9) over digits 1..9
Token convention: puzzles9 are OUR int arrays (blank 0, digits 1..9)."""
from __future__ import annotations
import json, os, sys, math
from pathlib import Path
import numpy as np
import torch

FC = Path(__file__).resolve().parents[1]           # runs/field_ckpts
SRC = FC / "src"; CK = FC / "ckpts"

def _sys_path(*roots):
    for r in roots:
        p = str(r)
        if p not in sys.path: sys.path.insert(0, p)

class FieldModel:
    def __init__(self, tag, inner, cfg, device, dtype, prefix_len, has_noise=False):
        self.tag, self.inner, self.cfg, self.device, self.dtype, self.P = tag, inner, cfg, device, dtype, prefix_len
        self.has_noise = has_noise
        self.hid = cfg["hidden_size"]; self.seq = 81 + self.P
    # ---- tokens ----
    def tokens(self, puz9):
        x = torch.as_tensor(np.asarray(puz9).reshape(-1, 81) + 1, dtype=torch.int32, device=self.device)   # theirs = ours + 1
        pid = torch.zeros(x.shape[0], dtype=torch.int32, device=self.device)
        return {"inputs": x, "puzzle_identifiers": pid}
    # ---- state ----
    def init_state(self, B, mode="fixed", std=1.0, gen=None):
        shape = (B, self.seq, self.hid)
        if mode == "fixed":
            zH = self.H_init.to(self.dtype).expand(shape).clone(); zL = self.L_init.to(self.dtype).expand(shape).clone()
        elif mode == "random":       # N(0, std) (our RI draw; EqR's own reset is trunc-normal, mode="trunc")
            zH = torch.randn(shape, generator=gen, device="cpu").to(self.device, self.dtype) * std
            zL = torch.randn(shape, generator=gen, device="cpu").to(self.device, self.dtype) * std
        elif mode == "trunc":        # EXACT replica of HRM/EqR trunc_normal_init_(std): inverse-erf sampling on (erf(-2/sqrt2), erf(2/sqrt2)), scaled by comp_std, clipped
            import math
            a_, b_ = math.erf(-2 / math.sqrt(2)), math.erf(2 / math.sqrt(2)); z_ = (b_ - a_) / 2; c_ = (2 * math.pi) ** -0.5
            pdf = c_ * math.exp(-2.0); comp = std / math.sqrt(1 - (2 * pdf + 2 * pdf) / z_ - 0.0)
            def draw():
                u = torch.rand(shape, generator=gen, device="cpu") * (b_ - a_) + a_
                return (torch.erfinv(u) * math.sqrt(2) * comp).clamp_(-2 * comp, 2 * comp).to(self.device, self.dtype)
            zH, zL = draw(), draw()
        else: raise ValueError(mode)
        return (zH, zL)
    def solution_state(self, sol9):
        """z_H := their embedding of the SOLUTION tokens (sqrt(hid)-scaled, prefix from the trained prefix), z_L := L_init."""
        x = torch.as_tensor(np.asarray(sol9).reshape(-1, 81) + 1, dtype=torch.int32, device=self.device)
        pid = torch.zeros(x.shape[0], dtype=torch.int32, device=self.device)
        emb = self.embed(x, pid)                       # (B, seq, hid) already scaled
        zL = self.L_init.to(self.dtype).expand(emb.shape).clone()
        return (emb.to(self.dtype).clone(), zL)
    # ---- readout helpers ----
    def logits9(self, logits_full):
        return logits_full[..., 2:11].float()          # digits 1..9 = their classes 2..10
    def full_argmax_is_digit(self, logits_full):
        return (logits_full.argmax(-1) >= 2)           # their metric argmax over all 11 classes; PAD/blank predictions count as wrong

# ---------------- HRM ----------------
def load_hrm(device, dtype):
    _sys_path(FC / "shims", SRC / "HRM")
    import yaml
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1_Inner, HierarchicalReasoningModel_ACTV1Config
    cfg = yaml.safe_load(open(CK / "hrm_sudoku" / "all_config.yaml"))["arch"]
    cfg = {k: v for k, v in cfg.items() if k not in ("name", "loss")}
    cfg.update(batch_size=1, seq_len=81, vocab_size=11, num_puzzle_identifiers=1, forward_dtype={torch.float32: "float32", torch.bfloat16: "bfloat16"}[dtype])
    inner = HierarchicalReasoningModel_ACTV1_Inner(HierarchicalReasoningModel_ACTV1Config(**cfg))
    sd = torch.load(CK / "hrm_sudoku" / "checkpoint", map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.model.inner.", ""): v for k, v in sd.items()}
    missing, unexpected = inner.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(m.startswith("puzzle_emb.local") for m in missing), missing
    inner.to(device).eval()
    m = FieldModel("hrm", inner, cfg, device, dtype, prefix_len=inner.puzzle_emb_len)
    m.H_init, m.L_init = inner.H_init.detach(), inner.L_init.detach()
    m.embed = lambda x, pid: inner._input_embeddings(x, pid)
    seq_info = dict(cos_sin=inner.rotary_emb())
    def step(st, batch):
        zH, zL = st
        emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        with torch.no_grad():
            for h in range(cfg["H_cycles"]):
                for l in range(cfg["L_cycles"]):
                    zL = inner.L_level(zL, zH + emb, **seq_info)
                zH = inner.H_level(zH, zL, **seq_info)
            out = inner.lm_head(zH)[:, inner.puzzle_emb_len:]
            q = inner.q_head(zH[:, 0]).float()
        return (zH, zL), out, q[..., 0]
    m.step = step
    return m

# ---------------- TRM (alphaXiv) ----------------
def load_trm(device, dtype, which="step_32550_sudoku_epoch50k", path=None):
    _sys_path(FC / "shims", SRC / "TRM_alphaxiv" / "src")
    from trm.models.architectures.trm import TinyRecursiveReasoningModel_ACTV1_Inner, TinyRecursiveReasoningModel_ACTV1Config
    cfg = dict(batch_size=1, seq_len=81, puzzle_emb_ndim=512, num_puzzle_identifiers=1, vocab_size=11, H_cycles=3, L_cycles=6, H_layers=0, L_layers=2,
               hidden_size=512, expansion=4, num_heads=8, pos_encodings="none", halt_max_steps=16, halt_exploration_prob=0.1,
               forward_dtype={torch.float32: "float32", torch.bfloat16: "bfloat16"}[dtype], mlp_t=True, puzzle_emb_len=16, no_ACT_continue=True)
    inner = TinyRecursiveReasoningModel_ACTV1_Inner(TinyRecursiveReasoningModel_ACTV1Config(**cfg))
    sd = torch.load(path or (CK / "trm_alphaxiv" / which), map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", "").replace("model.inner.", ""): v for k, v in sd.items()}
    missing, unexpected = inner.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(m.startswith("puzzle_emb.local") for m in missing), missing
    inner.to(device).eval()
    m = FieldModel("trmc" if path else "trm", inner, cfg, device, dtype, prefix_len=16)
    m.H_init, m.L_init = inner.H_init.detach(), inner.L_init.detach()
    m.embed = lambda x, pid: inner._input_embeddings(x, pid)
    seq_info = dict(cos_sin=None)
    def step(st, batch):
        zH, zL = st
        emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        with torch.no_grad():
            for h in range(cfg["H_cycles"]):
                for l in range(cfg["L_cycles"]):
                    zL = inner.L_level(zL, zH + emb, **seq_info)
                zH = inner.L_level(zH, zL, **seq_info)
            out = inner.lm_head(zH)[:, 16:]
            q = inner.q_head(zH[:, 0]).float()
        return (zH, zL), out, q[..., 0]
    m.step = step
    return m

# ---------------- EqR ----------------
def load_eqr(device, dtype, use_ema=True, noise_scale=None):
    _sys_path(FC / "shims", SRC / "EqR")
    from models.eqr import InnerNetwork, EqRConfig
    j = json.load(open(CK / "eqr" / "sudoku-extreme" / "eqr_extracted" / "eqr_config.json"))
    arch = {k: v for k, v in j["config"]["arch"].items() if k not in ("name", "short_name", "loss", "no_ACT_continue")}
    cfg = dict(arch); cfg.update(batch_size=1, seq_len=81 + 16, input_seq_len=81, num_puzzle_identifiers=1, vocab_size=11,
                                 forward_dtype={torch.float32: "float32", torch.bfloat16: "bfloat16"}[dtype])
    cfg.setdefault("lambda_", 0.95); cfg.setdefault("noise_scale", 0.01); cfg.setdefault("H_init_std", 1.0); cfg.setdefault("L_init_std", 1.0)
    inner = InnerNetwork(EqRConfig(**cfg))
    w = np.load(CK / "eqr" / "sudoku-extreme" / "eqr_extracted" / "eqr_weights.npz")
    pre = "ema/" if use_ema else "model/"
    sd = {k[len(pre):].replace("_orig_mod.model.inner.", ""): torch.from_numpy(w[k]) for k in w.files if k.startswith(pre)}
    if use_ema and "puzzle_emb.weights" not in sd:      # the EMA shadow carries no puzzle embedding (a buffer): take the model's
        sd["puzzle_emb.weights"] = torch.from_numpy(w["model/_orig_mod.model.inner.puzzle_emb.weights"])
    missing, unexpected = inner.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(m.startswith("puzzle_emb.local") or m in ("H_init", "L_init") for m in missing), missing
    inner.to(device).eval()
    for mod in inner.modules():
        if hasattr(mod, "noise_scale") and noise_scale is not None: mod.noise_scale = float(noise_scale)
    m = FieldModel("eqr", inner, cfg, device, dtype, prefix_len=16, has_noise=True)
    m.H_init, m.L_init = inner.H_init.detach(), inner.L_init.detach()      # random (non-persistent) buffers: EqR has no cold start
    m.embed = lambda x, pid: inner._input_embeddings(x, pid)
    def step(st, batch):
        zH, zL = st
        with torch.no_grad():
            x = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"]); seq = {"cos_sin": inner._cos_sin()}
            for h in range(cfg["H_cycles"]):
                zH, zL = inner.latent_recursion(zH, zL, x, seq)
            out = inner.lm_head(zH[:, 16:]); q = inner.q_head(zH[:, 0]).float()
        return (zH, zL), out, q[..., 0]
    m.step = step
    m.set_noise = lambda s: [setattr(mod, "noise_scale", float(s)) for mod in inner.modules() if hasattr(mod, "noise_scale")]
    return m

def load(tag, device="cpu", dtype=torch.float32, **kw):
    if tag == "trmc": return load_trm(torch.device(device), dtype, path=CK / "trm_cgar" / "pytorch_model.bin", **kw)
    return {"hrm": load_hrm, "trm": load_trm, "eqr": load_eqr}[tag](torch.device(device), dtype, **kw)

def count_params(m):
    return sum(p.numel() for p in m.inner.parameters())
