# Ledger: hyperparameters for C1–C13; the capacity dial is `d` (params ∝ d²).
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    d: int = 12          # per-field feature width (operating point 16; toy 12)
    d_b: int = 6         # boundary-stream width per field (C5)
    d_ir: int = 32       # IR summary width
    d_code: int = 32     # rule-token width (C8)
    K: int = 64          # rule codebook size (spec default 128; toy 64)
    M: int = 2           # rule slots
    T: int = 4           # recursion iterations (C9)
    scales: int = 5      # 32 -> 16 -> 8 -> 4 -> 2 -> 1
    canvas: int = 32
    attn_max_hw: int = 8  # coarse attention active when grid side <= this (C6, Amendment B)
    beta_flux: float = 0.0   # H-4 dial; 0 during Phase-1 trainability gates
    w_void: float = 0.1      # weight of VOID-region CE relative to canvas region (C1)
    lambda_size: float = 0.3  # canvas-head loss weight
