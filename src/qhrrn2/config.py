# Ledger: hyperparameters for C1–C14; the capacity dial is `d` (params ∝ d²).
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    d: int = 12          # per-field feature width (operating point 16; toy 12)
    d_b: int = 6         # boundary-stream width per field (C5)
    d_a: int = 6         # attention message width per field (C14, Amendment D)
    d_ir: int = 32       # IR summary width
    d_code: int = 32     # rule-token width (C8)
    K: int = 64          # rule codebook size (spec default 128; toy 64)
    M: int = 2           # rule slots
    T: int = 4           # recursion iterations (C9)
    scales: int = 5      # 32 -> 16 -> 8 -> 4 -> 2 -> 1
    canvas: int = 32
    attn_max_hw: int = 32  # attention active when grid side <= this; 32 = all scales
    #                        (C14, Amendment D); 8 = coarse-only (Amendment-B ablation);
    #                        0 = absent (ablation)
    d_task: int = 32     # per-task program embedding width (C16); the task vector
    #                      is OPTIONAL at forward time — None reproduces the
    #                      pre-C16 model exactly
    beta_flux: float = 0.0     # H-4 dial; 0 during Phase-1 trainability gates
    beta_flux_nl: float = 0.0  # C14 wormhole toll on attention flux A_s; 0 in Phase 1
    use_obj: bool = False      # C17 cluster-update layers (ledger 2026-08-02);
    #                            False = exact pre-C17 graph, zero overhead
    beta_flux_obj: float = 0.0  # price on the I_object channel (logged free in v1)
    w_void: float = 0.1      # weight of VOID-region CE relative to canvas region (C1)
    lambda_size: float = 0.3  # canvas-head loss weight
