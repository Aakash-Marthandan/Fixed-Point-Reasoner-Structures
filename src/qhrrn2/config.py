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
    remat: bool = False        # gradient-checkpoint each recursion step
    #                            (~30% compute for ~T-fold activation memory;
    #                            unlocks B=64 at d>=24+C17 — 2026-08-05)
    w_void: float = 0.1      # weight of VOID-region CE relative to canvas region (C1)
    lambda_size: float = 0.3  # canvas-head loss weight
    # E10 equilibrium core (ledger 2026-08-09; [H-2'], FPRM-informed).
    # equilibrium=False reproduces the pre-E10 graph exactly.
    equilibrium: bool = False
    t_max: int = 16          # inference iteration ceiling (probes/halt)
    res_tau: float = 0.05    # relative-residual halt threshold
    # pretrain-9 dials (ledger 2026-08-10; defaults reproduce pretrain-8
    # exactly). eta_floor = the annealing/temperature dial ([H-5] reading:
    # quasi-static was LEARNED, not chosen — the floor keeps flow finite);
    # z_gate_init warm-opens the bulk scratchpad ([H-24]; 0-init never
    # engaged under anchor training).
    eta_floor: float = 0.0
    z_gate_init: float = 0.0
    # pretrain-13 dials (ledger 2026-08-12, EqR/FPRM deep-read; defaults
    # reproduce pretrain-12 exactly — each mechanism is branch-inert at its
    # default and carries its named test in tests/test_p13.py).
    eq_coupled: bool = False   # FPRM coupled residual scaling: y <- a1*y + a2*p,
    #                            a1/a2 learnable, init contractive .75/.25
    #                            (their Thm-1 recipe); False = the damped
    #                            y + eta*(p-y) update, untouched
    ni_sigma: float = 0.0      # EqR per-step TRAINING noise (state-space,
    #                            simplex-tangent, canonical scale); active only
    #                            when an rng is threaded (training) — inference
    #                            probes (rng=None) never see it
    flux_floors: str = ""      # B1-full free-bits floors: comma nats per scale
    #                            ("350,75,50,15,30"); priced term becomes
    #                            beta * sum_s relu(I_s - F_s); "" = global toll.
    #                            STRING, not tuple: ckpt config values must
    #                            survive as python scalars (2026-08-01 law,
    #                            tests/test_episodic.py) — parsed at trace time
    # SPRINT S2 wave 2 (ledger 2026-08-22): Sudoku canvas layout — "origin"
    # (every prior arm) or "box4" (the registered box-aligned control arm:
    # each 3x3 box in a 4x4 block). Carried in the ckpt so the batched
    # evaluator places/unplaces consistently with training; the model graph
    # does not read it (no effect on ARC or on any old checkpoint).
    # CHAMPION TRACK (2026-09-01): "native9" = the 3-adic native geometry —
    # canvas IS the 9x9 grid (no padding), pool_arity 3 (9 -> 3 -> 1,
    # box-aligned: each level-1 pooling block IS a Sudoku box), scales 2,
    # mixer_kind "group9".
    sudoku_layout: str = "origin"
    # CHAMPION TRACK geometry dials (2026-09-01, Plan_2026-09-01_Champion_Track
    # §2). Defaults reproduce every prior checkpoint/graph bit-exactly; the
    # native9 configuration sets canvas=9, scales=2, pool_arity=3,
    # mixer_kind="group9", attn_max_hw=9.
    pool_arity: int = 2        # RG branching factor (2 = dyadic, 3 = 3-adic)
    mixer_kind: str = "seam"   # "seam" = offset-block seam mixer (C4);
    #                            "group9" = the factorized all-different GROUP
    #                            mixer: ONE shared operator applied to every
    #                            constraint group (9 rows + 9 cols + 9 boxes at
    #                            s0; the box grid as one group at s1), with
    #                            per-partition-type and per-slot embeddings.
    #                            Sudoku's 27 constraints as the operator basis.
    # SPRINT S2 wave 3a (ledger 2026-08-23, H-45 contractivity collapse): FIXED-POINT
    # ANCHOR rows — per training pair, additionally roll the FINAL map (t_norm=1)
    # for fpa_k steps from a lightly corrupted solution (eps ~ U[0, fpa_eps] of the
    # true-extent cells resampled) and deep-supervise those steps, weight fpa_w.
    # Trains local contraction of the final map at the solution (the probe's
    # ladder instrument as a loss). fpa_k=0 = off, bit-exact pre-existing graph.
    fpa_k: int = 0
    fpa_eps: float = 0.2
    fpa_w: float = 1.0
    # ── sportC1 (2026-09-02; Plan_2026-09-02_Champion_sportC1 §11–§12) ──
    # z-NORM (H-50's mechanism, the z-channel trace): normalize the carried
    # latent at its ONE entry point (model.forward_fields). "" = off, the
    # exact pre-existing graph; "rms" = z <- z + alpha_z * RMSNorm_d(z_in) * g
    # with g = params["eq"]["z_gain"] ((d,), init ones) — per site over the
    # feature axis, shared over fields (S9-safe). Every caller (iterate_eq,
    # the evaluator's run_batch, the monitor, the census) inherits it.
    z_norm: str = ""
    # sportC2 grafts (2026-09-04; Freethink 2026-09-03 X-4/X-7): inner_k = latent passes per outer
    # step before the readout update (R2; 1 = bit-exact); hard_p = probability per outer step,
    # TRAINING only (rng threaded), that the feedback uses the HARD argmax readout with a
    # straight-through gradient (R3; 0 = bit-exact); trm_token_mixer "group9" = our factorized
    # group mixer as the field cell's token mixer on a trm_gm_dim projection (X2; "mlp" = TRM exact).
    inner_k: int = 1
    hard_p: float = 0.0
    trm_token_mixer: str = "mlp"
    trm_gm_dim: int = 64
    # Fixed equilibrium dampings (0 = learned, bit-exact). The TRM/EqR cell
    # carries its latent undamped across segments (eta_z_fixed 1) and reads y
    # out (eta_fixed 1); EqR's damping lambda lives INSIDE that cell.
    eta_fixed: float = 0.0
    eta_z_fixed: float = 0.0
    # X0 — the FIELD-RECIPE cell. "rg" = the QHRRN RG cell (every prior ckpt);
    # "trm" = the TRM/EqR block stack (qhrrn2.trm_cell; §11.1/§11.2): 81 tokens
    # + a trm_puzzle_emb_len prefix, trm_layers POST-norm blocks (token-mixing
    # SwiGLU + channel SwiGLU, parameter-free RMSNorm), H_cycles x (L_cycles+1)
    # stack passes per segment with the gradient through the last H-cycle only,
    # states z_H / z_L carried as z_fine of shape (2, S, hidden).
    cell_kind: str = "rg"
    trm_hidden: int = 512
    trm_layers: int = 2
    trm_h_cycles: int = 3
    trm_l_cycles: int = 6
    trm_expansion: float = 4.0
    trm_puzzle_emb_len: int = 16
    trm_lambda: float = 0.0      # EqR Eq. 2 damping per inner pass (0 = TRM exact; EqR .05)
    trm_beta: float = 0.0        # EqR path noise per inner pass, training only (EqR .01)
    trm_ri_sigma: float = 0.0    # EqR RI: z0 ~ N(0, sigma) when an rng is threaded (A.3 default 1)
    loss_kind: str = "softmax"   # "stablemax" = HRM/TRM's stablemax cross-entropy (X0, labeled)
