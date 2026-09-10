"""G4 (Plan_2026-09-10_DEC-ARC_Build §3): the MAC instrument. MEASURED = XLA's cost analysis (flops / 2) of the evaluator's
own jitted one-step map on ONE example (batch 1) — the DEC / TRM cells through eval_sudoku_extreme._step, the native rg cell
through probe_e1e3._traced_fwd_eq — so the count is the compiled graph's, not a formula. ANALYTIC = the per-token formulas
of the plan (§1.7) for the field's cells and for the DEC-ARC design at candidate widths, calibrated against the measured
Sudoku DEC. Descriptive; no rules.   .venv/bin/python tools/mac_count.py [--ckpt ... --label ...]*"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np, jax, jax.numpy as jnp
from qhrrn2.config import Config
from qhrrn2 import model as M, grid as G, sudoku as SUD
import eval_sudoku_extreme as EV
import probe_e1e3 as P


def load(path):
    import pickle
    with open(path, "rb") as f: saved = pickle.load(f)
    d = Config()
    cfg = Config(**{k: type(getattr(d, k))(v) for k, v in saved["config"].items() if hasattr(d, k)})
    return cfg, saved["state"]["model"], jnp.asarray(saved["state"]["table"][0])


def n_params(p):
    return int(sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(p)))


def flops_of(lowered):
    c = lowered.compile().cost_analysis()
    c = c[0] if isinstance(c, (list, tuple)) else c
    return float(c["flops"]) if c and "flops" in c else float("nan")


def measure_sudoku(cfg, params, tvj):
    layout = getattr(cfg, "sudoku_layout", "origin") or "origin"
    x_can = EV.place_batch(np.zeros((1, 9, 9), np.int32), layout)
    cv = SUD.layout_canvas(layout)
    void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    y0 = jnp.broadcast_to(void, (1,) + void.shape)
    first = EV._step(cfg, 1.0, 1.0, True); _, z = first(params, x_can, y0, tvj, jnp.zeros(1))
    step = EV._step(cfg, 1.0, 1.0, False)
    return flops_of(step.lower(params, x_can, y0, tvj, z)) / 2.0


def measure_arc(cfg, params, tvj):
    x_can = jnp.zeros((G.CANVAS, G.CANVAS), jnp.int32)
    y = jax.nn.one_hot(jnp.full((G.CANVAS, G.CANVAS), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    fwd = P._traced_fwd_eq(cfg, 1.0, 1.0)
    out = fwd(params, x_can, y, tvj, jnp.zeros(1))
    return flops_of(fwd.lower(params, x_can, y, tvj, out.z_fine)) / 2.0


def analytic_dec_arc(w, S=900, F=10, layers=2, h=3, l=6, expansion=4, heads=4):
    """MACs per example per outer segment for the DEC-ARC design: per token per block: attention 4w^2 + 2*S*w, channel SwiGLU
    3*w*(expansion*w), field coupling ~2w^2; tokens F*S; block passes per segment = layers * (h * l) (the L-cycle body) + layers * h (the H updates)."""
    per_tok = 4 * w * w + 2 * S * w + 3 * w * (expansion * w) + 2 * w * w
    passes = layers * (h * l + h)
    return per_tok * F * S * passes, 2 * layers * (4 * w * w + 3 * w * expansion * w + 2 * w * w) * 0 + layers * (4 * w * w + 3 * w * expansion * w + 2 * w * w) * 1  # (MACs per segment, params per block stack approx)


def analytic_field_arc(w=512, S=916, layers=2, h=3, l=4, expansion=4):
    per_tok = 4 * w * w + 2 * S * w + 3 * w * (expansion * w)
    return per_tok * S * layers * (h * l + h)


def analytic_field_sudoku_mlp(w=512, S=81, layers=2, h=3, l=6, expansion=4):
    """TRM-MLP: token-mixing MLP over S tokens (S*expansion*S per channel... the mlp_t is (S -> eS -> S) per channel: 2*e*S*S*w per pass) + channel SwiGLU."""
    per_pass = 2 * expansion * S * S * w + S * 3 * w * expansion * w
    return per_pass * layers * (h * l + h)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", action="append", default=[]); ap.add_argument("--label", action="append", default=[])
    a = ap.parse_args()
    rows = []
    for path, lab in zip(a.ckpt, a.label):
        cfg, params, tvj = load(path); kind = getattr(cfg, "cell_kind", "rg")
        try:
            macs = measure_sudoku(cfg, params, tvj) if kind in ("trm", "dec") else measure_arc(cfg, params, tvj)
        except Exception as e:
            macs = float("nan"); print(f"  {lab}: measure failed: {type(e).__name__}: {str(e)[:160]}")
        rows.append((lab, kind, n_params(params), macs))
        print(f"MEASURED {lab:28s} cell {kind:4s} params {n_params(params):>10,d}  MACs per outer step per example {macs/1e9:8.2f} G")
    print("ANALYTIC (per example; the field's cells and the DEC-ARC design; labeled inferred)")
    print(f"  TRM-MLP Sudoku (w512, 81 tokens, L2 H3 L6): {analytic_field_sudoku_mlp()/1e9:7.2f} GMAC per segment")
    print(f"  TRM-Attention ARC-1 (w512, 916 tokens, L2 H3 L4): {analytic_field_arc()/1e9:7.2f} GMAC per segment ({analytic_field_arc()*16/1e12:.2f} TMAC per 16-segment rollout)")
    for w in (128, 160, 192, 256):
        m, _ = analytic_dec_arc(w)
        print(f"  DEC-ARC w{w:<4d} (10 fields x 900 cells, L2 H3 L6): {m/1e9:7.2f} GMAC per segment = {m/analytic_field_arc():.2f} x the field's ARC cell; {m*16/1e12:.2f} TMAC per 16-segment rollout")
    out = ROOT / "runs/analysis/mac_count_20260910.json"
    prev = json.load(open(out))["measured"] if out.exists() else []
    merged = {r["label"]: r for r in prev}
    for l, k, n, m in rows: merged[l] = {"label": l, "cell": k, "params": n, "macs_per_step": m, "method": "XLA cost_analysis of the evaluator's jitted one-step map, batch 1, CPU backend; MACs = flops / 2"}
    json.dump({"measured": list(merged.values())}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
