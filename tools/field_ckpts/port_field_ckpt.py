#!/usr/bin/env python3
"""BUILD B2 (Plan_2026-09-07_Instrument_Suite §4.1 / §6; 2026-09-07): the public TRM-class Sudoku checkpoints ported into OUR
checkpoint format so `tools/eval_sudoku_extreme.py` (and every standing tool) reads them through the identical code path as our
arms. The conversion is `verify_port.py`'s (I12 / P14: fp32 step-1 max |Δlogit| 1.7e-3 on logits of scale 19.5): token rows
shifted by one (their PAD 0 -> our VOID 10; their 1..10 -> our 0..9), the puzzle prefix = their one 512-vector on prefix row 0
(zeros on rows 1..15, as their F.pad), the heads transposed, and THEIR H_init / L_init buffers carried as params["H_init"] /
params["L_init"] (trm_cell.init_states reads them when present). EqR's buffers are non-persistent random draws in their code
(no cold start exists): the port stores ONE trunc-normal(std 1) draw at seed 20260907 as the labeled "cold" start — the B = 1
column of the paper is a per-puzzle random reset (the evaluator's --z0-mode trunc draws).

  runs/field_ckpts/venv/bin/python tools/field_ckpts/port_field_ckpt.py [--models trmpub,trmpub60k,cgar,eqr] [--n 64] [--D 16]
      -> runs/field_ckpts/ported/<tag>/ckpt_latest.pkl + port_check.json (torch fp32 vs our JAX fp32 on n strat puzzles)

Config written into the checkpoint = X0's field-recipe config (cell_kind trm, hidden 512, layers 2, H3 x L6, prefix 16,
native9, T 16, eta pinned) with the model's own λ / β: TRM (alphaXiv, CGAR) λ = β = 0; EqR λ = .05 (their λ_ = .95 on F),
β = .01 (their training noise; the evaluator applies per-pass noise ONLY under --seg-noise-beta, so the default rows are the
noise-0 = B = 1 protocol). `state` = the released weights; `state_ema` = the same weights (TRM: released as-is) or EqR's EMA
shadow (the headline; `state` = EqR's raw model weights = the alt row). The eq scalars + table are copied from X0's checkpoint
(pinned by eta_fixed / eta_z_fixed = 1.0; never read on this cell)."""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = next(p_ for p_ in Path(__file__).resolve().parents if (p_ / "src/qhrrn2").exists()); FC = ROOT / "runs/field_ckpts"
# field_models resolves the vendored sources relative to ITS OWN location (runs/field_ckpts/harness/ = the working copy;
# tools/field_ckpts/ = the byte-identical git copy) -> import the loaders from the harness copy
sys.path.insert(0, str(FC / "harness")); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
from qhrrn2 import trm_cell as TC, episodic as E, sudoku_extreme as SX
from qhrrn2.config import Config
import field_models as FM

X0_CKPT = ROOT / "runs/pretrainsportC1_X0/ckpt_latest.pkl"
NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"
MODELS = {  # tag -> (loader tag, loader kwargs, lambda, beta, EMA handling)
    "trmpub":    dict(fm="trm",  kw={}, lam=0.0, beta=0.0, note="alphaXiv/trm-model-sudoku step_32550_sudoku_epoch50k (the registered TRM-MLP reproduction)"),
    "trmpub60k": dict(fm="trm",  kw={"which": "step_39060_sudoku_epoch_60k"}, lam=0.0, beta=0.0, note="alphaXiv step_39060_sudoku_epoch_60k (the second released grid; optional row)"),
    "cgar":      dict(fm="trmc", kw={}, lam=0.0, beta=0.0, note="Kaleemullah/trm-cgar-sudoku pytorch_model.bin (curriculum-trained TRM-MLP)"),
    "eqr":       dict(fm="eqr",  kw={"use_ema": True, "noise_scale": 0.0}, lam=0.05, beta=0.01, note="locuslab/EqR-model sudoku-extreme (EMA shadow = headline; raw model = alt row); lambda_ .95 on F <-> trm_lambda .05; training noise .01"),
}
EQR_BUF_SEED = 20260907


def torch_sd(m):
    return {k: v.detach().float().cpu().numpy() for k, v in m.inner.state_dict().items()}


def conv(W, H0, L0):
    """verify_port.py's conversion (byte-for-byte the same mapping) + the buffers."""
    tok = np.zeros((11, 512), np.float32); tok[:10] = W["embed_tokens.embedding_weight"][1:11]; tok[10] = W["embed_tokens.embedding_weight"][0]
    lm = np.zeros((512, 11), np.float32); lm[:, :10] = W["lm_head.weight"][1:11].T; lm[:, 10] = W["lm_head.weight"][0]
    pe = np.zeros((16, 512), np.float32); pe[0] = W["puzzle_emb.weights"][0]
    q = {"w": np.ascontiguousarray(W["q_head.weight"].T), "b": np.ascontiguousarray(W["q_head.bias"])}
    blocks = [{"mlp_t": {"gate_up": np.ascontiguousarray(W[f"L_level.layers.{i}.mlp_t.gate_up_proj.weight"].T), "down": np.ascontiguousarray(W[f"L_level.layers.{i}.mlp_t.down_proj.weight"].T)},
               "mlp": {"gate_up": np.ascontiguousarray(W[f"L_level.layers.{i}.mlp.gate_up_proj.weight"].T), "down": np.ascontiguousarray(W[f"L_level.layers.{i}.mlp.down_proj.weight"].T)}} for i in range(2)]
    return {"tok_emb": tok, "puzzle_emb": pe, "lm_head": lm, "q_head": q, "blocks": blocks,
            "H_init": np.asarray(H0, np.float32), "L_init": np.asarray(L0, np.float32)}


def trunc_normal_np(rng, shape, std=1.0):
    """HRM/EqR trunc_normal_init_(std): normal truncated at +-2, rescaled so the TRUNCATED std is `std` (inverse-erf sampling)."""
    from scipy.special import erfinv
    a_, b_ = math.erf(-2 / math.sqrt(2)), math.erf(2 / math.sqrt(2)); z_ = (b_ - a_) / 2; c_ = (2 * math.pi) ** -0.5
    pdf = c_ * math.exp(-2.0); comp = std / math.sqrt(1 - (2 * pdf + 2 * pdf) / z_)
    u = rng.uniform(a_, b_, size=shape)
    return np.clip(erfinv(u) * math.sqrt(2) * comp, -2 * comp, 2 * comp).astype(np.float32)


def build_ckpt(tag, spec, x0):
    m = FM.load(spec["fm"], "cpu", torch.float32, **spec["kw"])
    W = torch_sd(m)
    if spec["fm"] == "eqr":
        rng = np.random.default_rng(EQR_BUF_SEED)
        H0, L0 = trunc_normal_np(rng, (512,)), trunc_normal_np(rng, (512,))
        m.H_init, m.L_init = torch.from_numpy(H0), torch.from_numpy(L0)      # the torch side uses the SAME labeled buffers for the check
        p_head = conv(W, H0, L0)                                            # EMA shadow (headline)
        m_raw = FM.load("eqr", "cpu", torch.float32, use_ema=False, noise_scale=0.0)
        p_alt = conv(torch_sd(m_raw), H0, L0)                               # raw model weights (alt row)
    else:
        H0, L0 = W["H_init"], W["L_init"]
        p_head = conv(W, H0, L0); p_alt = p_head
    defaults = Config()
    cfg_d = dict(x0["config"]); cfg_d.update(cell_kind="trm", trm_hidden=512, trm_layers=2, trm_h_cycles=3, trm_l_cycles=6, trm_puzzle_emb_len=16,
                                             trm_expansion=4.0, trm_lambda=float(spec["lam"]), trm_beta=float(spec["beta"]), trm_ri_sigma=0.0, T=16)
    cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in cfg_d.items()})
    eq = {k: np.asarray(v) for k, v in x0["state"]["model"]["eq"].items()}
    table = np.asarray(x0["state"]["table"])
    ck = dict(config=cfg_d, step=0, rng=None, opt_state=None,
              state={"model": {"eq": eq, "trm": p_alt}, "table": table},
              state_ema={"model": {"eq": eq, "trm": p_head}, "table": table},
              port=dict(tag=tag, note=spec["note"], source="tools/field_ckpts/port_field_ckpt.py", date="2026-09-07",
                        eqr_buffer_seed=(EQR_BUF_SEED if spec["fm"] == "eqr" else None), lam=spec["lam"], beta=spec["beta"]))
    return ck, cfg, m, p_head


def check(tag, cfg, m, pj, n, D):
    """torch fp32 (their code, noise 0) vs our JAX fp32 through trm_cell on n strat puzzles for D outer steps (fixed start on both sides)."""
    d = SX.load_prepared(NPZ); ids = SX.stratified_subsample(d["test_rating"], 256, 20260821)[:n]
    puz = d["test_q"][ids].astype(np.int32); sol = d["test_a"][ids].reshape(n, 81).astype(np.int64)
    pjx = jax.tree.map(jnp.asarray, pj)
    batch = m.tokens(puz.astype(np.int64)); st = m.init_state(n, "fixed")
    seg = jax.jit(lambda p_, emb, zH, zL: TC.segment(p_, cfg, emb, zH, zL)); emb = jax.vmap(lambda x: TC.embed(pjx, cfg, x))(jnp.asarray(puz))
    z = jnp.broadcast_to(TC.z0(cfg, 81, p=pjx), (n,) + tuple(TC.carry_shape(cfg, 81))); zH, zL = z[:, 0], z[:, 1]
    rows = []; ex_j_all = []
    for t in range(D):
        zH, zL = jax.vmap(lambda e, h, l: seg(pjx, e, h, l))(emb, zH, zL); lg_j, q_j = jax.vmap(lambda h: TC.readout(pjx, cfg, h, (9, 9)))(zH)
        st, lg_t, q_t = m.step(st, batch)
        a9 = np.asarray(lg_j)[..., 1:10].reshape(n, 81, 9); b9 = m.logits9(lg_t).numpy()
        ex_j = (a9.argmax(-1) + 1 == sol).all(1); ex_t = (b9.argmax(-1) + 1 == sol).all(1); ex_j_all.append(ex_j)
        rows.append(dict(t=t + 1, max_abs=float(np.abs(a9 - b9).max()), mean_abs=float(np.abs(a9 - b9).mean()), scale=float(np.abs(b9).mean()),
                         argmax_agree=float((a9.argmax(-1) == b9.argmax(-1)).mean()), exact_jax=float(ex_j.mean()), exact_torch=float(ex_t.mean()),
                         exact_agree=float((ex_j == ex_t).mean()), q_max_abs=float(np.abs(np.asarray(q_j)[:, 0] - q_t.numpy()).max())))
    return dict(tag=tag, n=n, D=D, idx=ids.tolist(), rows=rows, step1_max_abs=rows[0]["max_abs"], step1_scale=rows[0]["scale"],
                exact_agree_D=rows[-1]["exact_agree"], exact_jax_D=rows[-1]["exact_jax"], exact_torch_D=rows[-1]["exact_torch"],
                exact_jax_by_step=np.stack(ex_j_all).astype(np.uint8).T.tolist())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="trmpub,cgar,eqr,trmpub60k"); ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--D", type=int, default=16); ap.add_argument("--out", default=str(FC / "ported")); a = ap.parse_args()
    # the field venv has no optax (X0's pickle carries optax opt_state classes): a template with config + eq + table only,
    # written by the main venv: PYTHONPATH=src .venv/bin/python -c "..." (see runs/field_ckpts/ported/x0_template.pkl; made 2026-09-07)
    import pickle
    tpl = FC / "ported" / "x0_template.pkl"
    if tpl.exists():
        t = pickle.load(open(tpl, "rb")); x0 = dict(config=t["config"], state={"model": {"eq": t["eq"]}, "table": t["table"]})
    else:
        x0 = E.load_ckpt(str(X0_CKPT))
    for tag in a.models.split(","):
        t0 = time.time(); spec = MODELS[tag]; od = Path(a.out) / tag; od.mkdir(parents=True, exist_ok=True)
        ck, cfg, m, p_head = build_ckpt(tag, spec, x0)
        E.save_ckpt(str(od / "ckpt_latest.pkl"), ck)
        chk = check(tag, cfg, m, p_head, a.n, a.D); chk["wall_s"] = round(time.time() - t0, 1); chk["note"] = spec["note"]
        (od / "port_check.json").write_text(json.dumps(chk, indent=1))
        nparams = sum(int(np.asarray(v).size) for v in jax.tree.leaves({k: v for k, v in p_head.items() if k not in ("H_init", "L_init")}))
        print(f"PORTED {tag}: params {nparams:,} | step-1 max|dlogit| {chk['step1_max_abs']:.2e} (scale {chk['step1_scale']:.1f}) | "
              f"D{a.D} exact jax {100*chk['exact_jax_D']:.1f} torch {100*chk['exact_torch_D']:.1f} agree {100*chk['exact_agree_D']:.1f} % | {chk['wall_s']}s -> {od}", flush=True)


if __name__ == "__main__":
    main()
