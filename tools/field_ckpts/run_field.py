"""Instrument runner for the public field checkpoints (registration: Plan_2026-09-05_FieldCheckpoints_Instruments.md).
Modes: cold | draws | dyn | retain | sym | prefix | jac | train.  Every output = one npz + json under --out (skipped if present)."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
H = Path(__file__).resolve().parent; FC = H.parent; ROOT = FC.parents[1]
sys.path.insert(0, str(H)); sys.path.insert(0, str(ROOT / "src"))
import field_models as FM
from qhrrn2 import sudoku_extreme as SX

UNITS = [[r * 9 + c for c in range(9)] for r in range(9)] + [[r * 9 + c for r in range(9)] for c in range(9)] + \
        [[(3 * br + i) * 9 + (3 * bc + j) for i in range(3) for j in range(3)] for br in range(3) for bc in range(3)]
UNITS = np.asarray(UNITS)
def violations(pred):                       # (n,81) -> duplicate count over the 27 units
    u = pred[:, UNITS]                                       # (n,27,9)
    return 9 * 27 - np.stack([(np.sort(u, -1)[..., 1:] != np.sort(u, -1)[..., :-1]).sum(-1) + 1]).sum(0).sum(-1) if False else \
        (9 - np.array([[len(np.unique(row)) for row in p] for p in u])).sum(-1)

def load_sets():
    d = SX.load_prepared(ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz")
    Q, A, R = d["test_q"], d["test_a"], d["test_rating"]
    def scan20k():
        return np.load(ROOT / "runs/sxscan_psportC2W0/records_all.npz")["idx"].astype(np.int64)
    S = {"scan20k": scan20k, "sub5k": lambda: np.sort(np.random.default_rng(20260905).permutation(scan20k())[:5000]),
         "strat512": lambda: SX.stratified_subsample(R, 512, 20260821), "strat256": lambda: SX.stratified_subsample(R, 256, 20260821),
         "strat84": lambda: SX.stratified_subsample(R, 84, 20260821), "ref500": lambda: np.load(ROOT / "runs/analysis/reference_decoders.npz")["idx"].astype(np.int64),
         "tiny": lambda: SX.stratified_subsample(R, 32, 20260821)}
    return Q, A, R, S

def train_eqr_set():
    """EqR's exact seed-42 1k subsample (their builder: np.random.seed(42); np.random.choice(N, 1000, replace=False) as the FIRST draw)."""
    p = FC / "sets" / "train_eqr.npz"
    if p.exists(): z = np.load(p); return z["q"], z["a"], z["row"]
    csv = ROOT / "data/sudoku_extreme/train.csv"; n = SX.count_rows(csv)
    np.random.seed(42); rows = np.random.choice(n, size=1000, replace=False)
    keep = {int(i) for i in rows}; q, a, r = {}, {}, {}
    for i, _, pq, pa, rt in SX.read_rows(csv, indices=keep): q[i] = pq; a[i] = pa; r[i] = rt
    order = [int(i) for i in rows]
    Qt = np.stack([q[i] for i in order]); At = np.stack([a[i] for i in order]); p.parent.mkdir(exist_ok=True)
    np.savez(p, q=Qt, a=At, row=np.asarray(order), rating=np.asarray([r[i] for i in order]), n_rows=n); return Qt, At, np.asarray(order)

def to_np(t): return t.detach().float().cpu().numpy()

def run_steps(m, puz, sol, D, init, std, gen, record_every=True, prefix_zero=False, state=None):
    """One cold rollout of a batch: returns dict of per-step stats. sol may be None."""
    B = len(puz); batch = m.tokens(puz)
    if prefix_zero:
        pe = m.inner.puzzle_emb.weights; saved = pe.clone(); pe.zero_()
    st = state if state is not None else m.init_state(B, init, std, gen)
    sol_t = None if sol is None else torch.as_tensor(sol.reshape(B, 81), device=m.device)
    ex_full, ex_dig, qh, res_c, res_l, preds, maxp, ent = [], [], [], [], [], [], [], []
    prev_c, prev_l = None, None
    for t in range(D):
        st, lg, q = m.step(st, batch)
        l9 = m.logits9(lg); pred = l9.argmax(-1) + 1                       # digits 1..9
        full_am = lg.argmax(-1)                                              # their 11-class argmax
        if sol_t is not None:
            ex_dig.append(to_np((pred == sol_t).all(1)).astype(bool)); ex_full.append(to_np((full_am == sol_t + 1).all(1)).astype(bool))
        qh.append(to_np(q))
        c = torch.cat([st[0], st[1]], 1).float()
        res_c.append(None if prev_c is None else to_np((c - prev_c).abs().mean((1, 2)))); prev_c = c
        lf = lg.float(); res_l.append(None if prev_l is None else to_np((lf - prev_l).norm(dim=-1).mean(-1))); prev_l = lf
        if record_every:
            p9 = torch.softmax(l9, -1); mp = p9.max(-1).values; e = -(p9 * torch.log(p9 + 1e-12)).sum(-1) / np.log(9)
            preds.append(to_np(pred).astype(np.int8)); maxp.append(to_np(mp).astype(np.float16)); ent.append(to_np(e).astype(np.float16))
    if prefix_zero: pe.copy_(saved)
    out = dict(q=np.stack(qh).astype(np.float16), pred_final=to_np(pred).astype(np.int8), state=st,
               resid3=np.mean(np.stack([r for r in res_c[-3:] if r is not None]), 0), resid3_logit=np.mean(np.stack([r for r in res_l[-3:] if r is not None]), 0),
               resid_c=np.stack([r if r is not None else np.zeros(B) for r in res_c]).astype(np.float32), resid_l=np.stack([r if r is not None else np.zeros(B) for r in res_l]).astype(np.float32))
    if sol_t is not None: out["ex_full"] = np.stack(ex_full); out["ex_dig"] = np.stack(ex_dig)
    if record_every: out["preds"] = np.stack(preds); out["maxp"] = np.stack(maxp); out["ent"] = np.stack(ent)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--mode", required=True); ap.add_argument("--set", default="strat256")
    ap.add_argument("--D", type=int, default=16); ap.add_argument("--k", type=int, default=8); ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--init", default=None); ap.add_argument("--std", type=float, default=1.0); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="mps"); ap.add_argument("--dtype", default="fp32"); ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--ema", type=int, default=1); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(a.seed)
    if (out / "done.json").exists(): print("SKIP (done):", out); return
    t0 = time.time(); dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    kw = {}
    if a.model == "eqr": kw = dict(noise_scale=0.0 if a.noise is None else a.noise, use_ema=bool(a.ema))
    m = FM.load(a.model, a.device, dtype, **kw)
    init = a.init or ("trunc" if a.model == "eqr" else "fixed")
    Q, A, R, S = load_sets()
    if a.mode == "train":
        puz_all, sol_all, rows = train_eqr_set(); idx = rows; rat = np.load(FC / "sets" / "train_eqr.npz")["rating"]
    else:
        idx = S[a.set](); puz_all, sol_all, rat = Q[idx], A[idx], R[idx]
    if a.limit: idx, puz_all, sol_all, rat = idx[:a.limit], puz_all[:a.limit], sol_all[:a.limit], rat[:a.limit]
    n = len(idx); giv = (puz_all != 0).reshape(n, -1).sum(1)
    puz_all = puz_all.reshape(n, 81).astype(np.int64); sol_all = sol_all.reshape(n, 81).astype(np.int64)
    proto = dict(model=a.model, mode=a.mode, set=a.set, D=a.D, k=a.k, noise=a.noise, init=init, std=a.std, dtype=a.dtype, device=a.device, seed=a.seed, ema=a.ema, n=n)
    print("PROTO", json.dumps(proto), flush=True)
    rec = dict(idx=idx, rating=rat, givens=giv)
    gen = torch.Generator().manual_seed(a.seed)
    if a.mode in ("cold", "train", "prefix"):
        exf, exd, qs, r3, r3l, pf, mp, en = [], [], [], [], [], [], [], []
        for b0 in range(0, n, a.batch):
            o = run_steps(m, puz_all[b0:b0 + a.batch], sol_all[b0:b0 + a.batch], a.D, init, a.std, gen, record_every=False, prefix_zero=(a.mode == "prefix"))
            exf.append(o["ex_full"]); exd.append(o["ex_dig"]); qs.append(o["q"]); r3.append(o["resid3"]); r3l.append(o["resid3_logit"]); pf.append(o["pred_final"])
            if b0 % (a.batch * 8) == 0: print(f"  {b0 + len(o['pred_final'])}/{n}  exact@D so far {np.concatenate([e[-1] for e in exf]).mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)
        rec.update(exact_by_step=np.concatenate(exf, 1), exact_digit_by_step=np.concatenate(exd, 1), q_by_step=np.concatenate(qs, 1),
                   resid3=np.concatenate(r3), resid3_logit=np.concatenate(r3l), pred_final=np.concatenate(pf))
        rec["cold_exact"] = rec["exact_by_step"][-1]; fe = rec["exact_by_step"].argmax(0); rec["first_exact"] = np.where(rec["exact_by_step"].any(0), fe, -1)
        rec["violations"] = violations(rec["pred_final"].astype(np.int64)); rec["givens_kept"] = ((rec["pred_final"] == puz_all) | (puz_all == 0)).all(1)
        summ = dict(exact_acc=float(rec["cold_exact"].mean()), exact_by_step=[float(x) for x in rec["exact_by_step"].mean(1)], valid_wrong_frac=float(((rec["violations"] == 0) & ~rec["cold_exact"]).mean()),
                    given_violating_valid_frac=float(((rec["violations"] == 0) & ~rec["givens_kept"]).mean()), q_halt_acc=float(((rec["q_by_step"][-1] >= 0) == rec["cold_exact"]).mean()))
    elif a.mode == "draws":
        K = a.k; ex = np.zeros((n, K), bool); rs = np.zeros((n, K), np.float32); rl = np.zeros((n, K), np.float32); ok = np.zeros((n, K), bool)
        pairs = [(i, j) for i in range(n) for j in range(K)]
        for b0 in range(0, len(pairs), a.batch):
            pb = pairs[b0:b0 + a.batch]; ii = np.array([p[0] for p in pb]); jj = np.array([p[1] for p in pb])
            g = torch.Generator().manual_seed(a.seed + b0)
            o = run_steps(m, puz_all[ii], sol_all[ii], a.D, "random", a.std, g, record_every=False)
            ex[ii, jj] = o["ex_full"][-1]; rs[ii, jj] = o["resid3"]; rl[ii, jj] = o["resid3_logit"]
            v = violations(o["pred_final"].astype(np.int64)); gk = ((o["pred_final"] == puz_all[ii]) | (puz_all[ii] == 0)).all(1); ok[ii, jj] = (v == 0) & gk
            if b0 % (a.batch * 20) == 0: print(f"  {b0 + len(pb)}/{len(pairs)} draws  b1 so far {ex[:, 0][:max(1, (b0 + len(pb)) // K)].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
        fh = np.where(ok.any(1), ok.argmax(1), -1)
        rec.update(mi_exact_k=ex, mi_resid_k=rs, mi_resid_logit_k=rl, mi_verified_k=ok, mi_first_hit=fh, mi_verified=ok.sum(1))
        summ = dict(b1_exact=float(ex[:, 0].mean()), verified_at_k={str(k): float(((fh >= 0) & (fh < k)).mean()) for k in (1, 2, 4, 8) if k <= K}, mean_draw_exact=float(ex.mean()))
    elif a.mode == "dyn":
        o = run_steps(m, puz_all, sol_all, a.D, init, a.std, gen, record_every=True)
        rec.update(preds=o["preds"], maxp=o["maxp"], ent=o["ent"], q_by_step=o["q"], resid_c=o["resid_c"], resid_l=o["resid_l"], exact_by_step=o["ex_full"], sol=sol_all.astype(np.int8), puz=puz_all.astype(np.int8))
        summ = dict(exact_acc=float(o["ex_full"][-1].mean()), exact_by_step=[float(x) for x in o["ex_full"].mean(1)])
    elif a.mode == "retain":
        st = m.solution_state(sol_all)
        o = run_steps(m, puz_all, sol_all, a.D, init, a.std, gen, record_every=False, state=st)
        rec.update(retained_by_step=o["ex_full"], q_by_step=o["q"], resid_c=o["resid_c"])
        summ = dict(retained_by_step=[float(x) for x in o["ex_full"].mean(1)], retained_all=float(o["ex_full"].all(0).mean()), retained_final=float(o["ex_full"][-1].mean()))
    elif a.mode == "sym":
        rng = np.random.default_rng(a.seed); V = 9; exs = np.zeros((V, n), bool); pb = np.zeros((V, n, 81), np.int8)
        for v in range(V):
            if v == 0: dmap = np.arange(10); tr = False
            else: dmap = np.concatenate([[0], rng.permutation(9) + 1]); tr = bool(rng.random() < .5)
            pv = dmap[puz_all]; sv = dmap[sol_all]
            if tr: pv = pv.reshape(n, 9, 9).transpose(0, 2, 1).reshape(n, 81); sv = sv.reshape(n, 9, 9).transpose(0, 2, 1).reshape(n, 81)
            o = run_steps(m, pv, sv, a.D, init, a.std, torch.Generator().manual_seed(a.seed), record_every=False)
            exs[v] = o["ex_full"][-1]; pr = o["pred_final"].astype(np.int64)
            if tr: pr = pr.reshape(n, 9, 9).transpose(0, 2, 1).reshape(n, 81)
            inv = np.argsort(dmap); pb[v] = inv[pr]                 # mapped back to the original frame
            print(f"  variant {v}: exact {exs[v].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
        oh = (pb[:, :, :, None] == np.arange(1, 10)).sum(0); maj = oh.argmax(-1) + 1; vote = (maj == sol_all).all(1)
        rec.update(exact_by_variant=exs, preds_back=pb, orbit_vote_exact=vote)
        summ = dict(exact_identity=float(exs[0].mean()), exact_mean_over_orbit=float(exs.mean()), orbit_consistent=float((exs.all(0) | ~exs.any(0)).mean()),
                    orbit_any=float(exs.any(0).mean()), orbit_vote=float(vote.mean()), orbit_vote_gain=float(vote.mean() - exs[0].mean()))
    elif a.mode == "initrad":
        # EXPLORATORY (post-hoc 2026-09-06, motivated by HRM's dead random-init draws): the init-basin radius — exact@D vs the size of a
        # perturbation of the TRAINED fixed init (z0 = init + eps * N(0,1)), eps in EPS; eps = 0 is the cold pass, eps -> inf the random draw.
        EPS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]; res = {}
        for eps in EPS:
            g = torch.Generator().manual_seed(a.seed); exs = []
            for b0 in range(0, n, a.batch):
                B = min(a.batch, n - b0); st0 = m.init_state(B, "fixed"); zn = m.init_state(B, "random", 1.0, g)
                st = (st0[0] + eps * zn[0], st0[1] + eps * zn[1])
                o = run_steps(m, puz_all[b0:b0 + B], sol_all[b0:b0 + B], a.D, init, a.std, g, record_every=False, state=st); exs.append(o["ex_full"][-1])
            res[str(eps)] = float(np.concatenate(exs).mean()); print(f"  eps {eps}: exact@{a.D} {res[str(eps)]:.4f}", flush=True)
        rec.update(eps=np.array(EPS), exact_vs_eps=np.array([res[str(e)] for e in EPS])); summ = dict(exact_vs_eps=res)
    elif a.mode == "jac":
        o = run_steps(m, puz_all, sol_all, a.D, init, a.std, gen, record_every=False)
        st = o["state"]; batch = m.tokens(puz_all); zc = torch.cat([st[0], st[1]], 1).float()
        with torch.no_grad():
            st1, _, _ = m.step(st, batch); z1 = torch.cat([st1[0], st1[1]], 1).float(); resid = ((z1 - zc).norm(dim=(1, 2)) / zc.norm(dim=(1, 2)))
            eps = 1e-2 * zc.norm(dim=(1, 2), keepdim=True) / np.sqrt(zc[0].numel()); v = torch.randn_like(zc); v = v / v.norm(dim=(1, 2), keepdim=True); rad = []
            for it in range(8):
                zp = zc + eps * v; sp = (zp[:, :m.seq].to(m.dtype), zp[:, m.seq:].to(m.dtype)); s2, _, _ = m.step(sp, batch); z2 = torch.cat([s2[0], s2[1]], 1).float()
                dv = (z2 - z1); nrm = dv.norm(dim=(1, 2), keepdim=True); rad.append(to_np(nrm.squeeze() / eps.squeeze())); v = dv / (nrm + 1e-12)
        rec.update(exact=o["ex_full"][-1], resid_rel=to_np(resid), radius_iters=np.stack(rad))
        summ = dict(exact=float(o["ex_full"][-1].mean()), radius_med_solved=float(np.median(rad[-1][o["ex_full"][-1]])) if o["ex_full"][-1].any() else None,
                    radius_med_unsolved=float(np.median(rad[-1][~o["ex_full"][-1]])) if (~o["ex_full"][-1]).any() else None,
                    resid_med_solved=float(np.median(to_np(resid)[o["ex_full"][-1]])) if o["ex_full"][-1].any() else None, resid_med_unsolved=float(np.median(to_np(resid)[~o["ex_full"][-1]])) if (~o["ex_full"][-1]).any() else None)
    else: raise ValueError(a.mode)
    proto["wall_s"] = round(time.time() - t0, 1)
    np.savez_compressed(out / "records_all.npz", **rec); (out / "summary.json").write_text(json.dumps(dict(proto=proto, summary=summ), indent=1)); (out / "done.json").write_text(json.dumps(proto))
    print("SUMMARY", json.dumps(summ)[:600]); print(f"DONE {out} ({proto['wall_s']}s)")

if __name__ == "__main__":
    main()
