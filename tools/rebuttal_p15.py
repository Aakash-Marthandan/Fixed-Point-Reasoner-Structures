#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P15 — how the network treats a changed decision, and whether the hidden state controls it
# (registration Documentation/Note_2026-10-02_Rebuttal_P15_P16_P17_Registration.md, written before this build). MEASUREMENT, $0,
# inference only on the banked attention checkpoints; imports the P1/P2 runner (tools/rebuttal_p1p2.py) unchanged.
"""P15: decision flips (a swap of two digits' readout components at one cell) applied to intact fixed-start states, with and
without the readout-invisible slow state replaced, continued four outer iterations. Rules R1 (treatment of the displayed decision),
R2 (hidden-state control), R3 (coordinated credit assignment). `linear` measures the one-iteration Jacobian on DISCOVERY states."""
import argparse, json, math, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p1p2 as P                                   # the P1/P2 runner (model loading, projection), unchanged

OUT = ROOT / "runs/analysis/rebuttal_20261002"
SEED = 20261002
WIDTHS = (256, 192, 128)
TS = (1, 2, 3)
K = 4
CLASSES = ("HA", "HN", "XB", "XN"); CODE = {c: i + 1 for i, c in enumerate(CLASSES)}
PER_CLASS = 3
KINDS = ("intact_noflip", "intact_flip", "kernel_noflip", "kernel_flip", "reencode_flip"); KIND = {k: i for i, k in enumerate(KINDS)}
VBATCH = 128
PEERS = P.UNITS                                             # (81, 81) bool, self excluded
E_LOW, E_HIGH = 0.10, 0.50                                  # R1
D_HIGH, D_LOW = 0.20, 0.05                                  # R2
EJ_HIGH, EJ_LOW, BLAME_HIGH, MIN_PAIRS = 0.30, 0.05, 0.75, 30   # R3


def utc(): return datetime.now(timezone.utc).isoformat()


# ---------------- pure helpers (selftested) ----------------
def discovery_ids():
    """The seeded puzzle-level split: the first half of a permutation of the 512 sorted repair-puzzle IDs is DISCOVERY."""
    ATS, STUDY = P.study_paths()
    with np.load(STUDY / "inputs/repair.npz", allow_pickle=False) as d: rid = np.sort(d["ids"])
    perm = np.random.default_rng(SEED).permutation(len(rid))
    return set(int(x) for x in rid[perm[:len(rid) // 2]])


def eligible(grid, sol, puz):
    """grid, sol, puz: (81,) displayed digits (1..9 everywhere), solution, puzzle (0 = empty). -> class -> [(cell, digit)]."""
    out = {c: [] for c in CLASSES}
    for i in np.where(puz == 0)[0]:
        disp = set(int(x) for x in grid[PEERS[i]])
        if grid[i] != sol[i]:
            b = int(sol[i]); out["HA" if b in disp else "HN"].append((int(i), b))
        else:
            for b in range(1, 10):
                if b != sol[i]: out["XB" if b in disp else "XN"].append((int(i), b))
    return out


def select(elig, pid, t):
    """Up to PER_CLASS distinct cells per class, uniformly, then a digit uniformly among that cell's eligible digits."""
    chosen = []
    for c in CLASSES:
        rng = np.random.default_rng([SEED, int(pid), int(t), CODE[c]])
        cells = sorted(set(i for i, _ in elig[c]))
        for i in (rng.permutation(cells)[:PER_CLASS] if cells else []):
            digs = [b for (j, b) in elig[c] if j == int(i)]
            chosen.append((c, int(i), int(digs[rng.integers(len(digs))])))
    return chosen


def partners(grid, i, b):
    """Peers of i displaying b."""
    return [int(j) for j in np.where(PEERS[i] & (grid == b))[0]]


def swap_flip(h, s, i, a, b, lm_unit):
    """h (F, S, w) slow state, s (S, F) its scores: swap the readout components of digits a and b at cell i (b takes a's score)."""
    h = h.copy(); da = s[i, b - 1] - s[i, a - 1]
    h[a - 1, i] += (da * lm_unit).astype(h.dtype)
    h[b - 1, i] -= (da * lm_unit).astype(h.dtype)
    return h


def ratio_gain(c_flip, c_no):
    """Share of the available gain realized: (P(c|flip) - P(c|no)) / (1 - P(c|no)), pooled as a ratio of means."""
    a, b = float(np.mean(c_flip)), float(np.mean(c_no))
    return (a - b) / (1.0 - b) if b < 1.0 else float("nan")


def ratio_loss(c_flip, c_no):
    """Share of naturally kept correct decisions destroyed: (P(c|no) - P(c|flip)) / P(c|no)."""
    a, b = float(np.mean(c_flip)), float(np.mean(c_no))
    return (b - a) / b if b > 0 else float("nan")


def letter_r1(ep, em):
    if not (np.isfinite(ep) and np.isfinite(em)): return "UNDEFINED"
    if ep <= E_LOW and em <= E_LOW: return "DISPLAY-INERT"
    if ep >= E_HIGH and em >= E_HIGH: return "FOLLOWER"
    if ep >= E_HIGH and em <= E_LOW: return "SELECTIVE"
    return "MIXED"


def letter_r2(dp, dm):
    if not (np.isfinite(dp) and np.isfinite(dm)): return "UNDEFINED"
    if max(abs(dp), abs(dm)) >= D_HIGH: return "KERNEL-CONTROLS"
    if abs(dp) <= D_LOW and abs(dm) <= D_LOW: return "KERNEL-NEUTRAL"
    return "MIXED"


def letter_r3(ej, blame, n_pairs):
    if n_pairs < MIN_PAIRS or not np.isfinite(ej): return "UNDEFINED"
    if ej >= EJ_HIGH and np.isfinite(blame) and blame >= BLAME_HIGH: return "COORDINATES"
    if ej <= EJ_LOW: return "LOCAL"
    return "MIXED"


# ---------------- the runner ----------------
class R7(P.R2):
    def __init__(self, width, out=OUT):
        super().__init__(width, out)
        self.meta.update(study="rebuttal_20261002", pipeline="tools/rebuttal_p15.py over tools/rebuttal_p1p2.py over tools/attention_transfer_study.py")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)
        jax, jnp, DC, cfg = self.jax, self.jnp, self.DC, self.cfg
        self.p_dec = self.params["dec"]
        self.lm32 = np.asarray(self.lm, np.float32); self.lm_unit = (self.lm / (self.lm @ self.lm)).astype(np.float32)
        lmj = jnp.asarray(self.lm32); p_dec = self.p_dec; lam = cfg.trm_lambda; eta_z = self.eta_z

        def outer(zH, zL, emb):                    # the release's `segment` without its two training-time gradient stops; rng None
            def step(z, inj):
                Fz = DC._stack(p_dec, z + inj, cfg)
                return z + (1.0 - lam) * (Fz - z) if lam > 0 else Fz
            for c in range(cfg.trm_h_cycles):
                for _ in range(cfg.trm_l_cycles): zL = step(zL, zH + emb)
                zH = step(zH, zL)
            return zH, zL
        self.outer_fn = outer

        def rollout(zH, zL, emb):                  # K outer iterations; scores from the segment output, carry as the study's update
            sc = []
            for _ in range(K):
                fH, fL = outer(zH, zL, emb)
                sc.append(jnp.einsum("fsw,w->sf", fH, lmj))
                zH = zH + eta_z * (fH - zH); zL = zL + eta_z * (fL - zL)
            return jnp.stack(sc)                   # (K, S, F)
        self.roll = jax.jit(jax.vmap(rollout))
        self.outer_b = jax.jit(jax.vmap(outer))
        self.emb_b = jax.jit(jax.vmap(lambda xt: DC.embed(p_dec, cfg, xt)))

    def release_states(self, puz):
        """The study's loop at the given batch: per t = 1..3 the carried state z, the segment output and the logits."""
        jax, jnp, EV = self.jax, self.jnp, self.EV
        x = EV.place_batch(puz, self.layout)
        void = jax.nn.one_hot(jnp.full((9, 9), self.G.VOID, jnp.int32), 11).transpose(2, 0, 1)
        y = jnp.broadcast_to(void, (len(puz),) + void.shape); z = None; out = []
        for t in range(max(TS)):
            first = z is None
            lg_dev, zf = EV._step(self.cfg, 1., 0., first)(self.params, x, y, self.tv, jnp.zeros(1) if first else z)
            z = zf if first else z + self.eta_z * (zf - z)
            p = jax.nn.softmax(lg_dev, axis=-1).transpose(0, 3, 1, 2); y = y + self.eta * (p - y)
            out.append(dict(z=np.asarray(z), zf=np.asarray(zf), logits=np.asarray(EV.layout_gather(lg_dev, self.layout), np.float32)))
        return x, out

    def run(self, smoke=False):
        try:
            ATS, STUDY = P.study_paths()
            with np.load(STUDY / "inputs/interventions.npz", allow_pickle=False) as d: ids_all, puz_all, sol_all = d["ids"], d["puz"], d["sol"]
            disc = discovery_ids(); dirr = self.dir / ("flips_smoke" if smoke else "flips"); gates = {}
            for b0 in range(0, len(ids_all), 128):
                sl = slice(b0, b0 + 128); ids, puz, sol = ids_all[sl], puz_all[sl], sol_all[sl]
                paths = {t: dirr / f"b{b0:04d}_t{t}.npz" for t in TS}
                if not smoke and all(self.cached_any(paths[t]) for t in TS): continue
                x, st = self.release_states(puz)
                with np.load(STUDY / f"attention_{self.width}/interventions/intact/batch_{b0:04d}.npz", allow_pickle=False) as d: ref, refpred = d["logits"], d["pred"]
                g1 = all(np.array_equal(st[t]["logits"], ref[t]) for t in range(max(TS)))
                assert g1, "gate 1: release states differ from the study's logits"
                emb = np.asarray(self.emb_b(x))
                fH, fL = self.outer_b(st[0]["z"][:, 0], st[0]["z"][:, 1], emb)
                g2 = bool(np.array_equal(np.asarray(fH), st[1]["zf"][:, 0]) and np.array_equal(np.asarray(fL), st[1]["zf"][:, 1]))
                assert g2, "gate 2: the gradient-stop-free outer iteration differs from the release step"
                gates[f"b{b0:04d}"] = dict(release_states_bitwise=g1, outer_bitwise=g2)
                n = 16 if smoke else len(ids)
                for t in (TS[:1] if smoke else TS):
                    if not smoke and self.cached_any(paths[t]): continue
                    started = time.monotonic()
                    rec = self.flip_chunk(ids[:n], puz[:n], sol[:n], st[t - 1]["z"][:n], emb[:n], refpred[t:t + K, :n], t, disc, progress=dict(batch=b0, t=t))
                    rec["meta"] = json.dumps(dict(self.meta, t=t, batch_start=b0, n=n, seconds=time.monotonic() - started, created=utc(), gates=gates[f"b{b0:04d}"], smoke=smoke))
                    if smoke:
                        print(json.dumps(dict(gates=gates[f"b{b0:04d}"], **json.loads(rec["gate_summary"])))); return
                    self.ATS.write_npz(paths[t], **rec); self.ATS.write_json(paths[t].with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(paths[t])))
                    self.ATS.log("chunk_complete", width=self.width, batch=b0, t=t, states=int(len(rec["state_ids"])), variants=int(len(rec["var_kind"])), seconds=round(time.monotonic() - started, 1))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group="p15")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise

    def cached_any(self, path):
        if not path.exists(): return False
        side = path.with_suffix(".sha256.json")
        if not side.exists() or json.loads(side.read_text())["sha256"] != self.ATS.sha(path): raise RuntimeError(f"Incomplete or altered output: {path}")
        return True

    def flip_chunk(self, ids, puz, sol, z, emb, refpred, t, disc, progress=None):
        """Builds every variant for every eligible state of the batch (lazily), rolls them out K iterations, checks gates 3-5."""
        jnp = self.jnp; lm = self.lm; lmu = self.lm_unit
        specs, V = [], dict(state=[], kind=[], cls=[], cell=[], frm=[], to=[], flip=[], partners=[])
        S = dict(ids=[], t=[], split=[], grid=[], sol=[], puz=[], kshift64=[], kshift32=[]); per = {}
        for b in range(len(ids)):
            pid = int(ids[b]); pz, so = puz[b].reshape(81), sol[b].reshape(81)
            h = z[b, 0]
            s = np.einsum("fsw,w->sf", h.astype(np.float64), lm); grid = s.argmax(1) + 1
            if t > 1 and not bool(((grid != so) & (pz == 0)).any()): continue
            fl = select(eligible(grid, so, pz), pid, t)
            if not fl: continue
            si = len(S["ids"])
            par, perp = P.project(h, lm)                                         # the kernel replacement (P2's edit, once), scores kept
            hk64 = par + P.norm_matched_random(perp, lm, np.random.default_rng([SEED, pid, 51, t]))
            shift64 = float(np.abs(hk64 @ lm - h.astype(np.float64) @ lm).max()); assert shift64 < 1e-2, f"gate 5: score shift {shift64}"
            hk = hk64.astype(np.float32); sk = np.einsum("fsw,w->sf", hk.astype(np.float64), lm)
            shift32 = float(np.abs(np.einsum("fsw,w->sf", hk, self.lm32) - np.einsum("fsw,w->sf", h, self.lm32)).max())
            per[si] = dict(b=b, h=h, s=s, hk=hk, sk=sk, grid=grid)
            for key, val in (("ids", pid), ("t", t), ("split", 0 if pid in disc else 1), ("grid", grid.astype(np.int8)), ("sol", so.astype(np.int8)), ("puz", pz.astype(np.int8)),
                             ("kshift64", shift64), ("kshift32", shift32)):
                S[key].append(val)
            def add(kind, cls=0, cell=-1, frm=0, to=0, fi=-1, prt=()):
                specs.append((si, kind, cell, frm, to)); V["state"].append(si); V["kind"].append(KIND[kind]); V["cls"].append(cls); V["cell"].append(cell)
                V["frm"].append(frm); V["to"].append(to); V["flip"].append(fi); pp = np.full(20, -1, np.int16); pp[:len(prt)] = prt; V["partners"].append(pp)
            add("intact_noflip"); add("kernel_noflip")
            for fi, (c, i, bb) in enumerate(fl):
                a = int(grid[i]); prt = partners(grid, i, bb)
                for kind in ("intact_flip", "kernel_flip", "reencode_flip"): add(kind, CODE[c], i, a, bb, fi, prt)
        flip_gate_fail = 0; reencode_display = []

        def build(spec):
            nonlocal flip_gate_fail
            si, kind, i, a, bb = spec; q = per[si]; l = z[q["b"], 1]
            if kind == "intact_noflip": return q["h"], l
            if kind == "kernel_noflip": return q["hk"], l
            want = q["grid"].copy(); want[i] = bb
            if kind == "reencode_flip":
                ea = np.asarray(self.embed(jnp.asarray(want.reshape(1, 9, 9), jnp.int32)))[0]
                hr = q["h"].copy(); hr[:, i] = ea[:, i]
                reencode_display.append(int(np.einsum("fw,w->f", hr[:, i], self.lm32).argmax()) + 1 == bb)
                return hr, l
            h0, s0 = (q["h"], q["s"]) if kind == "intact_flip" else (q["hk"], q["sk"])
            hf = swap_flip(h0, s0, i, a, bb, lmu)
            if not np.array_equal(np.einsum("fsw,w->sf", hf, self.lm32).argmax(1) + 1, want): flip_gate_fail += 1
            return hf, l

        out = np.zeros((len(specs), K, 81, 9), np.float32)
        for c0 in range(0, len(specs), VBATCH):
            chunk = specs[c0:c0 + VBATCH]; pad = VBATCH - len(chunk)
            built = [build(sp) for sp in chunk]
            zH = np.stack([x[0] for x in built] + [built[0][0]] * pad); zL = np.stack([x[1] for x in built] + [built[0][1]] * pad)
            e = np.stack([emb[per[sp[0]]["b"]] for sp in chunk] + [emb[per[chunk[0][0]]["b"]]] * pad)
            out[c0:c0 + len(chunk)] = np.asarray(self.roll(jnp.asarray(zH), jnp.asarray(zL), jnp.asarray(e)))[:len(chunk)]
            if progress: self.status("running", **progress, variants_done=c0 + len(chunk), variants=len(specs))
        assert flip_gate_fail == 0, f"gate 4: {flip_gate_fail} flips did not produce the designed displayed grid"
        pred = (out.argmax(-1) + 1).astype(np.int8)                              # (V, K, S)
        st = np.asarray(V["state"]); kd = np.asarray(V["kind"]); g3_fail = 0
        for v in np.where(kd == KIND["intact_noflip"])[0]:                      # gate 3
            b = per[int(st[v])]["b"]
            if not np.array_equal(pred[v], refpred[:, b].reshape(K, 81).astype(np.int8)): g3_fail += 1
        assert g3_fail == 0, f"gate 3: {g3_fail} intact no-flip rollouts differ from the study's displayed grids"
        summary = dict(states=len(S["ids"]), variants=len(specs), flip_gate_failures=flip_gate_fail, noflip_gate_failures=g3_fail,
                       reencode_displays_b=float(np.mean(reencode_display)) if reencode_display else None,
                       max_kernel_shift64=float(max(S["kshift64"])) if S["kshift64"] else None, max_kernel_shift32=float(max(S["kshift32"])) if S["kshift32"] else None)
        return dict(state_ids=np.asarray(S["ids"], np.int64), state_t=np.asarray(S["t"], np.int8), state_split=np.asarray(S["split"], np.int8),
                    state_grid=np.stack(S["grid"]), state_sol=np.stack(S["sol"]), state_puz=np.stack(S["puz"]),
                    state_kshift64=np.asarray(S["kshift64"]), state_kshift32=np.asarray(S["kshift32"]),
                    var_state=st.astype(np.int32), var_kind=kd.astype(np.int8), var_cls=np.asarray(V["cls"], np.int8), var_cell=np.asarray(V["cell"], np.int16),
                    var_from=np.asarray(V["frm"], np.int8), var_to=np.asarray(V["to"], np.int8), var_flip=np.asarray(V["flip"], np.int8),
                    var_partners=np.stack(V["partners"]), pred=pred, scores=out.astype(np.float16), gate_summary=json.dumps(summary))


# ---------------- the read ----------------
def load_width(w, base=OUT):
    parts = sorted((base / f"attention_{w}/flips").glob("b*_t*.npz"))
    if not parts: return None
    rows = []
    for pth in parts:
        side = pth.with_suffix("").with_suffix(".sha256.json")
        import hashlib; assert hashlib.sha256(pth.read_bytes()).hexdigest() == json.loads(side.read_text())["sha256"], f"digest {pth}"
        with np.load(pth, allow_pickle=False) as d: rows.append({k: d[k] for k in d.files if k not in ("scores", "meta", "gate_summary")})
    return rows


def flip_table(rows, k):
    """One record per (state, flip) with outcomes at iteration t+k under the three flip channels and both no-flip references."""
    recs = []
    for r in rows:
        kd, st, fi = r["var_kind"], r["var_state"], r["var_flip"]
        for s in range(len(r["state_ids"])):
            m = st == s; sol = r["state_sol"][s].astype(int)
            no_i = np.where(m & (kd == KIND["intact_noflip"]))[0][0]; no_k = np.where(m & (kd == KIND["kernel_noflip"]))[0][0]
            for f in sorted(set(fi[m & (fi >= 0)].tolist())):
                vi = np.where(m & (fi == f) & (kd == KIND["intact_flip"]))[0][0]; vk = np.where(m & (fi == f) & (kd == KIND["kernel_flip"]))[0][0]
                vr = np.where(m & (fi == f) & (kd == KIND["reencode_flip"]))[0][0]
                i, b = int(r["var_cell"][vi]), int(r["var_to"][vi]); prt = [int(j) for j in r["var_partners"][vi] if j >= 0]
                P_ = r["pred"][:, k - 1].astype(int)                                 # (V, S)
                recs.append(dict(state=(int(r["state_ids"][s]), int(r["state_t"][s])), split=int(r["state_split"][s]), t=int(r["state_t"][s]), cls=CLASSES[int(r["var_cls"][vi]) - 1], i=i, b=b,
                                 c={kk: int(P_[v, i] == sol[i]) for kk, v in (("intact_flip", vi), ("kernel_flip", vk), ("reencode_flip", vr), ("intact_noflip", no_i), ("kernel_noflip", no_k))},
                                 keep={kk: int(P_[v, i] == b) for kk, v in (("intact_flip", vi), ("kernel_flip", vk), ("reencode_flip", vr))},
                                 pairs=[dict(j=j, left={kk: int(P_[v, j] != b) for kk, v in (("intact_flip", vi), ("kernel_flip", vk), ("reencode_flip", vr), ("intact_noflip", no_i), ("kernel_noflip", no_k))},
                                             i_has={kk: int(P_[v, i] == b) for kk, v in (("intact_flip", vi), ("kernel_flip", vk), ("reencode_flip", vr))}) for j in prt],
                                 other_repairs={kk: int((P_[v] == sol).sum() - (P_[no] == sol).sum() - (int(P_[v, i] == sol[i]) - int(P_[no, i] == sol[i]))) for kk, v, no in (("intact_flip", vi, no_i), ("kernel_flip", vk, no_k))}))
    return recs


def summarize(recs, channel="intact", noref=None):
    """e+, e-, e_j, blame for a channel ('intact', 'kernel' or 'reencode'); the no-flip reference is the channel's own (reencode uses intact)."""
    fk = f"{channel}_flip"; nk = noref or (f"{channel}_noflip" if channel != "reencode" else "intact_noflip")
    H = [r for r in recs if r["cls"] in ("HA", "HN")]; X = [r for r in recs if r["cls"] in ("XB", "XN")]
    ep = ratio_gain([r["c"][fk] for r in H], [r["c"][nk] for r in H]) if H else float("nan")
    em = ratio_loss([r["c"][fk] for r in X], [r["c"][nk] for r in X]) if X else float("nan")
    pairs = [(r, p) for r in recs if r["cls"] == "HA" for p in r["pairs"]]
    ej = ratio_gain([p["left"][fk] for _, p in pairs], [p["left"][nk] for _, p in pairs]) if pairs else float("nan")
    res = [(p["i_has"][fk], 1 - p["left"][fk]) for _, p in pairs]                 # (i displays b, j displays b)
    resolved = [(a, bb) for a, bb in res if a + bb == 1]
    blame = float(np.mean([a for a, _ in resolved])) if resolved else float("nan")
    return dict(n_H=len(H), n_X=len(X), n_pairs=len(pairs), n_resolved=len(resolved), e_plus=ep, e_minus=em, e_j=ej, blame=blame,
                P_c_flip_H=float(np.mean([r["c"][fk] for r in H])) if H else None, P_c_noflip_H=float(np.mean([r["c"][nk] for r in H])) if H else None,
                P_c_flip_X=float(np.mean([r["c"][fk] for r in X])) if X else None, P_c_noflip_X=float(np.mean([r["c"][nk] for r in X])) if X else None,
                keep_H=float(np.mean([r["keep"][fk] for r in H])) if H else None, keep_X=float(np.mean([r["keep"][fk] for r in X])) if X else None)


def report(base=OUT):
    lines = ["P15 REPORT (rules registered 2026-10-02): decision flips on intact fixed-start states; letters on CONFIRMATION states, k = 1"]; out = {}
    for w in WIDTHS:
        rows = load_width(w, base)
        if rows is None: lines.append(f"attention_{w}: not run"); continue
        res = {}
        for k in range(1, K + 1):
            recs = flip_table(rows, k); res[k] = {}
            for split, name in ((1, "confirmation"), (0, "discovery")):
                rs = [r for r in recs if r["split"] == split]
                si, sk, sr = summarize(rs, "intact"), summarize(rs, "kernel"), summarize(rs, "reencode")
                d = dict(intact=si, kernel=sk, reencode=sr, delta_plus=sk["e_plus"] - si["e_plus"], delta_minus=sk["e_minus"] - si["e_minus"],
                         by_class={c: summarize([r for r in rs if r["cls"] == c], "intact") for c in CLASSES},
                         by_t={t: summarize([r for r in rs if r["t"] == t], "intact") for t in TS},
                         cascade={ch: float(np.mean([r["other_repairs"][f"{ch}_flip"] for r in rs if r["cls"] in ("HA", "HN")])) if rs else None for ch in ("intact", "kernel")})
                if k == 1 and split == 1:
                    d["R1"] = letter_r1(si["e_plus"], si["e_minus"]); d["R2"] = letter_r2(d["delta_plus"], d["delta_minus"]); d["R3"] = letter_r3(si["e_j"], si["blame"], si["n_pairs"])
                res[k][name] = d
                lines.append(f"attention_{w} k={k} {name}: intact e+ {si['e_plus']:.3f} e- {si['e_minus']:.3f} e_j {si['e_j']:.3f} blame {si['blame']:.3f} (H {si['n_H']}, X {si['n_X']}, pairs {si['n_pairs']}) | "
                             f"kernel e+ {sk['e_plus']:.3f} e- {sk['e_minus']:.3f} e_j {sk['e_j']:.3f} blame {sk['blame']:.3f} | d+ {d['delta_plus']:+.3f} d- {d['delta_minus']:+.3f} | reencode e+ {sr['e_plus']:.3f} e- {sr['e_minus']:.3f}"
                             + (f" || R1 {d['R1']} R2 {d['R2']} R3 {d['R3']}" if "R1" in d else ""))
        out[f"attention_{w}"] = res
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(lines))


# ---------------- selftest ----------------
def selftest():
    rng = np.random.default_rng(0); w = 16; lm = rng.standard_normal(w); lmu = lm / (lm @ lm)
    h = rng.standard_normal((9, 81, w)); s = np.einsum("fsw,w->sf", h, lm)
    a = int(s[5].argmax()) + 1; b = 1 if a != 1 else 2
    hf = swap_flip(h, s, 5, a, b, lmu); s2 = np.einsum("fsw,w->sf", hf, lm)
    assert np.isclose(s2[5, b - 1], s[5, a - 1]) and np.isclose(s2[5, a - 1], s[5, b - 1]) and int(s2[5].argmax()) + 1 == b
    mask = np.ones_like(s, bool); mask[5, [a - 1, b - 1]] = False; assert np.allclose(s2[mask], s[mask])
    par, perp = P.project(hf, lm); assert np.allclose(par + perp, hf) and np.allclose(perp @ lm, 0, atol=1e-9)
    # a valid solution grid (pattern construction) and eligibility
    base = np.array([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]).reshape(81)
    puz = base.copy(); puz[rng.permutation(81)[:50]] = 0; grid = base.copy()
    e = np.where(puz == 0)[0]; grid[e[0]] = base[e[0]] % 9 + 1                  # one wrong cell
    el = eligible(grid, base, puz)
    assert all(i in e for c in CLASSES for i, _ in el[c]); assert any(i == e[0] for c in ("HA", "HN") for i, _ in el[c])
    assert all(grid[i] != base[i] for c in ("HA", "HN") for i, _ in el[c]) and all(grid[i] == base[i] for c in ("XB", "XN") for i, _ in el[c])
    for i, bb in el["HA"]: assert bb == base[i] and bb in set(grid[PEERS[i]])
    for i, bb in el["XN"]: assert bb not in set(grid[PEERS[i]]) and bb != base[i]
    sel = select(el, 7, 1); assert sel == select(el, 7, 1) and all(sum(1 for c, _, _ in sel if c == cc) <= PER_CLASS for cc in CLASSES)
    assert ratio_gain([1, 1, 0, 0], [0, 0, 0, 0]) == 0.5 and ratio_loss([0, 1, 1, 1], [1, 1, 1, 1]) == 0.25
    assert letter_r1(0.6, 0.05) == "SELECTIVE" and letter_r1(0.6, 0.6) == "FOLLOWER" and letter_r1(0.05, 0.02) == "DISPLAY-INERT" and letter_r1(0.3, 0.3) == "MIXED"
    assert letter_r2(0.25, 0.0) == "KERNEL-CONTROLS" and letter_r2(0.03, -0.04) == "KERNEL-NEUTRAL" and letter_r2(0.1, 0.0) == "MIXED"
    assert letter_r3(0.4, 0.8, 40) == "COORDINATES" and letter_r3(0.02, 0.5, 40) == "LOCAL" and letter_r3(0.4, 0.8, 10) == "UNDEFINED" and letter_r3(0.4, 0.6, 40) == "MIXED"
    d = discovery_ids(); assert len(d) == 256 and d == discovery_ids()
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--width", type=int, choices=WIDTHS); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest: selftest(); return
    out = Path(a.out) if a.out else OUT
    if a.action == "run":
        if a.width is None: ap.error("--width required")
        R7(a.width, out).run(a.smoke)
    elif a.action == "report": report(out)
    else: ap.print_help()


if __name__ == "__main__":
    main()
