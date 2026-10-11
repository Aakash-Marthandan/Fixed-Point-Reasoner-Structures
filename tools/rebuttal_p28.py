#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P28 — do score-preserving transplants between unsuccessful histories create complete solutions, and
# do they use computation better than the equal-work alternatives? (registration Documentation/Note_2026-10-11_Rebuttal_P28_Registration.md,
# written before any row). MEASUREMENT, $0, inference only on the banked attention checkpoints, through the network's own loop. Reuses P27's
# runner (model loading, release step), P27's controls and P26's frozen bases unchanged.
"""P28, a local pilot per receiver (192 primary, 256 replication). Phase 1 screens benchmark cold failures in a frozen rating-interleaved order:
a four-history bank (fixed start + three Gaussian starts, 16 iterations) per candidate; a puzzle is eligible when no bank readout verifies
(givens kept, every row / column / box a permutation of 1-9). Phase 2, for up to 128 eligible puzzles: at iteration 16, for all 12 ordered
recipient-donor pairs, the leading transplant h + (h_d - h) U U^T, its matched control, the full readout-invisible transplant and its control
(16 iterations each); twelve fresh restarts (16 iterations each); the four originals continued 48 iterations (the first 16 = the paired unchanged
reference). Equal model work: 192 iterations per strategy. Success = the first verified readout in the window (cross-checked against the
reference answer); final-iteration correctness and losses recorded. Sharded by blocks of 32 puzzles, each block checkpointed."""
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p1p2 as P
import rebuttal_p17 as R17
import rebuttal_p25 as P25
import rebuttal_p26 as T
import rebuttal_p27 as P27

OUT = ROOT / "runs/analysis/rebuttal_20261014"
SEED = 20261014
WIDTHS = (192, 256)
N_TARGET, MAX_SCREEN, SCREEN_CHUNK, BLOCK = 128, 2048, 128, 32
T_BANK, T_ARM, T_EXT_SEG, N_EXT_SEG = 16, 16, 16, 3
BANK_J, RESTART_J = (1, 2, 3), tuple(range(101, 113))
PAIRS = tuple((i, j) for i in range(4) for j in range(4) if i != j)                     # the 12 ordered (recipient, donor) pairs
ARMS = ("lead", "lead_ctrl", "full", "full_ctrl")
BUDGETS = (1, 2, 4, 8, 12)
VB, N_BOOT = T.VB, 4000


def utc(): return datetime.now(timezone.utc).isoformat()


# ---------------- pure helpers (selftested) ----------------
def candidates(width):
    """Benchmark cold failures at 16 iterations minus every earlier pool, in the frozen screening order: the failures' own rating octiles,
    a seeded permutation inside each, interleaved round-robin across octiles."""
    import rebuttal_p12 as P12, rebuttal_p20 as R20, rebuttal_p22 as R22
    rec = np.load(ROOT / f"paper/code/evidence/benchmark/attention_{width}_d16.npz", allow_pickle=False)
    r128 = np.load(ROOT / "paper/code/evidence/benchmark/attention_128_d16.npz", allow_pickle=False)
    excl = P25.study_ids() | set(int(i) for i in P12.pool_ids(r128["cold_exact"], r128["idx"])) | set(int(i) for i in R17.pool(width)) \
        | set(int(i) for i in R20.pool(width)) | set(int(i) for i in R22.pool(width)) | set(int(i) for i in P25.pool(width)) | set(int(i) for i in T.pool(width)) \
        | set(int(i) for i in P27.pool(width))
    fail = ~rec["cold_exact"].astype(bool); idx, rating = rec["idx"][fail], rec["rating"][fail]
    keep = np.array([int(i) not in excl for i in idx]); idx, rating = idx[keep], rating[keep]
    order = np.lexsort((idx, rating)); bins = np.empty(len(idx), int); bins[order] = (np.arange(len(idx)) * 8) // len(idx)
    per = [list(np.random.default_rng([SEED, 28, width, b]).permutation(np.sort(idx[bins == b]))) for b in range(8)]
    out = [int(per[b][k]) for k in range(max(len(p) for p in per)) for b in range(8) if k < len(per[b])]
    return np.array(out, np.int64)


def verify(grids, puz):
    """grids (N, T, 81) digits 1-9, puz (N, 81) with 0 = blank -> (N, T) bool: givens kept and every row, column and box a permutation of 1-9."""
    g = np.asarray(grids).astype(np.int64); N, Tn = g.shape[:2]
    giv = puz[:, None, :] > 0; ok_g = np.where(giv, g == puz[:, None, :], True).all(-1)
    oh = (g.reshape(N, Tn, 9, 9)[..., None] == np.arange(1, 10))                              # (N, T, r, c, d)
    rows = (oh.sum(3) == 1).all((-1, -2)); cols = (oh.sum(2) == 1).all((-1, -2))
    box = (oh.reshape(N, Tn, 3, 3, 3, 3, 9).sum((3, 5)) == 1).all((-1, -2, -3))
    return ok_g & rows & cols & box


def first_hit(valid):
    """(N, T) bool -> (N,) first True index or -1."""
    any_ = valid.any(1); return np.where(any_, valid.argmax(1), -1)


def pair_order(pid):
    """The outcome-independent candidate order of the 12 ordered pairs for a puzzle (frozen seed)."""
    return np.random.default_rng([SEED, int(pid), 3]).permutation(len(PAIRS))


# ---------------- the runner ----------------
class R22(P27.R21):
    def __init__(self, width, out=OUT, tag="main"):
        super().__init__(width, out / "_meta" / tag)                    # runtime / status files per process: concurrent shards never share them
        self.meta.update(study="rebuttal_20261014", pipeline="tools/rebuttal_p28.py over tools/rebuttal_p27.py", tag=tag)
        self.ATS.write_json(self.dir / "runtime.json", self.meta)
        self.data = out / f"attention_{width}"; self.data.mkdir(parents=True, exist_ok=True)

    def puzzles(self, pids):
        with np.load(R17.DATA, allow_pickle=False) as d: return d["test_q"][pids].astype(np.int32), d["test_a"][pids].astype(np.int32).reshape(-1, 81)

    def loop(self, z, y, x, steps, first=False):
        """The network's own loop through the release step, padded to VB rows: carry z <- z + eta_z (z_f - z), canvas y <- y + eta (softmax - y).
        Returns displays (B, steps, 81) int8 and the final (z, y) as numpy."""
        jax, jnp, EV = self.jax, self.jnp, self.EV; B = len(x); pad = VB - B
        xp = np.concatenate([x, np.repeat(x[:1], pad, 0)]) if pad else x
        yp = np.concatenate([y, np.repeat(y[:1], pad, 0)]) if pad else y
        zj = None if first else jnp.asarray(np.concatenate([z, np.repeat(z[:1], pad, 0)]) if pad else z)
        xj, yj = jnp.asarray(xp), jnp.asarray(yp); disp = []
        for t in range(steps):
            f = first and t == 0
            lg, zf = EV._step(self.cfg, 1., 0., f)(self.params, xj, yj, self.tv, jnp.zeros(1) if f else zj)
            zj = zf if f else zj + self.eta_z * (zf - zj); yj = yj + self.eta * (jax.nn.softmax(lg, axis=-1).transpose(0, 3, 1, 2) - yj)
            disp.append(np.asarray(EV.layout_gather(lg, self.layout), np.float32)[..., 1:10].reshape(VB, 81, 9).argmax(-1) + 1)
        return np.stack(disp, 1)[:B].astype(np.int8), np.asarray(zj)[:B], np.asarray(yj)[:B]

    def void_canvas(self, n):
        jax, jnp = self.jax, self.jnp; v = jax.nn.one_hot(jnp.full((9, 9), self.G.VOID, jnp.int32), 11).transpose(2, 0, 1); return np.asarray(jnp.broadcast_to(v, (n,) + v.shape))

    def bank(self, pids):
        """Four histories per puzzle (fixed start, Gaussian starts j = 1, 2, 3) for 16 iterations: displays (P, 4, 16, 81), states at 16.
        Chunks of up to 42 puzzles: the fixed starts in one padded batch, the three Gaussian starts packed in another."""
        puz, sol = self.puzzles(pids); x = np.asarray(self.EV.place_batch(puz, self.layout)); P_ = len(pids)
        D = np.zeros((P_, 4, T_BANK, 81), np.int8); Zc, Yc = [], []
        for b0 in range(0, P_, 42):
            sl = slice(b0, b0 + 42); xb = x[sl]; nb = len(xb)
            d0, z0f, y0f = self.loop(None, self.void_canvas(nb), xb, T_BANK, first=True); D[sl, 0] = d0; shp = z0f.shape[1:]
            zg = np.stack([self.EV.mi_z0(SEED, int(p_), j, shp, 1.0, "gauss") for j in BANK_J for p_ in pids[sl]]).astype(np.float32)
            dg, zgf, ygf = self.loop(zg, self.void_canvas(3 * nb), np.concatenate([xb] * 3), T_BANK)
            for k in range(3): D[sl, k + 1] = dg[k * nb:(k + 1) * nb]
            Zc.append(np.stack([z0f] + [zgf[k * nb:(k + 1) * nb] for k in range(3)], 1)); Yc.append(np.stack([y0f] + [ygf[k * nb:(k + 1) * nb] for k in range(3)], 1))
        puz81 = puz.reshape(P_, 81); valid = verify(D.reshape(P_ * 4, T_BANK, 81), np.repeat(puz81, 4, 0)).reshape(P_, 4, T_BANK)
        exact = (D == sol[:, None, None, :]).all(-1)
        return dict(puz=puz81, sol=sol, x=x, disp=D, valid=valid, exact=exact, z=np.concatenate(Zc), y=np.concatenate(Yc))

    def screen(self, smoke=False, shard=(0, 1)):
        """Phase 1, sharded and resumable: chunk c (32 candidates in the frozen order) belongs to shard c mod n. Before each chunk the shard
        counts the eligible puzzles in the contiguous prefix of finished chunks; it stops once that prefix holds the target, so the first
        N_TARGET eligible puzzles are exactly those a sequential screen would find. Each chunk is written atomically."""
        cand = candidates(self.width); chunk, (n_target, n_max) = (32, (3, 64)) if smoke else (SCREEN_CHUNK, (N_TARGET, MAX_SCREEN))
        cdir = self.data / ("screen_chunks_smoke" if smoke else "screen_chunks"); cdir.mkdir(exist_ok=True); s_, n_ = shard
        for c in range(s_, n_max // chunk, n_):
            if prefix_eligible(cdir, c) >= n_target: break
            f = cdir / f"chunk_{c:03d}.json"
            if f.exists(): continue
            pids = cand[c * chunk:(c + 1) * chunk]; js = self.screen_chunk(pids); js.update(chunk=c, created=utc()); elig = np.array([r["eligible"] for r in js["rows"]])
            tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(js)); os.replace(tmp, f)
            print(json.dumps(dict(phase="screen", width=self.width, shard=f"{s_}/{n_}", chunk=c, eligible_in_chunk=int(elig.sum()), at=utc())), flush=True)

    def screen_chunk(self, pids):
        """Eligibility in two stages with the same answer as the full bank: the fixed start for every candidate; the three Gaussian starts only
        for candidates whose fixed start never verifies (a candidate is ineligible as soon as any history verifies)."""
        puz, sol = self.puzzles(pids); x = np.asarray(self.EV.place_batch(puz, self.layout)); n = len(pids); puz81 = puz.reshape(n, 81)
        D0 = np.zeros((n, T_BANK, 81), np.int8); shp = None
        for b0 in range(0, n, VB):
            d0, z0f, _ = self.loop(None, self.void_canvas(len(x[b0:b0 + VB])), x[b0:b0 + VB], T_BANK, first=True); D0[b0:b0 + VB] = d0; shp = z0f.shape[1:]
        v0 = verify(D0, puz81); mism = int((v0 != (D0 == sol[:, None, :]).all(-1)).sum()); fh = [[int(first_hit(v0[k:k + 1])[0]), None, None, None] for k in range(n)]
        rest = [k for k in range(n) if not v0[k].any()]; elig = np.zeros(n, bool)
        if rest:
            jobs = [(k, jj) for jj in range(3) for k in rest]; Dg = np.zeros((len(jobs), T_BANK, 81), np.int8)
            for b0 in range(0, len(jobs), VB):
                jb = jobs[b0:b0 + VB]; zg = np.stack([self.EV.mi_z0(SEED, int(pids[k]), BANK_J[jj], shp, 1.0, "gauss") for k, jj in jb]).astype(np.float32)
                dg, _, _ = self.loop(zg, self.void_canvas(len(jb)), np.stack([x[k] for k, _ in jb]), T_BANK); Dg[b0:b0 + VB] = dg
            vg = verify(Dg, np.stack([puz81[k] for k, _ in jobs])); mism += int((vg != (Dg == np.stack([sol[k] for k, _ in jobs])[:, None, :]).all(-1)).sum())
            for q, (k, jj) in enumerate(jobs): fh[k][jj + 1] = int(first_hit(vg[q:q + 1])[0])
            for k in rest: elig[k] = all(fh[k][jj + 1] == -1 for jj in range(3))
        return dict(rows=[dict(pid=int(p_), eligible=bool(elig[k]), first_hit=fh[k]) for k, p_ in enumerate(pids)], verify_exact_mismatch=mism)

    def transplant(self, bk, pid, p, i, j, U):
        """The four arm states for recipient i <- donor j of puzzle p (slow state only; fast state the recipient's), and descriptives."""
        lm = self.lm; hr = bk["z"][p, i][0].astype(np.float64); hd = bk["z"][p, j][0].astype(np.float64); shp = hr.shape
        par_r, perp_r = P.project(hr, lm); perp_d = P.project(hd, lm)[1]; D = perp_d - perp_r; L = (D @ U) @ U.T; Mr = D - L; fl = lambda a: a.reshape(-1, shp[-1])
        xiL, _ = P27.matched_control(fl(hr), fl(L), P27.proj_lead(U), np.random.default_rng([SEED, int(pid), 8, i, j]))
        xiM, _ = P27.matched_control(fl(hr), fl(Mr), P27.proj_rem(U, lm), np.random.default_rng([SEED, int(pid), 9, i, j]))
        xiL, xiM = xiL.reshape(shp), xiM.reshape(shp); nh = np.linalg.norm(hr); s0 = np.einsum("fsw,w->sf", hr, lm)
        info = dict(p=p, i=i, j=j, energy_lead=float((L ** 2).sum() / max((D ** 2).sum(), 1e-30)), mag_lead=float(np.linalg.norm(L) / nh), mag_full=float(np.linalg.norm(D) / nh),
                    score_shift=max(float(np.abs(np.einsum("fsw,w->sf", hr + C, lm) - s0).max()) for C in (L, D, xiL, xiL + xiM)),
                    ctrl_norm_dev=float(np.abs(np.linalg.norm(fl(xiL), axis=-1) - np.linalg.norm(fl(L), axis=-1)).max()),
                    ctrl_ip_dev=float(np.abs((fl(hr) * fl(xiL)).sum(-1) - (fl(hr) * fl(L)).sum(-1)).max()))
        zl = bk["z"][p, i][1]; f32 = lambda h: np.stack([h.astype(np.float32), zl])
        return dict(lead=f32(hr + L), lead_ctrl=f32(hr + xiL), full=f32(hr + D), full_ctrl=f32(hr + xiL + xiM)), info

    def arms_block(self, pids, U):
        """Phase 2 for one block of eligible puzzles: per-row outcomes for every strategy (row states built lazily per batch)."""
        import time
        t_all = time.time(); t_build = 0.0; bk = self.bank(np.asarray(pids, np.int64)); P_ = len(pids)
        assert not bk["valid"].any(), "gate: a phase-2 bank verified a readout (phase 1 said eligible)"
        specs = []
        for p in range(P_):
            specs += [("ext1", p, i, -1) for i in range(4)] + [(a, p, i, j) for (i, j) in PAIRS for a in ARMS] + [("restart", p, k, jj) for k, jj in enumerate(RESTART_J)]
        specs += [("restore", 0, i, -1) for i in range(4)]                                                   # saved copies of the first puzzle's bank states
        shp = bk["z"].shape[2:]; vc1 = self.void_canvas(1)[0]; saved = {i: (bk["z"][0, i].copy(), bk["y"][0, i].copy()) for i in range(4)}
        n = len(specs); disp = np.zeros((n, T_ARM, 81), np.int8); zend, yend = {}, {}; infos = {}
        for c0 in range(0, n, VB):
            sl = list(range(c0, min(n, c0 + VB))); tb = time.time(); cache = {}; Z, Y, X = [], [], []
            for q in sl:
                k_, p, i, j = specs[q]
                if k_ == "ext1": z, y = bk["z"][p, i], bk["y"][p, i]
                elif k_ == "restore": z, y = saved[i]
                elif k_ == "restart": z, y = self.EV.mi_z0(SEED, int(pids[p]), j, shp, 1.0, "gauss").astype(np.float32), vc1
                else:
                    if (p, i, j) not in cache:
                        cache[(p, i, j)], inf = self.transplant(bk, pids[p], p, i, j, U); infos[(p, i, j)] = inf
                    z, y = cache[(p, i, j)][k_], bk["y"][p, i]
                Z.append(z); Y.append(y); X.append(bk["x"][p])
            t_build += time.time() - tb
            d_, z_, y_ = self.loop(np.stack(Z), np.stack(Y), np.stack(X), T_ARM); disp[sl] = d_
            for k2, q in enumerate(sl):
                if specs[q][0] == "ext1": zend[q], yend[q] = z_[k2], y_[k2]
        ext_idx = [q for q in range(n) if specs[q][0] == "ext1"]; ext_disp = [disp[ext_idx]]
        ze = np.stack([zend[q] for q in ext_idx]); ye = np.stack([yend[q] for q in ext_idx]); xe = np.stack([bk["x"][specs[q][1]] for q in ext_idx])
        for seg in range(1, N_EXT_SEG):                                                                     # extended segments 2 and 3 (iterations 33-64)
            dseg = np.zeros((len(ext_idx), T_EXT_SEG, 81), np.int8); zn, yn = [], []
            for c0 in range(0, len(ext_idx), VB):
                d_, z_, y_ = self.loop(ze[c0:c0 + VB], ye[c0:c0 + VB], xe[c0:c0 + VB], T_EXT_SEG); dseg[c0:c0 + VB] = d_; zn.append(z_); yn.append(y_)
            ext_disp.append(dseg); ze, ye = np.concatenate(zn), np.concatenate(yn)
        rst = [q for q in range(n) if specs[q][0] == "restore"]; restore_ok = all(np.array_equal(disp[a], disp[b]) for a, b in zip(rst, ext_idx[:4]))
        assert restore_ok, "gate: restored bank states did not reproduce the unchanged continuation bitwise"
        keepq = [q for q in range(n) if specs[q][0] != "restore"]
        puz_rows = np.stack([bk["puz"][specs[q][1]] for q in keepq]); sol_rows = np.stack([bk["sol"][specs[q][1]] for q in keepq])
        dk = disp[keepq]; valid = verify(dk, puz_rows); exact = (dk == sol_rows[:, None, :]).all(-1)
        ED = np.concatenate(ext_disp, 1); pe = np.stack([bk["puz"][specs[q][1]] for q in ext_idx]); se = np.stack([bk["sol"][specs[q][1]] for q in ext_idx])
        ev = verify(ED, pe); ee = (ED == se[:, None, :]).all(-1)
        return dict(kind=np.array([specs[q][0] for q in keepq]), p=np.array([specs[q][1] for q in keepq], np.int32), i=np.array([specs[q][2] for q in keepq], np.int32),
                    j=np.array([specs[q][3] for q in keepq], np.int32), first=first_hit(valid).astype(np.int16), final_valid=valid[:, -1], mism=int((valid != exact).sum()),
                    ext_first=first_hit(ev).astype(np.int16), ext_final_valid=ev[:, -1], ext_mism=int((ev != ee).sum()), ext_p=np.array([specs[q][1] for q in ext_idx], np.int32),
                    ext_i=np.array([specs[q][2] for q in ext_idx], np.int32), info=json.dumps([infos[k_] for k_ in sorted(infos)]), restore_ok=restore_ok,
                    seconds_total=time.time() - t_all, seconds_transplant_build=t_build)

    def run(self, phase, shard=(0, 1), smoke=False):
        try:
            if phase == "screen": self.screen(smoke, shard); return
            js = json.loads((self.data / ("screen_smoke.json" if smoke else "screen.json")).read_text()); elig = js["eligible"]; U = P27.load_basis(self.width, self.lm)
            blocks = [elig[b0:b0 + (BLOCK if not smoke else 2)] for b0 in range(0, len(elig), BLOCK if not smoke else 2)]; s, n = shard
            for b, pids in enumerate(blocks):
                if b % n != s: continue
                dst = self.data / f"block{'_smoke' if smoke else ''}_{b:02d}.npz"
                if dst.exists(): print(json.dumps(dict(block=b, status="exists, skipped")), flush=True); continue
                res = self.arms_block(pids, U); tmp = dst.with_suffix(".tmp.npz"); np.savez_compressed(tmp, pids=np.array(pids, np.int64), **res); tmp.rename(dst)
                print(json.dumps(dict(phase="arms", width=self.width, block=b, of=len(blocks), puzzles=len(pids), rows=int(len(res["kind"])), mism=res["mism"] + res["ext_mism"], at=utc())), flush=True)
            self.status("inference_complete", completed=utc())
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


def prefix_eligible(cdir, c):
    """Eligible count over chunks 0..c-1 if all are finished, else over the finished contiguous prefix (a lower bound: never stops early)."""
    tot = 0
    for k in range(c):
        f = cdir / f"chunk_{k:03d}.json"
        if not f.exists(): break
        tot += sum(r["eligible"] for r in json.loads(f.read_text())["rows"])
    return tot


def finalize_screen(width, base=OUT, smoke=False):
    """Assemble screen.json from the chunk files: the contiguous prefix, cut at the chunk where the target is reached (= a sequential screen)."""
    d = base / f"attention_{width}"; cdir = d / ("screen_chunks_smoke" if smoke else "screen_chunks"); n_target, chunk_max = (3, 2) if smoke else (N_TARGET, MAX_SCREEN // SCREEN_CHUNK)
    rows, used, extra = [], 0, 0
    for k in range(chunk_max):
        f = cdir / f"chunk_{k:03d}.json"
        if not f.exists(): break
        if sum(r["eligible"] for r in rows) >= n_target: extra += 1; continue
        js = json.loads(f.read_text()); rows += [dict(r, mism=js["verify_exact_mismatch"] if i_ == 0 else 0) for i_, r in enumerate(js["rows"])]; used += 1
    extra += sum(1 for f in cdir.glob("chunk_*.json") if int(f.stem.split("_")[1]) >= used + extra)
    elig = [r["pid"] for r in rows if r["eligible"]][:n_target]
    out = dict(created=utc(), width=width, smoke=smoke, screened=len(rows), chunks_used=used, extra_chunks_screened=extra, eligible=elig, rows=rows,
               order_sha256=__import__("hashlib").sha256(candidates(width).tobytes()).hexdigest())
    (d / ("screen_smoke.json" if smoke else "screen.json")).write_text(json.dumps(out)); print(f"attention_{width}: screened {len(rows)} in {used} chunks (+{extra} extra), eligible {len(elig)}")


def merge(width, base=OUT, smoke=False):
    d = base / f"attention_{width}"; js = json.loads((d / ("screen_smoke.json" if smoke else "screen.json")).read_text()); nb = -(-len(js["eligible"]) // (BLOCK if not smoke else 2))
    parts = [dict(np.load(d / f"block{'_smoke' if smoke else ''}_{b:02d}.npz", allow_pickle=False)) for b in range(nb)]
    secs = []; off, out = 0, {k_: [] for k_ in ("kind", "p", "i", "j", "first", "final_valid", "ext_first", "ext_final_valid", "ext_p", "ext_i", "pids")}; infos = []; mism = 0
    for pt in parts:
        for k_ in ("kind", "i", "j", "first", "final_valid", "ext_first", "ext_final_valid", "ext_i"): out[k_].append(pt[k_])
        out["p"].append(pt["p"] + off); out["ext_p"].append(pt["ext_p"] + off); out["pids"].append(pt["pids"]); infos += [dict(x, p=x["p"] + off) for x in json.loads(str(pt["info"]))]
        mism += int(pt["mism"]) + int(pt["ext_mism"]); off += len(pt["pids"]); secs.append((float(pt["seconds_total"]), float(pt["seconds_transplant_build"])))
    dst = d / ("p28_smoke.npz" if smoke else "p28.npz")
    np.savez_compressed(dst, **{k_: np.concatenate(v) for k_, v in out.items()}, info=json.dumps(infos), mism=mism, screen=json.dumps(js), seconds=np.array(secs))
    if not smoke: (dst.with_suffix(".sha256.json")).write_text(json.dumps(dict(sha256=P25.sha(dst))))
    print(f"attention_{width}: merged {nb} blocks, {off} puzzles -> {dst}")


# ---------------- the frozen reader ----------------
def per_puzzle(d):
    """Per eligible puzzle: success matrices per strategy (windows as registered), the paired unchanged reference, budget coverages."""
    kind, p, i, j, first = d["kind"], d["p"], d["i"], d["j"], d["first"]; nP = int(len(d["pids"])); pids = d["pids"]
    S = {a: np.zeros((nP, 12), bool) for a in ARMS}; R = np.zeros((nP, 12), bool); SH = np.zeros((nP, 4), bool)
    for q in range(len(kind)):
        k_, pp = str(kind[q]), int(p[q]); s_ = first[q] >= 0
        if k_ in ARMS: S[k_][pp, PAIRS.index((int(i[q]), int(j[q])))] = s_
        elif k_ == "restart": R[pp, int(i[q])] = s_
        elif k_ == "ext1": SH[pp, int(i[q])] = s_
    EX = np.full((nP, 4), -1, np.int64)
    for q in range(len(d["ext_p"])): EX[int(d["ext_p"][q]), int(d["ext_i"][q])] = int(d["ext_first"][q])
    out = dict(S=S, R=R, SH=SH, EX=EX, pids=pids)
    order = np.stack([pair_order(pid) for pid in pids])                                                     # (nP, 12)
    cov = {}
    for a in ARMS:
        Sa = np.take_along_axis(S[a], order, 1); cov[a] = {b: Sa[:, :b].any(1) for b in BUDGETS}
    cov["restart"] = {b: R[:, :b].any(1) for b in BUDGETS}
    cov["extended"] = {b: ((EX >= 0) & (EX < 4 * b)).any(1) for b in BUDGETS}                              # b x 16 iterations shared by the four originals
    out["cov"] = cov; out["short"] = SH.any(1)
    out["pri"] = {a: np.array([np.mean([float(S[a][pp, PAIRS.index((ii, jj))]) - float(SH[pp, ii]) for (ii, jj) in PAIRS]) for pp in range(nP)]) for a in ARMS}
    return out


def boot(fn, n_units, seed, n=N_BOOT):
    rng = np.random.default_rng(seed); v = [fn(rng.integers(0, n_units, n_units)) for _ in range(n)]; return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def report_width(width, base=OUT, smoke=False):
    f = base / f"attention_{width}/p28{'_smoke' if smoke else ''}.npz"
    if not f.exists(): return None
    d = np.load(f, allow_pickle=False); pp = per_puzzle(d); nP = len(pp["pids"]); cov = pp["cov"]; sh = pp["short"]; res = dict(eligible=nP, mism=int(d["mism"]))
    sc = json.loads(str(d["screen"])); res["screened"] = sc["screened"]; res["yield"] = nP / max(sc["screened"], 1)
    ci = lambda fn, k: boot(fn, nP, [SEED, 28, width, k]); m = lambda a: float(np.mean(a))
    res["coverage"] = {a: {b: m(cov[a][b]) for b in BUDGETS} for a in cov}; res["short_coverage"] = m(sh)
    for a in ARMS:
        A = cov[a][12]; add = A & ~sh; loss = sh & ~A
        res[f"{a}_additional"] = dict(E=m(add), ci=ci(lambda ix: m(add[ix]), 1), reverse_losses=m(loss), reverse_ci=ci(lambda ix: m(loss[ix]), 2))
        res[f"{a}_per_rollout"] = dict(E=m(pp["pri"][a]), ci=ci(lambda ix: m(pp["pri"][a][ix]), 3))
    for a, ctl in (("lead", "lead_ctrl"), ("full", "full_ctrl")):
        for bname in (ctl, "restart", "extended"):
            dlt = cov[a][12].astype(float) - cov[bname][12].astype(float); disc = int((cov[a][12] != cov[bname][12]).sum())
            res[f"{a}_vs_{bname}"] = dict(E=m(dlt), ci=ci(lambda ix: m(dlt[ix]), 4 + len(bname)), discordant=disc)
        alt1 = np.maximum.reduce([cov["restart"][1], cov["extended"][1], cov[a][1]]).astype(float)
        hd = cov[a][12].astype(float) - alt1; res[f"{a}_headroom"] = dict(E=m(hd), ci=ci(lambda ix: m(hd[ix]), 9), oracle_budget1=m(cov[a][12]))
        uni = cov[a][12] & ~cov["restart"][12] & ~cov["extended"][12]; res[f"{a}_unique"] = dict(E=m(uni), ci=ci(lambda ix: m(uni[ix]), 10))
    L = res; lo = lambda key: L[key]["ci"][0] > 0
    res["letters_lead"] = dict(ADDS_SOLUTIONS=lo("lead_additional"), PER_ROLLOUT_GAIN=lo("lead_per_rollout"), DONOR_SPECIFIC_COVERAGE=lo("lead_vs_lead_ctrl"),
                               EFFICIENT_BANK=lo("lead_vs_restart") and lo("lead_vs_extended"), SELECTION_OPPORTUNITY=lo("lead_headroom"))
    res["secondary_full"] = dict(ADDS_SOLUTIONS=lo("full_additional"), PER_ROLLOUT_GAIN=lo("full_per_rollout"), DONOR_SPECIFIC_COVERAGE=lo("full_vs_full_ctrl"),
                                 EFFICIENT_BANK=lo("full_vs_restart") and lo("full_vs_extended"), SELECTION_OPPORTUNITY=lo("full_headroom"))
    inf = json.loads(str(d["info"])); res["late_state"] = {k_: float(np.mean([x[k_] for x in inf])) for k_ in ("energy_lead", "mag_lead", "mag_full")}
    res["gates"] = dict(score_shift=max(x["score_shift"] for x in inf), ctrl_norm_dev=max(x["ctrl_norm_dev"] for x in inf), ctrl_ip_dev=max(x["ctrl_ip_dev"] for x in inf), verify_exact_mismatch=int(d["mism"]))
    sec = d["seconds"]; res["runtime"] = dict(block_seconds_total=float(sec[:, 0].sum()), transplant_build_seconds=float(sec[:, 1].sum()))
    kind, first, fv = d["kind"], d["first"], d["final_valid"]
    res["preservation"] = {a: dict(successes=int(((kind == a) & (first >= 0)).sum()), kept_at_end=int(((kind == a) & (first >= 0) & fv).sum())) for a in ARMS + ("restart",)}
    ef, efv = d["ext_first"], d["ext_final_valid"]; res["preservation"]["extended"] = dict(successes=int((ef >= 0).sum()), kept_at_end=int(((ef >= 0) & efv).sum()))
    return res


def report(base=OUT, smoke=False, quiet=False):
    lines = ["P28 REPORT (rules registered before any row; primary = Attention 192, leading transplant; 256 = replication; full = secondary)"]; js = {}
    f = lambda x: f"{x['E']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
    for w in WIDTHS:
        r = report_width(w, base, smoke)
        if r is None: lines.append(f"attention_{w}: not run"); continue
        js[f"attention_{w}"] = r; c = r["coverage"]
        lines += [f"attention_{w}: eligible {r['eligible']} of {r['screened']} screened (yield {r['yield']:.2f}) | gates {r['gates']}",
                  "   coverage at budgets " + str(BUDGETS) + ": " + " | ".join(f"{a} " + "/".join(f"{c[a][b]:.3f}" for b in BUDGETS) for a in c) + f" | unchanged short {r['short_coverage']:.3f}"]
        for a in ARMS:
            lines.append(f"   {a}: additional {f(r[a + '_additional'])} (reverse losses {r[a + '_additional']['reverse_losses']:.3f}) | per-rollout {f(r[a + '_per_rollout'])}")
        for a, ctl in (("lead", "lead_ctrl"), ("full", "full_ctrl")):
            lines.append(f"   {a} bank vs {ctl} {f(r[a + '_vs_' + ctl])} (discordant {r[a + '_vs_' + ctl]['discordant']}) | vs restart {f(r[a + '_vs_restart'])} | vs extended {f(r[a + '_vs_extended'])} | headroom {f(r[a + '_headroom'])} | unique {f(r[a + '_unique'])}")
        lines += [f"   LETTERS (leading): {r['letters_lead']}", f"   secondary (full): {r['secondary_full']}", f"   late-state descriptives {r['late_state']} | preservation {r['preservation']}"]
    sfx = "_smoke" if smoke else ""; (base / f"report{sfx}.txt").write_text("\n".join(lines) + "\n"); (base / f"report{sfx}.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(lines) if not quiet else f"report written ({len(lines)} lines; output suppressed)")


def selftest():
    sol = np.array([int(c) for c in "534678912672195348198342567859761423426853791713924856961537284287419635345286179"])
    puz = sol.copy(); puz[::3] = 0
    g = np.stack([sol, sol.copy(), sol.copy(), sol.copy()]); g[1, 1] = g[1, 2]; g[2, 0] = 9; g[3] = sol
    v = verify(g[None], puz[None])[0]; assert v.tolist() == [True, False, False, True]
    p2 = puz.copy(); p2[1] = (sol[1] % 9) + 1; assert not verify(sol[None, None], p2[None])[0, 0]                     # a given changed -> not kept
    assert first_hit(np.array([[False, True, True], [False, False, False]])).tolist() == [1, -1]
    c = candidates(192); assert len(c) == len(set(c.tolist())) and np.array_equal(c, candidates(192))
    assert not (set(c.tolist()) & set(int(i) for i in P27.pool(192))) and not (set(c.tolist()) & set(int(i) for i in T.pool(192)))
    o = pair_order(123); assert sorted(o.tolist()) == list(range(12)) and np.array_equal(o, pair_order(123)) and len(PAIRS) == 12
    # the reader on a synthetic two-puzzle file
    kinds, pp, ii, jj, ff = [], [], [], [], []
    for p in range(2):
        for i in range(4): kinds.append("ext1"); pp.append(p); ii.append(i); jj.append(-1); ff.append(-1)
        for a in ARMS:
            for (i, j) in PAIRS: kinds.append(a); pp.append(p); ii.append(i); jj.append(j); ff.append(3 if (a == "lead" and p == 0 and (i, j) == (0, 1)) else -1)
        for k in range(12): kinds.append("restart"); pp.append(p); ii.append(k); jj.append(101 + k); ff.append(-1)
    d = dict(kind=np.array(kinds), p=np.array(pp), i=np.array(ii), j=np.array(jj), first=np.array(ff), pids=np.array([5, 6]), ext_p=np.repeat([0, 1], 4), ext_i=np.tile(np.arange(4), 2), ext_first=np.array([-1, -1, -1, -1, 20, -1, -1, -1]))
    q = per_puzzle(d); assert q["cov"]["lead"][12].tolist() == [True, False] and q["cov"]["extended"][12].tolist() == [False, True] and q["cov"]["extended"][4].tolist() == [False, False]
    assert abs(q["pri"]["lead"][0] - 1 / 12) < 1e-12 and q["pri"]["lead"][1] == 0
    import tempfile                                                                                         # the sharded screen's prefix rule and finalize cut
    with tempfile.TemporaryDirectory() as tdir:
        base = Path(tdir); cdir = base / "attention_192" / "screen_chunks_smoke"; cdir.mkdir(parents=True)
        mk = lambda c, el: (cdir / f"chunk_{c:03d}.json").write_text(json.dumps(dict(chunk=c, rows=[dict(pid=100 * c + k, eligible=e, first_hit=[-1] * 4) for k, e in enumerate(el)], verify_exact_mismatch=0)))
        mk(1, [True, True, True]); assert prefix_eligible(cdir, 2) == 0                                           # chunk 0 missing: the prefix is empty, never stop early
        mk(0, [True, False, True]); assert prefix_eligible(cdir, 1) == 2 and prefix_eligible(cdir, 2) == 5
        finalize_screen(192, base, smoke=True); js = json.loads((base / "attention_192/screen_smoke.json").read_text())
        assert js["eligible"] == [0, 2, 100] and js["chunks_used"] == 2 and js["screened"] == 6
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["screen", "finalize", "arms", "merge", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--width", type=int, choices=WIDTHS); ap.add_argument("--shard", default="0/1"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = Path(a.out) if a.out else OUT
    if a.selftest: selftest(); return
    if a.action == "report": report(out, a.smoke, a.quiet); return
    if a.width is None: ap.error("--width required")
    if a.action == "merge": merge(a.width, out, a.smoke); return
    if a.action == "finalize": finalize_screen(a.width, out, a.smoke); return
    s, n = (int(v) for v in a.shard.split("/"))
    R22(a.width, out, tag=f"{a.action}{'_smoke' if a.smoke else ''}_{s}of{n}").run("screen" if a.action == "screen" else "arms", (s, n), a.smoke)


if __name__ == "__main__":
    main()
