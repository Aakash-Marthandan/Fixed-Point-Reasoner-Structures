"""Exploratory: do the P15-runner rollouts (roll_release: canvas held at void) reproduce the network's own loop (canvas y <- softmax(logits),
eta = 1) for unedited states? On the first 128 P27-pool puzzles per width, from the fixed-start and the Gaussian-start state after iteration 1,
compare displayed grids at k = 1..4 after that state."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p27 as T27, rebuttal_p17 as R17
for w in (192, 256):
    r = T27.R21(w, Path("/private/tmp/claude-501/-Users-aakash-Projects-HRRN/29fa6425-c970-40e0-b70e-5aa7c3c811d3/scratchpad/canvas_probe")); jax, jnp, EV = r.jax, r.jnp, r.EV
    ids = T27.pool(w)[:128]
    with np.load(R17.DATA, allow_pickle=False) as d: puz, sol = d["test_q"][ids].astype(np.int32), d["test_a"][ids].astype(np.int32).reshape(-1, 81)
    x = EV.place_batch(puz, r.layout); void = jax.nn.one_hot(jnp.full((9, 9), r.G.VOID, jnp.int32), 11).transpose(2, 0, 1)
    disp = lambda lg: np.asarray(EV.layout_gather(lg, r.layout), np.float32)[..., 1:10].reshape(len(ids), 81, 9).argmax(-1) + 1
    def natural(z0, steps):
        y = jnp.broadcast_to(void, (len(ids),) + void.shape); z = z0; out, zs = [], []
        for t in range(steps):
            first = z is None
            lg, zf = EV._step(r.cfg, 1., 0., first)(r.params, x, y, r.tv, jnp.zeros(1) if first else z)
            z = zf if first else z + r.eta_z * (zf - z); y = y + r.eta * (jax.nn.softmax(lg, axis=-1).transpose(0, 3, 1, 2) - y); out.append(disp(lg)); zs.append(np.asarray(z))
        return out, zs
    z0g = jnp.asarray(np.stack([EV.mi_z0(T27.SEED, int(p), 7, (2, 9, 81, w), 1.0, "gauss") for p in ids]).astype(np.float32))
    for name, z0 in (("fixed start", None), ("Gaussian start", z0g)):
        nat, zs = natural(z0, 5)                                   # step 1 makes the state after iteration 1; steps 2-5 are k = 1..4 after it
        vc = r.roll_release(zs[0], np.asarray(x)).argmax(-1) + 1    # P15-runner rollout from the same state, canvas held at void
        cells = [float((vc[:, k] == nat[k + 1]).mean()) for k in range(4)]; grids = [float((vc[:, k] == nat[k + 1]).all(1).mean()) for k in range(4)]
        ex_n = [float((nat[k + 1] == sol).all(1).mean()) for k in range(4)]; ex_v = [float((vc[:, k] == sol).all(1).mean()) for k in range(4)]
        print(f"attention_{w} {name}: cell agreement k1-4 {np.round(cells, 4).tolist()} | identical grids {np.round(grids, 3).tolist()} | exact natural {np.round(ex_n, 3).tolist()} vs void {np.round(ex_v, 3).tolist()}")
