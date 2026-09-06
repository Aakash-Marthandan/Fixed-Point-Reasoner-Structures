"""I12 / P14: the alphaXiv TRM-MLP weights converted into OUR JAX trm_cell; fp32 logits vs their PyTorch forward on 64 strat puzzles, D=16."""
import sys, json, time, numpy as np, torch
from pathlib import Path
H = Path(__file__).resolve().parent; FC = H.parent; ROOT = FC.parents[1]
sys.path.insert(0, str(H)); sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
from qhrrn2 import trm_cell as TC
from qhrrn2.config import Config
from qhrrn2 import sudoku_extreme as SX
import field_models as FM
sd = torch.load(FC / "ckpts/trm_alphaxiv/step_32550_sudoku_epoch50k", map_location="cpu", weights_only=True)
W = {k.replace("model.inner.", ""): v.float().numpy() for k, v in sd.items()}
cfg = Config(canvas=9, scales=2, pool_arity=3, mixer_kind="group9", attn_max_hw=9, equilibrium=True, sudoku_layout="native9", T=16, eta_fixed=1.0, eta_z_fixed=1.0,
             loss_kind="stablemax", cell_kind="trm", trm_hidden=512, trm_layers=2, trm_h_cycles=3, trm_l_cycles=6, trm_puzzle_emb_len=16, trm_lambda=0.0, trm_beta=0.0)
p = TC.init_params(jax.random.PRNGKey(0), cfg, hw=81)
def conv():
    tok = np.zeros((11, 512), np.float32); tok[:10] = W["embed_tokens.embedding_weight"][1:11]; tok[10] = W["embed_tokens.embedding_weight"][0]
    lm = np.zeros((512, 11), np.float32); lm[:, :10] = W["lm_head.weight"][1:11].T; lm[:, 10] = W["lm_head.weight"][0]
    pe = np.zeros((16, 512), np.float32); pe[0] = W["puzzle_emb.weights"][0]
    q = {"w": jnp.asarray(W["q_head.weight"].T), "b": jnp.asarray(W["q_head.bias"])}
    blocks = [{"mlp_t": {"gate_up": jnp.asarray(W[f"L_level.layers.{i}.mlp_t.gate_up_proj.weight"].T), "down": jnp.asarray(W[f"L_level.layers.{i}.mlp_t.down_proj.weight"].T)},
               "mlp": {"gate_up": jnp.asarray(W[f"L_level.layers.{i}.mlp.gate_up_proj.weight"].T), "down": jnp.asarray(W[f"L_level.layers.{i}.mlp.down_proj.weight"].T)}} for i in range(2)]
    return {"tok_emb": jnp.asarray(tok), "puzzle_emb": jnp.asarray(pe), "lm_head": jnp.asarray(lm), "q_head": q, "blocks": blocks}
pj = conv()
assert jax.tree.structure(pj) == jax.tree.structure(p), "pytree structure mismatch"
for a_, b_ in zip(jax.tree.leaves(pj), jax.tree.leaves(p)): assert a_.shape == b_.shape, (a_.shape, b_.shape)
H0, L0 = jnp.asarray(W["H_init"]), jnp.asarray(W["L_init"]); TC.init_states = lambda cfg_: (H0, L0)
d = SX.load_prepared(ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"); ids = SX.stratified_subsample(d["test_rating"], 256, 20260821)[:64]
puz = d["test_q"][ids].astype(np.int32); sol = d["test_a"][ids].reshape(64, 81).astype(np.int64)
m = FM.load("trm", "cpu", torch.float32); batch = m.tokens(puz.astype(np.int64)); st = m.init_state(64, "fixed")
seg = jax.jit(lambda p_, emb, zH, zL: TC.segment(p_, cfg, emb, zH, zL)); emb = jax.vmap(lambda x: TC.embed(pj, cfg, x))(jnp.asarray(puz))
z = jnp.broadcast_to(TC.z0(cfg, 81), (64,) + tuple(TC.carry_shape(cfg, 81))); zH, zL = z[:, 0], z[:, 1]
rows = []; t0 = time.time()
for t in range(16):
    zH, zL = jax.vmap(lambda e, h, l: seg(pj, e, h, l))(emb, zH, zL); lg_j, q_j = jax.vmap(lambda h: TC.readout(pj, cfg, h, (9, 9)))(zH)
    st, lg_t, q_t = m.step(st, batch)
    a9 = np.asarray(lg_j)[..., 1:10].reshape(64, 81, 9); b9 = m.logits9(lg_t).numpy()
    ex_j = (a9.argmax(-1) + 1 == sol).all(1); ex_t = (b9.argmax(-1) + 1 == sol).all(1)
    rows.append(dict(t=t + 1, max_abs=float(np.abs(a9 - b9).max()), mean_abs=float(np.abs(a9 - b9).mean()), scale=float(np.abs(b9).mean()), argmax_agree=float((a9.argmax(-1) == b9.argmax(-1)).mean()),
                     exact_jax=float(ex_j.mean()), exact_torch=float(ex_t.mean()), exact_agree=float((ex_j == ex_t).mean()), q_max_abs=float(np.abs(np.asarray(q_j)[:, 0] - q_t.numpy()).max())))
    print(json.dumps(rows[-1]), flush=True)
out = FC / "out" / "verify_port"; out.mkdir(parents=True, exist_ok=True); json.dump(dict(rows=rows, wall_s=time.time() - t0), open(out / "summary.json", "w"), indent=1); print("DONE", time.time() - t0)
