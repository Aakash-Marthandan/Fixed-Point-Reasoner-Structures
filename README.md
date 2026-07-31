# QHRRN-2 — A Holographic RG Network for ARC-AGI

A physics-grounded architecture for the [Abstraction and Reasoning Corpus](https://github.com/fchollet/ARC-AGI):
an equivariant recursive coarse-graining core in which **every channel that
moves information is priced**. Information flux across renormalization-group
cuts — and, since Amendment D, across attention "wormholes" at every scale —
is measured in nats and penalized, so the optimizer must *buy* exactly the
information each task needs. The result is a per-task, per-scale **information
ledger**: a new measurable for studying abstraction, memorization, and
locality in learned solvers.

**New here? Read in this order:**
1. This file (what & why, 5 minutes)
2. [`README_PHYSICS.md`](README_PHYSICS.md) — the physics inspiration and the
   mathematical tools, in depth
3. [`Documentation/Design_Ledger.md`](Documentation/Design_Ledger.md) — the
   epistemic source of truth: every design element tagged proven / hypothesis
   / refuted, with a dated, append-only evidence log
4. [`Documentation/QHRRN2_Architecture.md`](Documentation/QHRRN2_Architecture.md)
   — the full architecture specification

## The three load-bearing mechanisms

1. **Priced holographic streams.** Each 2×2 coarse-graining step splits its
   input into a *kept* channel (flows up the hierarchy) and a *boundary
   stream* (a variational code, retained and re-injected during decoding at
   the matching scale). Nothing is destroyed; everything transmitted is
   priced via a KL term — the **flux ledger** `I_s`. Identity-like tasks buy
   full fine-scale transmission; abstract tasks buy almost nothing.
2. **Priced attention at every scale** ("wormhole tolls", Amendment D).
   Long-range correspondence is handled by attention whose messages pass
   through their own variational bottleneck with flux `A_s`. The pair
   `(Σ I_s, Σ A_s)` decomposes each task's information demand into
   hierarchical vs nonlocal — the basis of a measurable locality taxonomy.
3. **Exact symmetry + discrete rule selection.** Colors 1–9 form an exact
   S₉-equivariant set axis (weight sharing, not augmentation); translation
   equivariance is native; D₄ handled by orbit voting. A small rule codebook
   is selected by temperature-annealed attention — the entropy of the
   selection distribution is an order parameter, and its collapse during
   per-task training is measurable symmetry breaking.

The model is deliberately tiny (~49k parameters at the toy operating point;
target ≤400k), betting that exact structure replaces brute capacity.

## Current status (2026-07-31) — honest and ledger-governed

- **No performance claims live in this README.** Claims belong to the ledger
  until evaluation day (claims are pre-registered
  with named tests and kill conditions before results exist).
- Architecture complete through Amendment D; **all six pre-cloud CI gates
  pass** (equivariance bit-exactness, anti-linearity/rank, sanity triad via
  fitting, seam-boundary task, flux-direction sanity, canvas/no-GT-size),
  reproduced cross-backend (CPU and TPU).
- **Constructed-family measurement program v1 CLOSED** (frozen dataset, ledger
  2026-07-30): per-family accuracy–flux **envelopes** with monotone frontiers,
  operational area-law spectra, analytic floors (`Documentation/S1_Floors.md`),
  a full locality-class ablation, and ~100× envelope compression via protocol +
  architecture-minimization — the closest floor-to-envelope sandwich stands at
  ~30×. Two early findings were corrected by later data (both artifacts of the
  redundant attention channel); the corrections are ledger entries, as designed.
- **Current phase: the solve-rate sprint** — dev-30 evaluation set (17/30
  tasks verified), then a joint-episodic trainer (shared bulk + per-task
  embeddings over the public training split), first pretrain at d=16, and the
  first real-task solve-rate + flux measurements from the same runs. Generator
  pretraining and full ARC evaluation follow.

## Repository map

```
src/qhrrn2/          the implementation (~1k lines, JAX)
  grid.py            canvas, VOID state, D4 group, S9 palettes, ARC episodes
  cell.py            seam mixers, pool/split streams, priced attention, FiLM
  model.py           encoder → rule codebook → decoder; the flux ledgers
  objective.py       masked CE + β·ΣI_s + β_nl·ΣA_s + size loss
  train.py           LoO-validated fitting (MDL selection), orbit voting
  config.py          every dial, ledger-annotated
tests/               CI gates + unit tests (pytest; all local, fast)
tools/
  run_gates.py       the six pre-cloud gates as runnable protocols
  measure.py         the measurement harness (flux + LoO-gap JSONL rows)
  dev30.py           dev-set manifest + terminal task renderer
  dispatcher.py      TPU session manager (up/run/down; budget-defensive)
  shard_run.sh       chip-parallel sweep launcher
Documentation/       governing documents (see reading order above), incl.
                     Thesis_Information_Holography.md (claims S1–S4 with
                     kill conditions) and S1_Floors.md (analytic floors,
                     gap decomposition, pre-registered analyses)
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone --depth 1 https://github.com/fchollet/ARC-AGI.git data/ARC-AGI  # vendored, git-ignored
.venv/bin/pytest                              # unit tests + fast CI gates
.venv/bin/python tools/run_gates.py --gate triad --steps 600   # slow gates (CPU: ~1h)
.venv/bin/python tools/dev30.py render 1e0a9b12                # look at a task
```

## Research discipline (why the ledger exists)

Three binding rules govern all development: every mechanism cites the ledger IDs it implements; no hypothesis
ships without its named test in the same change; status changes are
append-only with dated evidence. Failed hypotheses are reported results, not
silent deletions — several of this project's findings began as gate
failures.

## Provenance

The theory documents that seeded this project (`Documentation/*.pdf`,
Dec 2025 – Jan 2026) are preserved as-is; the ledger §1 catalogs every
load-bearing claim from them and its current status. Target venue: AAMAS
2027 (submission Oct 2026).
