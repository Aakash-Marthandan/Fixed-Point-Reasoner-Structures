# QHRRN-2 — A Holographic RG Network for ARC-AGI

Physics-grounded architecture for the Abstraction and Reasoning Corpus: an
equivariant recursive coarse-graining core with **priced boundary streams**
(information flux across RG cuts is measured and penalized), a discrete rule
codebook selected by annealed attention, and test-time training that adapts
only boundary parameters.

**Status: rebuild in progress.** No performance claims live in this README —
claims belong to the ledger (see below) until evaluation day.

## Current status (2026-07-27) & next steps

Done: post-mortem of the April system (measured, `d874427` preserved) · QHRRN-2 core
implemented at d=12 (45,509 params, 25 tests green incl. CI-1/CI-2 + C14 gates) ·
**Amendment D implemented** (ledger C14: KL-priced attention at all scales,
"wormhole tolls" — the `(I_local, A_nonlocal)` decomposition S3 needs is now a
per-task measurable; ledger §5, 2026-07-27) · **CI-3a 3/3 PASSED** (first full
triad pass, 2026-07-27, under C14; color-swap exact via orbit voting; the old
translate one-pixel miss did not recur — seed-ensemble remedy not needed,
kept queued for CI-3b) ·
**thesis narrowed to four kill-conditioned statements S1–S4**
(`Documentation/Thesis_Information_Holography.md` §6; ledger §3c) with two
load-bearing citations verified (RT-on-trees; CompressARC 76k/20% baseline).
Note: all-scales attention raises CPU gate cost to ~2 s/step (~20 min per
600-step task fit); fine on TPU.

**GATE BATTERY: 6/6 (2026-07-28)** — CI-1/2 (pytest, permanent), CI-3a
(CPU + TPU), CI-4/5/6 (TPU, ledger entry with numbers). Cloud training
spend is unlocked per the build discipline.

Next, in order:
1. Phase 2 measurements (`tools/measure.py`): β-frontier sweep on the five
   constructed families (S1 sandwich data), LoO-gap logging (S2), and the
   S3 stability check — including the measured attention-copy degeneracy
   (identity pays A₀ > I₀; ledger 2026-07-28). Then assemble dev-30
   (gate: Aug 31).
3. Phase 3: RE-ARC-style pretraining on TPU (GCP project `quantum-llm` is
   fully configured; session-persistent dispatcher `up`/`run`/`down` with
   always-armed dead-man's-switch backstop, spot default; unattended
   `cycle` mode for pretraining), then CI-3b (triad under frozen-core TTT)
   with seed-ensemble voting.
4. Also owed (cheap, paper hygiene): mechanical re-verification pass of the
   remaining thesis citations; S1 proof-obligation write-up (thesis §2).

Deadline: results freeze **Sep 28, 2026**; AAMAS 2027 submission early October.
Working constraint: prefer conscious sequential work over agent fleets
(Max-plan 5-hour token windows). Commits are **local-only** — push to
`origin/master` when ready to sync GitHub.

## The three governing documents

| Document | Role |
|---|---|
| [`Documentation/Design_Ledger.md`](Documentation/Design_Ledger.md) | **Epistemic source of truth.** Every design element is tagged proven / hypothesis / refuted, with tests and an append-only status log. Read this first. |
| [`Documentation/QHRRN2_Architecture.md`](Documentation/QHRRN2_Architecture.md) | The architecture spec (v0.2), with the expressivity audit. |
| [`Documentation/Divergence_Analysis_2026-07.md`](Documentation/Divergence_Analysis_2026-07.md) | Post-mortem of the April 2026 system (preserved at commit `d874427`) — the measured failures this design answers. |

## Build discipline

1. Every module header cites the ledger IDs it implements.
2. No hypothesis-tagged mechanism ships without its named test in the same change.
3. CI gates must PASS before any cloud **training** spend:
   equivariance bit-exactness · anti-linearity/rank · sanity triad via TTT ·
   seam-boundary task · flux-direction sanity · canvas/no-GT-size.
   (Amended 2026-07-27, PI-authorized: the gate battery itself may execute on
   a ≤$5 spot instance when local hardware limits — two thermal incidents in
   one day; fast tests stay local.)

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone --depth 1 https://github.com/fchollet/ARC-AGI.git data/ARC-AGI  # vendored, git-ignored
.venv/bin/pytest                                                          # run the gates
```

Layout: `src/qhrrn2/` (implementation) · `tests/` (CI gates + unit tests) ·
`Documentation/` (theory PDFs + governing docs) · `data/` (vendored ARC, ignored).
