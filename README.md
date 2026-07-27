# QHRRN-2 — A Holographic RG Network for ARC-AGI

Physics-grounded architecture for the Abstraction and Reasoning Corpus: an
equivariant recursive coarse-graining core with **priced boundary streams**
(information flux across RG cuts is measured and penalized), a discrete rule
codebook selected by annealed attention, and test-time training that adapts
only boundary parameters.

**Status: rebuild in progress.** No performance claims live in this README —
claims belong to the ledger (see below) until evaluation day.

## Current status (2026-07-21) & next steps

Done: post-mortem of the April system (measured, `d874427` preserved) · QHRRN-2 core
implemented at d=12 (45,653 params, 22 tests green incl. CI-1/CI-2 gates) ·
CI-3a triad at 2/3 with the translate miss diagnosed as non-structural noise ·
**thesis narrowed to four kill-conditioned statements S1–S4**
(`Documentation/Thesis_Information_Holography.md` §6; ledger §3c) with two
load-bearing citations verified (RT-on-trees; CompressARC 76k/20% baseline).

Next, in order:
1. **Amendment D** (ledger C14): KL-priced attention channels at all scales
   ("wormhole tolls") — enables S3; ~1 h of code + tests.
2. CI-3a to 3/3 via seed-ensemble voting (remedy queued in ledger log).
3. Phase 2 per the roadmap (`Divergence_Analysis_2026-07.md` §7): dev-30 +
   ablations + flux-frontier measurements on constructed families (S1/S2).
4. Phase 3: RE-ARC-style pretraining on TPU (GCP project `quantum-llm` is
   fully configured; dispatcher discipline + `--spot`).

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
3. CI gates (all local/CPU) must pass before any cloud spend:
   equivariance bit-exactness · anti-linearity/rank · sanity triad via TTT ·
   seam-boundary task · flux-direction sanity · canvas/no-GT-size.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
git clone --depth 1 https://github.com/fchollet/ARC-AGI.git data/ARC-AGI  # vendored, git-ignored
.venv/bin/pytest                                                          # run the gates
```

Layout: `src/qhrrn2/` (implementation) · `tests/` (CI gates + unit tests) ·
`Documentation/` (theory PDFs + governing docs) · `data/` (vendored ARC, ignored).
