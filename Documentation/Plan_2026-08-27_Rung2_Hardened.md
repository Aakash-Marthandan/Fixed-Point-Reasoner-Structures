# The Hardened Plan — Rung 2 to Paper 1 (2026-08-27)

**Mandate (PI):** thorough next steps; the ops inefficiency of rung 1 must not repeat; research integrity and the science are non-negotiable. This document is the working plan; the RUNG-2 LAUNCH REGISTRATION (rules locked pre-data) supersedes its science section on launch day, and each ops item lands only through the offline stub harness. Every delay countermeasure below cites the incident that motivates it.

---

## 0. Where the rung-1 time and money actually went (the honest accounting)

Rung 1 cost ≈ $408 against a registered $120–165 estimate and ~3 calendar days against ~1. Root causes, each with its countermeasure in §2:

| Loss | Magnitude | Countermeasure |
|---|---|---|
| Spot churn: ~20+ node losses across the campaign; 5 nodes for the tail alone; each loss = bring-up (10–30 min under scp flake) + cold compile (15–20 min) + up to one bank quantum of compute (~41 min) | dominant | O1 (code/venv from GCS), O2 (compile cache), O3 (finer bank quantum), O5 (night-window launches) |
| The v6e-16 experiments: multi-host bugs + US-Monday churn burned two launches before the demotion | ~$60–80 + a day | O5 (v6e-8-first posture retained), O6 (16 only in night windows, if at all) |
| The flex-start detour: 7 h queued with no grant, overnight, with the keep-awake mistake blinding the watch | ~9 h calendar, $0 | policy ledgered: spot night window first; flex-start documented fallback only; keep-awake outlives the supervisor whenever anything watches |
| Silent stalls / launch illusions (the 08-26 incident class) | ~1 day | already closed: artifact-verified launches, count-invariants, source-not-proxy discipline — now standing |
| PHASE4 monolith discarded for the hardened rebuild (PI-directed, correct call) | ~33 % of an 8 h run | the rebuild (bank-resume, coop shards, claim TTL) is now the standing machinery |

The science itself was never compromised — the audits prove it — but the wall-clock multiple is the thing to kill.

## 1. Science sequence (registrations own the details; this is the spine)

**R2 — Rung 2, d96 (`--width-scale 6`, 2.11 M params).**
- **Arms (default, pending PI confirmation at registration):** A. B2-class (plain T12 FPA @50k) × 2 seeds — the carrier, seeded per the statistics rule. B. B3-class (priced T12 @50k) — the stability/compression twin, carried by the coincide rule. C. T6@20k + FPA — the wide-funnel insurance, rebuilt on the mechanism (does the anchor rescue the shallow recipe at width?). D. B3-class at β/3 — the H-47 budget-relief cell (does a looser toll preserve the funnel?). B1/RI-NI: dropped (FPA absorbed it; the divergence datum stands).
- **Decision rules (locked at registration):** G-R2-1 breadth-scaling (carrier 20k vote@128 vs 66.16 d64 and 68.62 d16 — SCALES/FLAT/SHRINKS); G-R2-2 cold bands; G-R2-3 H-47 (the β/3 funnel vs standard-β across ≥3 checkpoints — opens/condenses); G-R2-4 insurance verdict (T6+FPA retfm ≥ .9 AND screen ≥ B4-d64's 55.9 → RESCUED); carriers to d128 mechanical. Numeric predictions registered with the entry.
- **Instrument fixes riding the chain (all named in the verdict):** monitors every 2k; screens on ≥3 checkpoints per arm (mid, three-quarter, val-best); ε-ladder extended rungs {.05,.1,.2,.4,.6,.8}; λ_y logged beside λ_joint; the D3 unverified-vote counter (demo cell); the D5 spurious-attractor analysis at close ($0, banked probe rows).
- **Inference-depth rider ($-small):** t=256 full-schedule eval on the rung-2 winner (the cold lever that costs no training — the depth axis EqR exploits; registered as a labeled row, not a headline).
- **Cost estimate: $150–250** (canary-measured before trust; d64 precedent says overhead-bound, not FLOPs-bound). Wall: one v6e-8 night window (§2-O5).

**R3 — Rung 3, d128 (ws8, 3.71 M — under EqR's 5.03 M).** Carrier × 2 seeds + the strongest secondary from R2; same instruments; own registration. ≈ $250–400.

**Paper 1 assembly (parallel from ~Sep 1).** The operational-differences manifest (Instrument_Map Part 2b) is the figure list — 9/10 banked, one cell riding R2. Sections: architecture + toll; the instrument suite; the four laws + the scale-flip results; the Sudoku-vs-ARC operational separation; limitations register. Abstract Sep 18, full Sep 25 (AAMAS fallback Oct 2/9 stands; venue = outcome).

**Held gates (unchanged):** no analyzer edits post-registration; artifact-level admission; measurement law (n ≥ 3 or pairing); breadth labeled, never the headline; indeterminates don't auto-spawn cells; goal-metric named per campaign.

## 2. Ops hardening (each item: what, why, verification gate)

Build order O1→O4 before R2 launch; O5–O8 are posture/policy (no code).

- **O1 — GCS code + venv distribution** (registered 2026-08-25, now due): supervise-start banks `code_<git-sha>.tgz` + a bootstrapped-venv tarball (pinned jax/libtpu inside); node bring-up = one short ssh pulling both (parallel, 5-s flake retries) instead of multi-minute scp passes. Kills the 10–30 min bring-up tax and the shipped-patch incident class (3 occurrences), and makes every node's code provably identical by hash. *Gate:* offline stub harness end-to-end + one live canary node before the campaign node.
- **O2 — JAX persistent compilation cache to GCS:** every fresh node re-paid 15–20 min of XLA compile; five nodes in the tail = ~1.5 h of pure recompute. Enable the persistent cache with a GCS-synced directory; same TPU generation ⇒ cache hits expected. *Gate:* two-node canary (cold node with warmed cache must skip the compile; numerics spot-check vs uncached — same compiled program, but verify by the canary's loss trace).
- **O3 — Finer eval bank quantum:** the 512-record eval batch made the bank quantum ~41 min (one full quantum + recompile ≈ 1 h worst-case churn loss, measured twice). Reduce the draw-batch to 128 records (~10 min quantum). Pure compute-shaping — numerics unchanged — but the instrument is registered, so: *Gate:* the named bit-identical bank-resume test re-run at the new batch size + resumed==uninterrupted equality on a real ckpt.
- **O4 — Harness completeness:** per-worker filesystem isolation in the stub (the 08-25 item; the shared-dir blind spot missed both multi-host bugs); a rung-2 phase for `audit_sportB_integrity`-class checking (generalize the auditor's tag or add `--tag`); the tail-runbook lesson standing: **no ops script launches without a harness pass** — zero exceptions, including "one-off" runbooks.
- **O5 — Launch-timing policy:** long runs LAUNCH INTO the measured-stable US-night window (~00:00–10:00 Z ≈ 05:30–15:30 IST) — the tail's clean node ran 04:34→07:54 Z untouched exactly there; the afternoon window ate three nodes. d96's compute fits inside one window. Corollary: registrations/builds happen IST evening, launches IST early morning.
- **O6 — Accelerator posture:** v6e-8-first ladder stays (the PI's standing one-pod directive); v6e-16 only if a night-window launch wants the halved wall AND the multi-host path re-passes its harness (it exists from wave 2/3a); flex-start remains a documented fallback with its measured ≥7 h grant latency — trigger only on multi-hour daytime droughts, never overnight.
- **O7 — Monitoring stack (standing, now with the rung-1 additions):** supervisor + caffeinate-outliving-everything + launchd watchdog + deadline; session layers = edge monitor with failure signatures, staleness watchdog on GCS artifact mtimes *plus record-count checks* (mtimes prove sync, counts prove compute — the proxy lesson), T+55 m count-invariant, hourly source-verified heartbeat. All re-arm text lives in HANDOFF §2; keep-awake pinned whenever *anything* watches.
- **O8 — Spend tripwire:** registration carries the cost estimate; the heartbeat compares measured spend vs it; at 1.5× the estimate with < 80 % science banked → report to the PI with the pause/continue decision framed (never silently ride past 2×, which is what rung 1 did).

## 3. Execution sequence and timeline

| When (IST) | What | Exit gate |
|---|---|---|
| Aug 27 eve | PI confirms arms/riders → **RUNG-2 LAUNCH REGISTRATION** written (rules + predictions locked; analyzer `analyze_sportB_r2.py` self-tested pre-data) | registration entry + analyzer self-test green |
| Aug 27 eve → 28 morn | Build: O1–O4 + instrument fixes + D3 evaluator stat; full test suite; offline stub harness end-to-end | suite green; harness green; ONE build commit |
| Aug 28 ~05:30 | **Launch** (night window): supervisor, canary-measured pace re-prices the estimate, artifact-verified launch, full monitoring stack | TAIL-START-equivalent verified at source |
| Aug 28 day | Ride (heartbeats; count-invariants; spend tripwire) | chain COMPLETE + self-teardown |
| Aug 28 eve | §6 close: counts + generalized audit 0-FAIL + ONE ledger line + ONE commit | audit 0-FAIL |
| Aug 29 | **Analysis pass** (fresh ingestion → analyzer untouched → physics pass → verdict + report; D5 $0 analysis; demo-figure sweep) | verdict entry + carriers |
| Aug 29–30 | d128 registration (carrier × 2 + secondary) → night launch → close → analysis by ~Sep 2 | same discipline |
| Sep 1 → | Paper-1 drafting in parallel (figures from the manifest); venue checkpoint ~Sep 10 | abstract Sep 18 / full Sep 25 |

Slack: ~9 days between the ladder's natural end (~Sep 2–3) and the abstract — absorbs two siege-class weather days without touching evidence standards.

## 4. Budget

Spent ≈ $1,380 (≈37 %). Committed by this plan: R2 $150–250 · R3 $250–400 · demo/riders ≤ $25 · churn tax 20 % ⇒ **≈ $500–800**, landing the program ≈ $1,900–2,200 with > $1,500 still in reserve for convert-phase/rebuttal — inside every registered envelope rule.

## Appendix A — Pre-mortem audit of the rung-2 build (PI-directed, 2026-08-27; run AFTER the harness passed)

Method: enumerate the live failure modes the campaign could hit, split **data-validity** (science first — must be impossible or detected) from **delay**, and verify each against a harness scenario, a chain guard, a monitor, or a procedure. Two real defects were found and fixed before launch.

**Catches (fixed + harness-covered):**
- **PM-1 (data-validity, critical): PHASE4 partition recomputed per run.** A node-shape change mid-PHASE4 (16→8 or 8→16 — certain to occur eventually with both shapes in the ladder) would mix incompatible shard slices in `p4/`. Fix: partition PINNED in GCS on first entry + per-shard claims (any worker takes any unbanked shard — which also makes the s6/s14 silent-straggler class self-healing) + an explicit n==20000 merge gate. Harness S6 proves the pin across a simulated shape change.
- **PM-2 (delay→completion failure): the depth rider dead-locked the claim queue** (its readiness is produced by a later phase). Fix: own claim-run after PHASE4; guard still requires it. Caught by harness S1 on first run.

**Data-validity register (all green):** wrong-flags → flags are code + artifact-level admission at analysis; wrong-ckpt screens → `step.txt` written per screen (auditor cross-check at close); partial-resume across config → provenance fingerprint (proven; uv flag added to it); partition mixing → PM-1; uv-instrument contamination → fingerprint + batch-invariance test; d3demo ckpt names verified against bucket listings; NaN divergence → detector + the registered one-labeled-relaunch contingency; compile-cache staleness → content-addressed by design (canary confirms); hollow final → hydrating guard (rung-1-verified).

**Delay register:** preemption anywhere → all resume paths proven + partials at 128-record quantum (~10 min); stale claims → TTL (rung-1-proven); wall ceiling → resume; 4-worker staggering → pod.sh aggregation (waves 2/3a); Mac sleep → keep-awake-outlives-everything policy; premature PHASE4 winner on screen-timeout → bounded wait is generous vs measured screen walls, and the analyzer recomputes the winner from complete data (a mis-picked scan wastes compute, never validity); GCS transients → per-pass retries; spend runaway → O8 tripwire.

**Deferred (built during the ride, needed at close, cannot affect the run):** the auditor's `--tag sportBr2` generalization; the venv-tarball half of O1 (code-dist shipped; venv build ≈3–5 min/node stands).

## 5. What is explicitly NOT changing

The evidence discipline: pre-data rules, untouched analyzers, artifact admission, labeled n, append-only ledger, breadth-vs-cold labeling, the measurement law, PI gates on registrations. Hardening buys wall-clock and dollars; it never buys statistics.
