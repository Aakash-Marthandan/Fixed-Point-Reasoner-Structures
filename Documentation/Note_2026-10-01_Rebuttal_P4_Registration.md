# Note 2026-10-01 — P4: the freeze-time ladder (messages frozen from iteration t₀) — registration, before any row

**Goal (page one).** A MEASUREMENT. No accuracy target; $0; the Mac's CPU; inference only on the manuscript's banked attention checkpoints; nothing trained. Follows `Note_2026-09-26_Review_Period_P1c_P2b_Registration.md` (P2b: the iteration-1 messages replayed at iterations 2–16 preserve every initial solution and discover 0 / 6 / 0 against intact's 192 / 156 / 171). Plan: `Plan_2026-10-01_Rebuttal_Experiments.md` §2. Tool: `tools/rebuttal_p4.py` (subclass of the P2b runner, release code unchanged; outputs `runs/analysis/rebuttal_20261001b/`). Queued behind P3 on PID waits; registered before the build.

## Why

P2b showed that with the exchange held at its iteration-1 values the state settles to a fixed point a few cells from the first grid, so discovery needs updated messages. It does not say **until when**. If stale messages recorded later in the trajectory still complete most of what intact continuation completes, the exchange matters early and the late iterations run on the carried state; if freezing at any iteration stops the remaining discoveries, the messages must keep updating up to the completing iteration. §3 reports that completion is abrupt and late; this ladder ties the communication finding to that time course.

## Design

**Population and receivers.** The study's 256 shared first states (every second repair puzzle), Attention 128 / 192 / 256 at their benchmark weights, batch 64, 16 iterations, float32 CPU; the trajectories coincide with the study's intact continuation through iteration t₀ − 1.

**Conditions.** `frozen_from_t{t₀}` for t₀ ∈ {2, 4, 6, 8}: iterations 1 … t₀ − 1 run intact with the hook off; during iteration t₀ − 1 the 42 cross-field message tensors are recorded per puzzle; at iterations t₀ … 16 every message tensor is replaced by the recorded one at the same (application, block). t₀ = 2 is P2b's `messages_frozen` and must reproduce it bitwise.

**Gates.** (1) The hook-off path reproduces the study's intact logits bitwise at every iteration on batch 0 (n = 64). (2) `frozen_from_t2` reproduces P2b's `messages_frozen` chunk 0 bitwise. (3) Replaying the iteration-(t₀ − 1) record at iteration t₀ − 1 itself reproduces that iteration bitwise, checked for t₀ = 4 on batch 0 (the replay path is exact at a later step too).

## Quantities (per receiver, per t₀)

With intact's per-iteration exactness from the study's chunks: the late set L(t₀) = puzzles not exact after iteration t₀ − 1 that are exact at 16 under intact; D(t₀) = members of L(t₀) exact at 16 under the freeze; F(t₀) = D / L. Preservation: puzzles exact after iteration t₀ − 1 that are not exact at 16 under the freeze (count). Descriptive: among D(t₀), the wrong-cell count after iteration t₀ − 1 and the intact completion iteration τ relative to t₀ (whether the freeze completes only grids that intact completes at t₀ itself); the settling of the frozen state (mean cells changed between consecutive iterations); exact at 16 in full.

## Rule (confirmatory)

EXCHANGE-UNTIL-COMPLETION if F(t₀) ≤ 0.25 for every t₀ ∈ {4, 6, 8} on all three widths; LATE-EXCHANGE-DISPENSABLE if F(8) ≥ 0.75 on all three widths; MIXED otherwise; a t₀ with L(t₀) < 10 on a width is UNDEFINED there and excluded from the letter, with the exclusion stated. Preservation is reported beside the letter: a freeze that loses more than 2 % of the solutions reached before t₀ is said so.

## Predictions and credences (before any row)

EXCHANGE-UNTIL-COMPLETION 0.55; MIXED 0.35; LATE-EXCHANGE-DISPENSABLE 0.10. F(t₀) rising with t₀ on every width: 0.6. No solution reached before t₀ lost at any t₀: 0.7. Among D(t₀), a majority with τ_intact = t₀ (completed in the very next iteration anyway): 0.6.

## Wording under each outcome

- **EXCHANGE-UNTIL-COMPLETION.** "Holding the exchange at its values from any iteration stops nearly every discovery that intact continuation would still make, at whatever iteration the hold begins, while every solution already reached is kept: the messages must keep updating up to the completing iteration." The §4.1 claim gains a time course that matches §3's abrupt completion.
- **LATE-EXCHANGE-DISPENSABLE.** "Once the first iterations have passed, replaying stale messages still completes most of what intact continuation completes: the exchange matters early, and the late iterations run on the carried state." §4.1 is narrowed to the early iterations.
- **MIXED.** F(t₀) per width with both readings.
Under every outcome: the frozen messages are the model's own but stale; no message content is identified; the "pinning" reading (a replayed sequence drives the state back to the recorded iteration's fixed point) is not separated from the communication reading.

## Labels and plan

Confirmatory: the letter. Exploratory: the τ-relative distribution, settling, preservation counts. Build now; selftest and a 16-puzzle smoke with gates (1)–(3) before queueing; the queue waits on P3's width PIDs; three widths in parallel; report by the tool; the Outcome appended here and a ledger line written when read.
