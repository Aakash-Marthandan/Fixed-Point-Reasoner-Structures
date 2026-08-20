# Rung 1b — β-rescale, Floors, the rt Gradient, and the Budget Reattribution

*2026-08-20. Registered rules: ledger 2026-08-19 RUNG-1b LAUNCH REGISTRATION (locked pre-data; analyzer `tools/analyze_r1b.py` self-test 15/15, run untouched). Artifacts: `runs/analysis/r1b_verdict.txt`, `runs/analysis/r1b_physics_20260820.txt`. All 15 cells admitted at artifact level. Measurement law binds throughout.*

## 1. Registered verdicts

| rule | verdict | reading |
|---|---|---|
| **R1b-1a (H-38 β-rescale)** | **INDETERMINATE** | rg-96 vs β non-monotone — 3e-5: 32.0 (n=3), 1e-5: 27.0 (n=2), 3e-6: 35.5 (n=2) — within seed noise (spreads 6–10). Kill did **not** fire (A7 both seeds > the 27 band). Clean monotonicities: throat 455→793→1604 nats (×~2 per β÷3; still 200× below plain at 3e-6); **N rises at lower β** (38.0→44.5→42.5) — the knee β over-prunes the d64 codebook. |
| **R1b-1b (ceiling mechanism)** | **INDETERMINATE → PI decides** (depth-lean default) | Best arm A7: transfer **recovers to the anchor** (35.5 ≥ 34.3) but codebook does not (42.5 < 44). Tiebreak non-dominated: (42.5, .251) vs (47.3, .235) — at the right β, **d64 sits ON the (N, r̄) frontier**; rung-1's "frontier-interior" was the floors+knee-β cell. |
| **R1b-2 (H-39 floors)** | **CONFIRMED — floors cost transfer** | A5 {34,34,28} = 32.0 vs A4 {27,23,24} = 24.7; 3/3 A5 seeds above A4's max; floors cost +77 nats. **Consequence fired: floors dropped; operating substrate = A5-class (global+NI).** Honesty: per-pair McNemar .052–1.0 (sign 8/9) — modest effect carried by the seeded rule. |
| **R1b-3 (rt OOD gradient)** | **CONFIRMED** | rt-48 flat across width (19.7 vs 22.0, Δ−2.3; seed spreads 12/9) while val-hard fell −6.7 and rg-96 −9.6. Priced rt 15–27 vs plain 6–7 — Law 4 extends to the rt gate. |
| **R1b-4 (A8: η, budget)** | both **INDETERMINATE** by their letter | η(A8)=.209, exactly between the registered bands; the budget rule required an I_med drop that didn't occur. The cell itself is the campaign's headline (§2). |

## 2. The headline: the "width ceiling" is likely a **budget** ceiling (n=1, directional)

A8 = the anchor's own arm (A4-class, d48) at the **rung-1 budget** (53,333 vs 40,000):

| cell | N | rg-96 | I_med | η |
|---|---|---|---|---|
| d48@40k A4 (n=3) | 47.3 | 34.3 | 586 | .179 |
| **d48@53k A4 = A8 (n=1)** | **39** | **25** | 582 | .209 |
| d64@53k A4 (n=3) | 40.7 | 24.7 | 532 | .234 |

A8 reproduces **97% of the rg-96 deficit and ~125% of the N deficit at fixed width**, with the throat flat — so the deficit tracks optimization length, not compression. Rung 1 compared d64@53,333 to d48@40,000 (steps ∝ d by convention): width and budget were confounded, and the deconfounder points at budget. η reads as a progress clock (.179 → .209 → .234; length and width each ~half). Physical reading: at fixed β, longer optimization keeps sharpening trained-family basins while nothing in the loss protects unseen-family structure — the transfer tax is paid in optimization time, not in nats. **If confirmed, every rung's steps-law inherits this tax unless β is re-tuned with budget — including depth-lean rung 2 at steps ∝ T·d.** Registered as **H-40** with the 2×2 completion as its test. Adversarial: n=1; paired McNemar vs the anchor p=.175; A8 seeds may regress.

## 3. Physics corrections (both self-corrections of rung-1 readings)

1. **IR "conservation" withdrawn.** s3+s4 = 34 / 62 / 97 nats across the β decade (plain 2652) — the coarse-cut flux scales with β, so its width-invariance at fixed β was a **price artifact**, not a task floor. S1's floor claim still requires the β→0⁺ frontier sandwich, as originally registered.
2. **H-34 restated: profile = f(task, β).** d(free) falls continuously as β drops (.201→.159→.127) — the two-fixed-point picture is a β-parametrized family; the "knee" is its β=3e-5 member. What rungs 0–1 established survives precisely: at *fixed* β the profile is width/seed/budget-invariant.

## 4. Curiosity

- **W-α first crack:** Copy retains 1–2/9 on 5 of 7 new substrates — first nonzero in 39 substrates. Count-resolution: WATCH, not claim. ExtractObjects and Count remain 0/9 everywhere.
- η rises at the lowest β (A7 .268 > plain .246, n=2) — the freest priced map flows fastest.
- A6s0 is the best single d64 cell ever (N 48 = anchor level, exact 15) — but its seed sibling is 10 counts away; the β axis is seed-noise-dominated at n=2.
- Center 0–3 scattered (H-35 refutation holds).

## 5. Recommendation (R1b-1b says PI decides; depth-lean is the default)

Run the **WIDTH×BUDGET 2×2 completion** before committing rung 2: A8 seeds {1,2} (d48@53k → n=3) + d64@40k A5-class seeds {0,1}, full batteries — ~4 arms, ~2.5 pod-h, ≈$18. Decision rules lockable at registration: d64@40k recovers rg-96 ≥31 AND A8 seeds confirm ≤27 → **budget is the killer** (H-40 confirmed; the ladder re-prices β with budget; rung-2 depth must not scale steps without re-pricing); d64@40k stays ≤27 while A8 recovers → width after all; both low → the taxes stack. Operating substrate for all future cells: **A5-class (global+NI)** per R1b-2.
