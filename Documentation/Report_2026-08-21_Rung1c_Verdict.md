# Rung 1c — The Width×Budget Completion: Verdict, the Floors×Budget Interaction, and What Rung 2 Inherits

*2026-08-21. Registered rules: ledger 2026-08-20 RUNG-1c LAUNCH REGISTRATION (R1c-1..R1c-4, locked pre-data; analyzer `tools/analyze_r1c.py` self-test 9/9, run untouched). Artifacts: `runs/analysis/r1c_verdict.txt` (registered verdicts), `runs/analysis/r1c_physics_20260821.txt` (physics pass, `tools/analyze_r1c_physics.py`). All 20 cells ADMITTED at artifact level. Measurement law binds; n labeled throughout. No number below is hand-typed.*

---

## 0. What was measured

Ten new pretrains (B64/T6/NI σ=.01, same corpus, knee β=3e-5/1e-5 except A13): **A8s1/s2** (A4-class d48@53,333 → A8 n=3 with rung-1b's A8s0), **A10s0/s1** (A4-class d64@40,000 — the missing 2×2 corner), **A11s0/s1** (A5-class d48@40,000 — the d96 baseline), **A12s0/s1** (A5-class d64@40,000), **A13s0/s1** (A5-class d48@53,333 at β=6e-5/2e-5 — the re-pricing rescue). Batteries lad/rg-96/rt ×10 + e1e3 on the four 2×2 corners; 44/44 files at 48/48. Comparators: rung-0 anchor (A4 d48@40k ×3), rung-1 A4/A5 d64@53k, rung-1b A8s0.

**Integrity (post-results critique):** task sets bit-identical across all 20 cells per battery (one hash each, zero duplicate rows); seed files distinct; only A13s0 resumed (twice: evening churn + the overnight drought — valid draw, batch stream re-seeded per the standing caveat, params/optimizer/LR exact); **cross-campaign replication control: A8s0 (rung-1b) = 25 lands inside the fresh A8 seeds {24, 27}** — the instrument is stable across campaigns. The A10 seed delta (+13) is **diffuse** (max single-family contribution 2 counts across 79 gate families) — real seed noise, not an artifact. Absences named: no A5-class d48@53k cell at β=3e-5 exists (A13 is at 2×β); corner c has n=2; e1e3 absent on A11/A12/A13 by design.

---

## 1. The registered verdicts (R1c-1..R1c-4, as locked 2026-08-20)

| rule | reading | verdict |
|---|---|---|
| **R1c-1 (H-40) the A4-class 2×2** | corners rg-96: a=d48@40k [34,35,34]=**34.3** · b=d48@53k [25,24,27]=**25.3** · c=d64@40k [37,24]=**30.5** · d=d64@53k [27,23,24]=**24.7**. Δbudget\|d48 **+9.0** (seed-overlap NONE; 9/9 McNemar pairs signed a>b, individually p=.11–.32) · Δwidth\|40k **+3.8** (corner c spread 13, seed-overlap YES) · Δwidth\|53k +0.7 · interaction −3.2. N secondary: Δbudget +5.7, Δwidth +2.3. | **INDETERMINATE** by the composite rule — the width leg landed at 3.8, in the (3,4) gap between BUDGET-DOMINANT's ≤3 guard and BOTH's ≥4. The **budget leg alone fired unambiguously** (≥5 with no seed overlap); the width leg is genuinely unresolvable at n=2/spread 13. See §3.1. |
| **R1c-2 (A5 anchors)** | A11 [29,28]=**28.5** vs floors anchor 34.3 → Δ **−5.8** (2/2 seeds below the anchor's min by ≥5; spreads 1/1). A12 [26,27]=**26.5** vs A11 → Δ −2.0. | **A11-BELOW** (floors-inert-at-d48 NOT confirmed on the A5 substrate — floors *help* +5.8 at d48@40k; §3.2) / **WIDTH-TAX ABSENT** on A5 at 40k (−2.0 ∈ ±3) → **the d96-GO prior at the 40k budget FIRED**. A11/A12 = the d96 baseline row, in hand. |
| **R1c-3 (H-41) re-pricing rescue** | A13 [23,26]=**24.5** vs A8 25.3 (Δ **−0.8**; +5 required) vs anchor 34.3 (Δ −9.8). I_med A13 **343** vs A8 565. | **RESCUE-FAILS** — the registered kill fired. **H-41 FALSIFIED**: β∝steps does not hold transfer; 2×β compresses −222 nats transfer-neutrally. Registered consequence fires: **budgets held at 40k ladder-wide**. |
| **R1c-4 (e1e3, directional)** | wrong-stable: a .891 → b .921 / c .922 / d .924. nd1 (endpoint condensation): a 50.0 → b 46.0 / c 39.5 / d 38.0. limit-exact 15.0 → 11/11/10.7. | Reported (no rule): wrong-stable rises **equally** with budget (+.029) and width (+.030); endpoint **proliferation is width-led** (nd1 −10.5 width vs −4.0 budget). §3.4. |

---

## 2. The headline: the budget tax is real, replicated — and **substrate-scoped**

**H-40's core phenomenon is confirmed where it was registered.** A8s0's n=1 (rg-96 25) replicated to n=3 almost exactly (25.3): at fixed width d48 and fixed β, +33% optimization erases 9.0 rg-96 counts and 5.7 N with the throat flat (586→565). This is the cleanest budget-tax measurement the program has.

**But the tax does not appear on the operating substrate.** The physics pass's floors×cell table (artifact §C):

| cell | floors (A4-class) | global (A5-class) | floors effect | floors' extra nats |
|---|---|---|---|---|
| d48@40k | 34.3 [34,35,34] | 28.5 [29,28] | **+5.8** | +66 |
| d64@40k | 30.5 [37,24] | 26.5 [26,27] | +4.0 (noisy) | +77 |
| d64@53k | 24.7 [27,23,24] | 32.0 [34,34,28] | **−7.3** (H-39) | +77 |
| d48@53k | 25.3 (A8) | 24.5 (A13, 2×β) | +0.8 (β-confounded) | +222 (β) |

And the A5-class budget axis (artifact §D): at d64, 40k→53k moves rg-96 **+5.5** (26.5→32.0) — the *opposite sign* of the A4-class tax. Within the A5-class the whole (d, steps) surface spans 24.5–32.0 with seed spreads 1–6: **no tax is resolvable on the going-forward substrate**, while the floors substrate shows a clean −9.0.

**The unifying reading (registered as H-42):** the free-bits floors exempt the first ~F_s nats from pricing; those locally-unpriced nats are shaped only by the fitting term. At short budget the slack is generic and transfers (+5.8 at 40k); with longer optimization the fitting term specializes exactly that unpriced capacity to trained-family structure — a **memorization pocket** — and the floors' effect inverts (−7.3 at 53k). Fully-priced codes have no pocket: every nat pays rent continuously, and more optimization refines rather than specializes (A5-class budget-neutral-to-positive). This is S2's excess-flux-buys-memorization measured *inside* priced arms at the ~70-nat scale (previously only plain-vs-priced at the ~10⁵-nat scale), and it retroactively explains why the "width ceiling" (H-36) and the "budget ceiling" (H-40) were both measured on floors cells.

**H-36 is now resolved by decomposition:** the rung-1 deficit (a−d = 9.7) = budget 9.0 + width|53k 0.7 — **~93% budget, ~7% width** on the floors substrate; on the A5 substrate the width tax at 40k is ABSENT (−2.0). Width is back for the ladder at matched budget. **Depth-lean's motivation dissolves; d96 is the right rung 2.**

---

## 3. Physics reading

### 3.1 The two taxes dissociate mechanistically
The e1e3 battery separates what budget and width each do to the landscape (artifact §G): **width proliferates distinct wrong endpoints** (nd1 50→39.5 at matched 40k; −10.5) while barely moving transfer at 40k; **budget erodes unseen-family basins** (rg-96 −9.0 on floors cells) while moving nd1 only −4.0. Wrong-stable rises identically (+.03) along both axes. Two mechanisms, previously conflated in "the width ceiling": a dimensional/inventory effect (more spurious fixed points) and an optimization effect (unpriced capacity specializes).

### 3.2 The OOD gradient survives on the budget axis
The budget tax at d48 is distribution-graded like rung-1's width deficit was (artifact §H): rt −18% (spread-9 baseline; directional), vh N −12%, rg-96 −26% — hardest furthest from the training distribution, consistent with specialization-of-trained-structure rather than uniform degradation.

### 3.3 The β-transfer frontier is FLAT: compression and transfer have decoupled at d≥48
Assembling every priced cell at 53k: d64 global+NI at β=3e-6/1e-5/3e-5 → rg-96 35.5/27.0/32.0 with throat 1604/793/455 (rung-1b); d48 at β=3e-5/6e-5 → 25.3/24.5 with throat 565/343 (this round). **Across a 20× β range at two widths, transfer never responds beyond seed noise while the throat moves ~4.7×.** The budget tax is therefore *not* a rate–distortion artifact at any accessible β (H-38's last branch closes; H-41 falsified) — it is optimization-dynamical, and the only lever that governs it is the budget itself (held at 40k) or the substrate (fully-priced). A13's throat **343 nats is the program's lowest priced throat ever** (prior low 380 at d16/1e-4) with S(.4)=14.5 and r̄=.237 intact — compression remains geometrically near-free; the only hint of a dose cost is rt 13.0 (−28% vs A8, n=2, directional watch).

### 3.4 η — the progress clock acquires its 2×2 (artifact §F)
Means by (steps, d): d48@40k **.179** · d64@40k **.198** · d48@53k **.206** · d64@53k **.238**. Steps effect +.027/+.040, width effect +.019/+.032 — monotone in both, β- and floors-invariant (A13 .206 = A8 .206; A11 .179 = anchor .179). H-30's restated form now has a seeded 4-cell surface. Extrapolation for d96@40k: **η ≈ .21–.22** (a pre-registrable prediction).

### 3.5 Throat and profile on the operating substrate
L1 extends on A5-class: 520 (d48@40k) → 499 (d64@40k) → 456 (d64@53k) — capacity and budget both still shorten the code. IR (s3+s4) is **not** conserved across substrates at fixed β (floors cells 46–56, A5-class 32–42, A13 25) — it scales with floors and β, completing rung-1b's withdrawal of IR-conservation-as-floor. **New H-34 nuance: the knee profile is partly floors-*enforced***. Floors cells sit uniformly on-profile (L1 .014–.061; the floor vector *is* the knee profile), but A5-class cells wander seed-wise (one seed of each pair on-profile at .04–.05, the other at .086–.126; A13 .120–.139 with UV .66–.75). The natural (global-priced) profile at d≥48 has seed-scale drift the floors used to pin. d96 predictions should use L1 ≤ .15, not ≤ .10.

### 3.6 Code geometry — every effect is in N/transfer; r̄ is a constant of the program
Packing plane (artifact §J): r̄ ∈ [.222, .241] across all eight groups — sixth consecutive confirmation of L2's "distance saturates, inventory moves." A11 = (48.0, .222) vs anchor (47.3, .235): the d96 baseline holds codebook size at 66 fewer nats; its −5.8 deficit is **transfer-specific**, not inventory.

---

## 4. Adversarial pass — what these numbers do not show

1. **Corner c is bimodal and n=2.** A10s0 (rg-96 37) is the best d64 transfer cell ever measured; A10s1 (24) sits at the d64@53k level. Every width-at-40k statement on the A4-class rests on this spread-13 pair; the diffuse per-family delta says real seed noise, and n=2 cannot resolve it. A third A10 seed (~1 pod-h, ~$8) closes R1c-1 formally if wanted for the paper. **The d96-GO decision does not depend on it** (it rides R1c-2, which is A5-class).
2. **The A5-class budget-immunity (+5.5) is 2-vs-3 with spreads 1/6** — directional. Its d48 counterpart runs through A13's β-bridge (assume β-inelasticity, itself partly supported by the same cells) — a circularity risk, stated. The missing de-confounding cell is A5-class d48@53k at β=3e-5 (~1 pod-h); it is also H-42's named test.
3. **McNemar is individually marginal everywhere** (budget pairs p=.11–.32 despite 9/9 sign) — the budget claim rides seed-mean separation with zero overlap (measurement-law compliant), the same epistemic grade as R1b-2's floors verdict.
4. **H-42 is a post-hoc unification** registered from the same data that suggested it; its confirmations must come from cells it did not see (the A5-d48@53k cell; d96 behaving budget-robust).
5. **rt and vh-N instruments are noisy at these n**: rt seed spreads up to 9 (anchor 26/23/17), vh-N spreads 10–16 at the new cells (A10 16, A12 14, A13 10). All rt statements directional; rg-96 remains the claim-bearing instrument.
6. **Cross-rung corners**: corner a is rung-0 data, corner d rung-1, b spans r1b+r1c. Harness/flags/gates verified identical (hashes; admission); the A8 cross-campaign replication (25 ∈ {24,27}) is the positive control, but environment rot between campaigns remains an unquantified residual.
7. **Per-family numbers are 9-pair counts** — HorizontalVertical −2.3 / Order −2.7 under budget are directional pattern candidates only.
8. **A13's verdict is for its dose** (6e-5 = 2× knee, β∝steps exactly); the flat frontier makes other doses unpromising but they are untested.

---

## 5. Curiosity pass (each with its n)

1. **A12 (the d96 baseline row at d64) is the best *reachability* cell measured**: exact@T 15.0 (16/14) beats the anchor's 14.3, and rt 31.0 (33/29) is the program record (+41% over the anchor; anchor spread 9, directional). The operating substrate at 40k leads in-family generalization and quench reachability while sitting mid-pack on rg-96 — reachability and transfer decouple at the substrate level.
2. **The Copy crack has a substrate pattern**: nonzero Copy retention on 5/11 new cells — A11s0 1, A11s1 1, A12s0 **2**, A8s0 1 (r1b) — concentrated on **A5-class** cells; all A4-class r1c cells read 0. Global pricing (which compresses UV harder) cracks the content-verbatim family floors don't. Counts of 1–2/9: watch, not claim. ExtractObjects stays 0 on every substrate ever measured (~48); Count 0–1.
3. **A8s2 is a partial anchor-revenant**: N 47 (anchor level) with rg-96 27 (budget level) — N and rg-96 decouple within a seed; the budget tax hits transfer before inventory.
4. **Basin-edge fuzz** 5–13 pairs/arm — within the 4–10% instrument-noise precedent; A10s0's 13 is the table's max and its seed-delta is diffuse regardless.

---

## 6. Registered consequences → what rung 2 (d96) inherits

Fired by this round's rules:
- **Budgets held at 40k ladder-wide** (R1c-3 consequence; H-41 falsified — no β(budget) law exists).
- **d96-GO prior at the 40k budget** (R1c-2: width tax ABSENT on A5-class), with **A11 (28.5, N 48.0, I 520) and A12 (26.5, N 44.0, I 499) as the baseline row**. Width direction re-opened by H-36's resolution (93% budget); depth-lean's motivation is gone.
- **Operating substrate stays A5-class** (R1b-2 ledgered decision, now with its cost quantified: −5.8 vs floors at d48@40k — the price of a substrate with no memorization pocket and no per-width floor calibration; H-42 predicts the floors' 40k dividend inverts at scale/budget anyway).

Materials for the d96 registration (predictions this data supports pre-registering):
| quantity | prediction basis | prediction for d96 A5-class @40k |
|---|---|---|
| rg-96 | width-tax-absent prior (28.5→26.5 across d48→d64) | **≥ 23.5** (A12 − 3); kill if < 21 |
| N | A11/A12 44–48, r̄ flat | 42–50, r̄ .22–.25 |
| throat I_med | L1 monotone 520→499 | **< 499**, declining |
| profile | H-34 (A5-class wander) | knee-member, **L1 ≤ .15** |
| η | progress clock §3.4 | **.21–.22** |
| IR s3+s4 | A5-series 42→35 | ~30–35 |
| vacancy | W-α watch | ExtractObjects 0; Copy 0–2 (watch) |

Optional cheap cells (PI menu, ~1 pod-h ≈ $8 each, can piggyback on the d96 pod):
1. **A10s2** — closes R1c-1 formally (resolves the bimodal c corner for the paper's 2×2 figure).
2. **A5-class d48@53k @3e-5** — de-confounds A13, completes the A5 2×2, and is **H-42's registered test** (predicts ≈ A11 within noise if the pocket story is right; ≈ 24.5 if budget taxes A5-class too).

**New hypothesis registered (H-42, §3):** the budget tax lives in unpriced code capacity ("free-bits memorization pocket") — floors' free-bits specialize to trained-family structure under long optimization while fully-priced code is budget-robust. Test: the A5-d48@53k cell above + d96 budget behavior. Falsified if that cell drops to ≤25 (A5-class taxed too) or if a floors arm at long budget ever beats its global twin on rg-96 at n≥2.

---

*Artifacts: `runs/analysis/r1c_verdict.txt`, `runs/analysis/r1c_physics_20260821.txt`; data `runs/pretrainr1c_*`, `runs/*_pr1c*` (+ GCS `rr1c`). Analyzers: `tools/analyze_r1c.py` (pre-registered, untouched), `tools/analyze_r1c_physics.py` (this pass).*
