# Champion sportC1 — the d128 Native Round, Designed From the Pilot's Mechanisms

**Date:** 2026-09-02 · **Status:** DESIGN + REGISTRATION DRAFT (rules drafted here; they LOCK in `tools/analyze_sportC1.py` with a selftest before any launch; the chain is a derivative of `chain_sportC0.sh` with the §4 hardening; nothing here is a result). · **Inputs:** `Report_2026-09-02_ChampionPilot_Verdict.md` (every claim cited there), the program review's protocol table, the standing laws. · **PI decisions are marked ▶.**

## §1. What the pilot changed in the design (one line each, with the measurement)
1. The native regime memorizes (train CE → 0.000 by 40k; P1@50k solves 99.9 % of its training puzzles and 19.4 % of the test) because native9 removed the canvas's implicit ×576 placement augmentation → **aug ≥ 1000 + two-phase schedule + val-selected full tests are mandatory**, and the pilot's ARITY-HURTS letter is read as a regime failure, not a geometry failure (P3s1 native cold 37.35 = program record; native ≥ canvas at every val-selected checkpoint).
2. RI is the per-draw lever (+34–38pp b1) and makes the map init-invariant (b1 ≈ cold per octile; EqR's residual selector ≈ the free verifier) but drives η → .92–.99 with inference-time state explosions (P2 5/64 puzzles, P5 |z| 3e18) and 4/6 training deaths → **RI stays, with a one-variable stabilizer arm (η cap) and ×3 seeds; the explosion census becomes a standing eval.**
3. RI funnels are deep-narrow (ρ .36–.44) — breadth needs a different map → **a wide-funnel arm rides alongside** (no-RI native and canvas C3X-class), and the portfolio union is a registered readout (our coverage column).
4. aug1000 pays at equal steps (+5.70pp paired at 10k) → aug 1000 baseline (▶ 2000 if the PI wants the corpus-diversity axis pushed; the position group has 3.36M elements).
5. Native is cheap (arm ≈ 1.75–2h incl. evals at d96 on a v6e-8) → the round affords 7–8 arms at d128 and a d160 point for ≈ $130–190.

## §2. Frontier framing (the numbers this round can honestly move)
| statistic (column) | frontier | pilot native d96 (1.71M) | this round's target (d128/d160) |
|---|---|---|---|
| single prediction, cold | HRM 55.0 · TRM 87.4 | 37.35 (P3s1) | **≥ 45 at d128; ≥ 50 at d160** (floor claim: beat HRM 55 at ≤ 5M is the d160/d192 push) |
| B=1 single random-init draw | EqR 93.0 | 38.47 | ≥ 45 |
| Top-1-by-residual @128 (verification-free) | EqR 99.8 | 52.49 | ≥ 60 |
| unverified majority @128 | aug-HRM 96.9 | 43.95 | ≥ 50 |
| verified coverage @128 (ours; no external counterpart) | — | 52.85 native / 88.89 canvas C3X | ≥ 92 on the wide-funnel arm; **portfolio union ≥ 95 = B-M3 (labeled protocol)** |
TRM-class and EqR-class per-draw numbers are not promised for this round; the paper states the gap and the levers (budget, capacity, RI stabilization).

## §3. Arms (all d128 / ws8 = 3,004,530 params unless noted; native9; FPA k4 ε.2; β_nl 1e-6; β_flux 0; lr 1e-3; B64; warmup 500; two-phase = 50k cosine (1e-3→3e-5) then +30k floor at 3e-5 with a fresh optimizer, the C3X pattern; aug 1000; monitors 2k; grids 5k; one-shot amputation; ▶ marks PI choices)

| arm | recipe | purpose | seeds |
|---|---|---|---|
| **A0, A1, A2** | T16 + RI .5 (the P3s1 recipe, scaled) | THE CHAMPION; survival is part of the readout | 0, 1, 2 |
| **B0** | A + **η cap 0.90** (`eta = eta_floor + (eta_cap − eta_floor)·sigmoid`; evaluator/monitor mirror it; named test; cap=1.0 is bit-exact) | the mechanism-derived stabilizer, one variable from A0 | 0 |
| **C0** | T16, NO RI, aug1000, two-phase (P1-class in the fixed regime) | (i) does the regime alone rescue the no-RI native map? (ii) the native wide-funnel/coverage carrier | 0 |
| **D0** | canvas C3X-class at d128 (canvas32, seam mixers, T6 FPA, 20k cosine + 30k floor, β_nl 1e-6) | the PROVEN breadth carrier scaled; the portfolio partner; canvas-vs-native at d128 | 0 |
| **E0** ▶ optional | A-recipe at **d160 / ws10 (4.66M)** | the ladder point toward HRM 55 (still under EqR's 5.03M) | 0 |
| **F0** ▶ optional | A0 + **long floor**: +90k more floor steps (total 170k ≈ 11 epochs at aug1000) | the optimization-budget axis the series exploits and we never pulled; memorization tripwire guards it | 0 |

NI: out (NI-NEUTRAL, no per-draw gain). Placement offsets: none (native). EMA of weights: noted (TRM's +7.5pp lever), unadopted this round ▶ unless the PI wants it as a rider on A0 (eval-side EMA copy; one flag).

## §4. Chain hardening (each a pilot lesson; all through the offline stub harness before silicon)
1. Trainer **early-NaN abort**: non-finite loss at any monitor/log row → stop, bank the last finite grid, STOPPED label (saves 25–30k wasted steps per death).
2. **Full test on vsel AND final**; the scan on vsel; screens at {15k, 35k, stage-A end, B+5k, B+15k, vb}; **skip fixed-step screens beyond a STOPPED arm's last finite grid**.
3. Ship **stage-A dir + metrics + config.json** for two-stage arms; copy `config.json` on stage-A death.
4. `select_ckpt` fallback made LOUD (`VB-FALLBACK-FINAL`) and the banked grids + metrics re-pulled on any node change before evals.
5. **Explosion census** (CPU or idle chip; strat-512, t=64 and t=256) on every arm's vsel and final grid — a standing artifact.
6. **CE tripwire** logged: `ce_in` at end; rule R-C1-0 reads it.
7. Pulls: `gcloud storage cp` per object + crc32c; bash-3.2-portable scripts; resume steps recorded to `runs/<arm>_resumes.txt` so resume-adjacent deaths are labeled automatically.
8. Ops unchanged otherwise: pod.sh supervise + campaign env, US-first zones, v6e-16 4×4 first / v6e-8 fallback, static worker map, deadline cap, node-side guard, one-shot amputation, self-teardown.

## §5. Decision rules (DRAFT — lock verbatim in `analyze_sportC1.py --selftest`)
Noise: **FNC1** = max(|b1(A0) − b1(A1)|, |b1(A1) − b1(A2)| over clean A pairs, .02); **CNC1** likewise on vsel-cold (.01 floor). Clean := retfm ≥ .9 AND not STOPPED AND end-CE ≥ .02 AND explosion fraction ≤ .02 at t=64 on the vsel grid.
- **R-C1-0 REGIME:** every arm's end-of-training `ce_in` ≥ .02 AND |vsel-cold − final-cold| ≤ CNC1 → NO-MEMORIZATION (the regime holds); any arm with CE < .02 → MEMORIZED (named; its final numbers are labeled and vsel numbers carry).
- **R-C1-1 SURVIVAL:** clean A-arms ≥ 2/3 → RI-CARRIES; 1/3 → RI-LOTTERY (champion = the survivor, fragility datum); 0/3 → RI-DEAD-AT-d128 (champion = C0 if clean).
- **R-C1-2 STABILIZER (η cap):** B0 clean AND b1(B0) ≥ mean b1(clean A) − FNC1 → CAP-WORKS (the cap enters the recipe); B0 clean but b1 lower by > FNC1 → CAP-COSTS (kept as contingency); B0 STOPPED → CAP-FAILS.
- **R-C1-3 CHAMPION:** champion = argmax vsel-cold over clean {A*, B0, E0}; bands on its full-test vsel-cold: ≥ 55 → HRM-BEATEN (the floor claim lands at ≤ 4.7M); ≥ 45 → ON-TRACK (register the d192 push); < 45 → PLATEAU (PI consult).
- **R-C1-4 ARITY-AT-vsel:** vsel-cold(C0) vs vsel-cold(D0): ≥ D0 − CNC1 → NATIVE-CARRIES; below by > CNC1 on cold AND on screen-vb → ARITY-HURTS-AT-vsel (the plan §2 fallbacks re-open); mixed → PI consult.
- **R-C1-5 BREADTH:** wide-funnel carrier = argmax verified@128 over clean {C0, D0}; ≥ 88.89 → BREADTH-SCALES-AT-d128; the champion's own verified@128 reported (expected narrow).
- **R-C1-6 PORTFOLIO (labeled fact):** union verified@128 over clean arms on the identical 20k; ≥ .95 → B-M3-BY-PORTFOLIO (our column; protocol named).
- Descriptive (no rules): b1 / t1r@k / majority@k per arm (the protocol table); (ρ,r) per octile; explosion census; η trajectories; depth rider t=256 on the champion (with its explosion read).

## §6. Predictions (numeric, pre-data; lock at registration)
- A-arms (clean): vsel-cold ∈ [42, 52]; b1 ∈ [40, 55]; t1r@128 ∈ [55, 70]; verified@128 ∈ [55, 72]; η final ∈ [.85, .95]; survival 2/3 (credence 55 %).
- B0: clean (70 %); b1 within FNC1 of A (60 %); η pinned at the cap.
- C0: vsel-cold ∈ [36, 46]; b1 ≤ 5; screen-vb v256 ≥ 85 (the native wide funnel exists at vsel); CE end ≥ .05 (the regime rescues the no-RI arm).
- D0: verified@128 ≥ 88.89 (d128 ≥ d96, 65 %); cold ∈ [25, 32].
- E0: vsel-cold ≥ A0 + 3 (60 %). F0: vsel-cold ≥ A0 + 2 with CE ≥ .02 (50 %) — the budget axis either pays or memorizes; either is the finding.
- All arms: A_total closed ≤ 500 nats by 10k; no free-attention deaths; any death carries the census signature (η ≥ .92 at the last monitor) at 70 % credence.

## §7. Cost, calendar, and the freeze rule
Native d128 T16 80k steps ≈ 1.75× the d96 pace → ≈ 1.5–2h pretrain + ≈ 1.5h evals per arm on a v6e-8; 6 core arms ≈ 20h sequential ≈ $140 (US 8) or ≈ 9h on a 16 (4×4) ≈ $125; E0 +$40; F0 +$35. **Total ≈ $130–200** (program lands ≈ $2.5–2.6k of $3.7k). Launch on PI go after the registration lock (analyzer + harness); one night; analysis the next session. Freeze rule stands: champion numbers by ~Sep 10–12 or the section ships in the AAMAS version; the riders owed to the program review (canvas C3X/D4 EqR-statistic evals) ride this node's idle chips.

## §8. PI decisions embedded here
1. Depart from the ARITY-HURTS letter's registered consequence (keep native9; no §2 fallback) — recommended, on the memorization evidence.
2. Keep RI with the η-cap arm and ×3 seeds — recommended; alternative = drop RI (then the champion is C0/D0-class and the per-draw column stays ~0).
3. aug 1000 vs 2000; EMA rider yes/no.
4. E0 (d160) and F0 (long floor): both recommended if the night window allows (≈ +$75).
5. The canvas D0 arm (portfolio partner and canvas-vs-native at d128): recommended.
6. Two-phase split 50k/30k (registered) vs 35k/15k (the pilot's surviving arm): recommend 50k/30k at aug1000 (memorization pressure is 10× lower; the tripwire guards it).
