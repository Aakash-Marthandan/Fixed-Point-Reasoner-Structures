# Plan 2026-10-01 — discussion-period experiments for the manuscript (Mac first, pods second)

**Status.** Written 2026-10-01 after the complete read of the submitted manuscript (32 pp build of 2026-09-26, unchanged since) and the discussion-period notes of 2026-09-26 (P1, P2, P1c, P2b, all read and folded in). Reviews arrive in early November; the discussion period allows one more main-text page. Every experiment below is registered in its own note before any row is produced; nothing in this plan is a result. ARC work starts the following week and is out of scope here.

## 1. The thesis the experiments must serve

The manuscript's results are currently read by outside readers as several properties of recurrent networks. The unifying claim they support, stated as a diagnostic:

> Common summaries of recursive computation — the current error count, the current prediction and its scores, the endpoint accuracy, and the size of the state's movement — leave out distinctions that decide whether more computation helps.

Each omission already has a measurement: error count misses configuration (matched-error and structured corruptions); the current scores miss readout-invisible state (score-preserving replacements); endpoint accuracy misses preservation and selection (replay, resets, nested pools); small state movement misses the selector's failures (residual ranking). What is missing is the **connection** between the two strongest findings — configuration-dependent repair and dependence on readout-invisible state — and a **common accounting** that places every experiment on one ledger. The experiments below supply both, plus the robustness items that outside reviewers asked for unanimously.

Established within the tested settings stays separate from hypothesis throughout: the learned-relaxation-with-memory hypothesis is what these tests probe, not what they assume.

## 2. Mac set (CPU, release code, inference only; registered one at a time)

| id | experiment | question it connects | cost | status |
|---|---|---|---|---|
| **P3** | One-shot, score-preserving replacement of the readout-invisible slow state after the first iteration, across the corruption ladder (uniform, clue-compatible random, mutually consistent random, the released model's own grid) at matched error counts; graded rotations as a dose–response; a reassembly sham as the numerical control; the same one-shot edit on the 256 shared first states to compare with the repeated edit of P2 | §4 repair ↔ §4.2 state: does repair of difficult configurations draw more on state beyond the displayed answer? Is P2's loss a temporary setback or cumulative disruption? | ~6–8 h wall, three widths in parallel | registered 2026-10-01 (`Note_2026-10-01_Rebuttal_P3_Registration.md`); launching |
| **P4** | Freeze-time ladder: messages frozen from iteration t₀ ∈ {2, 4, 8} instead of 1 (the P2b hook, recording at t₀) on the 256 shared first states; discovery after the freeze against each puzzle's intact completion time | §4.1 communication ↔ §3 abrupt completion: is updated exchange needed until completion, or only early? | ~1 h | to register after P3 launches |
| **P5** | Access robustness at 94k: the post-50k distribution of fixed-start versus Gaussian-start successes across the 38 checkpoints (saved records), extra Gaussian draws at 94k, and the cross-device check the manuscript names as missing | §5 access: the 94k contrast is one checkpoint's post-hoc minimum; show the distribution | saved records + ~1 h | to register |
| **P6** | The common accounting from saved records: for every configuration, U (ever correct), C (correct candidate available to the return rule), A (returned correct), with 1 − A = (1 − U) + (U − C) + (C − A); unmeasured cells marked; the majority-vote and verifier selection curves on the saved nested banks | §3, §5, §6, §7 on one ledger; C − A is the selector's failure | saved records only | to register |

P3 is first because it is the connection experiment; P6 is the figure the revision is organized around; P4 and P5 are cheap and answer standing critiques.

## 3. Pod set (the GPU project; nothing launches without a registration note and an explicit go)

Verified read-only on 2026-10-01: the GPU project has quota for 16 on-demand and 64 preemptible A100s and 32 L4s in us-central1. **The project is shared with other users: none of its existing VMs is started, stopped, resized or deleted by us.** Our work uses one new VM with an unmistakable name (`rr-rebuttal-<date>`), created for a run and deleted at its end, with a deadline guard scoped to that name only (the TPU watchdog deletes by project and must not be pointed at this project). Efficiency: spot/preemptible A100 for training with checkpoint-resume, an L4 or the CPU for inference-only items, and nothing left running idle. The release code pins JAX 0.10.2 (CPU/TPU wheels); the GPU path needs the CUDA plugin and a bring-up smoke before any run is trusted. The TPU project remains available at the recorded rates (about $115 per 50k-update run per width on a v6e-8).

| id | experiment | question | first step |
|---|---|---|---|
| **G0** | GPU bring-up on a **new** VM (never an existing one): create `rr-rebuttal-<date>` from a Deep Learning image with one spot A100, install the CUDA JAX plugin, run the training smoke (1,000 updates) and the evaluation smoke, record throughput and the CPU-vs-GPU prediction agreement on 128 puzzles, then stop or delete it | can the GPU project carry a 50k run, and at what wall time and price? | the gating step for everything below |
| **G1** | Training arms with paired seeds at Attention 128: {randomized starts on, off} × anchors fixed × seeds {1, 2} (plus the existing seed 0 reference), same optimizer, schedule, data order and budget; evaluate every arm on the full battery: fixed-start full test at 16/64, Gaussian-start test, nested restart banks (coverage, selected accuracy, missed answers), and the matched-error repair assay | does training-state exposure reproducibly change inference reliability, separately from single-start accuracy? Also answers "one seed per width" | after G0; four to six runs |
| **G2** | U for the restart banks: re-run a registered subsample of the TRM and attention banks with intermediate readouts recorded, so the U − C term of P6 is measured rather than marked | closes the one unmeasured cell of the accounting | after G0; inference only |
| **G3** | Full-test accuracy from a Gaussian start for the three benchmark attention models at depth 64 | the fixed start is never a training start; show the access result at full scale | after G0; inference only |
| G4 (optional) | A common-budget attribution arm for the SE-RRM comparison (their mixer in our recipe is already measured; the missing arm is our mixer under their recipe) | the benchmark advantage's source | only if reviews demand it |

Order: G0 → G1 (the long pole; launch first) → G3 and G2 while G1 trains. Standing policies apply: one pod at a time, the deadline file and the billing watchdog re-armed before launch, verify at source, the launch ritual in the ops notes.

## 4. What each outcome changes in the manuscript

- **P3 configuration-dependent:** §4 and §4.2 become one claim (difficult configurations draw more on state beyond the answer), the hypothesis paragraph gains its first direct support, and the overview figure shows it. **P3 independent:** the dependence is general; the one-shot versus repeated contrast still replaces "ongoing dependence" with a measured recovery, and the hypothesis stays a hypothesis.
- **P4:** "continued exchange" gains a time scale relative to completion, or loses it.
- **P5:** the 94k sentence becomes a distribution statement with its interval, or is narrowed.
- **P6:** the new overview figure and the page-one framework paragraph; the training–selection table moves into the main text on the added page.
- **G1:** the one-seed limitation is lifted for width 128 and the training-exposure sentence becomes causal within the tested arms, or is dropped.

## 5. Rules carried over

Registration before data, with the rule, the letters, the credences and the wording per outcome fixed in the note; bitwise gates against archived records; no shared tool edited while a run is live; PID-based waits; outputs under `runs/analysis/` (ignored) with the report files tracked; the public record names no venue, title or headline number.

## 6. Addendum (2026-10-01 ~10:20Z) — the PI's decisions; P7–P9; pods reprioritized

- **The thesis is finalized after the rounds below, not before.**
- **Three more Mac experiments.** All three are registered in `Note_2026-10-01_Rebuttal_P7_P8_P9_Registration.md` and run on the Mac beside P3–P5, at lower priority:
  - P7: cell-level repair by local evidence, from saved records; read.
  - P8: the completing step's intact control and a score-preserving slow-state edit.
  - P9: state swaps at identical displayed answers, at 94k.

  A plateau test with edits timed to each puzzle's completion (P10) is designed after P3 reads.
- **Pods.**
  - G1's seed arms are deferred. The width ladder already supplies three independent initializations of one recipe, and the closest comparators report single runs.
  - The first priority is G4, promoted: attribute the accuracy gain over SE-RRM.
  - SE-RRM's released code (read 2026-10-01) differs from ours in:
    - one recurrent state updated 18 times per step, with no slow/fast split;
    - no damping;
    - batch 272 at lr 1e-4;
    - about 10M slot-steps of training;
    - random 5 % halting instead of a halting head;
    - dropout 0.2;
    - AdamATan2;
    - a bfloat16 forward pass.
  - The rounds, each registered before launch:
    - Round 1 (our code, no new model code): SE-RRM's batch and learning rate at 30k steps, and at matched rows.
    - Round 2 (a new cell variant): the single-state recurrence; halting; dropout.
    - Round 3 (GPU, their code): their released command, and their model at our batch and budget.
