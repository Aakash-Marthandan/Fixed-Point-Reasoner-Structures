# Documentation — what to read, in what order (index as of 2026-09-10)

The science of this repository lives in dated, append-only records. Nothing in a README is a claim; every number traces to an analysis script under `runs/analysis/` and a ledger entry. Read in this order.

## 1. The record of truth

| file | what it is |
|---|---|
| `Design_Ledger.md` | the epistemic source of truth: §0–§4 the rules, the components and the hypothesis register (H-numbers with their annotations); §5 the append-only status-change log — one entry per registration, verdict, analysis pass and correction, newest first; the ARC-era entries (2026-07-16 → 08-21) are indexed one line each and kept verbatim in `Design_Ledger_Archive_2026-07-16_to_2026-08-21.md` |
| `Research_Brainstorm.md` | the freethinks (the staged questions and the answers from the record) that precede each registration; earlier ones in `Research_Brainstorm_Archive_2026-07_to_2026-08-13.md` |
| `Sudoku_vs_ARC_Instrument_Map.md` | the instrument-by-instrument status on both domains, the Sudoku-vs-ARC differences catalog D1–D15 (the paper's §6 spine, kept current after every pass), and the owed register |

## 2. The Sudoku-Extreme campaign (2026-08-21 → 09-10), newest first

The cycle is registration → run → verdict by a frozen analyzer → physics pass → report; each report names its analyzer, its artifacts and its ledger entry.

| date | file | role |
|---|---|---|
| 09-10 | `Note_2026-09-10_Champion_Mechanisms.md` | the explanation-enrichment pass that closes the campaign: why the champion night came out as it did, instrument by instrument, cross-read with the field |
| 09-10 | `Comparison_Frontier_Compute_Params_2026-09-09.md` | the frontier tables by parameters and by measured inference compute (regenerated from artifacts by `tools/comparison_tables.py`) |
| 09-10 | `Plan_2026-09-10_FullSet_Evals.md` | the full-set evaluation plan (deferred; a PI decision) |
| 09-09 | `Report_2026-09-09_Champion_Verdict.md` | the champion night: the seeded DEC triple, four one-variable arms, the labeled best model, the registered letters and the physics pass |
| 09-08 | `Plan_2026-09-08_Champion_Night.md` | its registration (rules locked in `tools/analyze_champ.py --selftest`; predictions with credences) |
| 09-08 | `Report_2026-09-08_Frontier_Analysis.md` | the field's public Sudoku weights (TRM, CGAR, EqR) at the full protocol through our evaluator; the seven laws first stated on 38–40 grids |
| 09-07 | `Plan_2026-09-07_Instrument_Suite.md` · `Plan_2026-09-07_Frontier_Registration.md` | the definitive instrument catalog (31 instruments) and the frontier run's registration |
| 09-07 | `Report_2026-09-07_NightA_Verdict.md` | Night A: the DEC reaches parity with the field's recipe with no digit augmentation |
| 09-05/06 | `Plan_2026-09-05_FinalPhase.md` · `Report_2026-09-06_FieldCheckpoints_Instruments.md` · `Note_2026-09-05_Field_KnowledgeBase.md` | the DEC's design (the field's loop on our field-structured state), the field's checkpoints through the suite on the Mac, the knowledge base of the concurrent field |
| 09-02 → 09-05 | `Plan_2026-09-01_Champion_Track.md` · `Report_2026-09-02_ChampionPilot_Verdict.md` · `Plan_2026-09-02_Champion_sportC1.md` · `Report_2026-09-03_Champion_sportC1_Verdict.md` · `Plan_2026-09-04_sportC2.md` · `Report_2026-09-05_sportC2_Verdict.md` | the champion track on the native cell: the d128 round, the field-recipe reproduction (X0), the graft night whose failures motivated the DEC |
| 08-31 → 09-03 | `Program_Review_2026-08-31.md` · `_09-02.md` · `_09-03.md` | the program reviews (claim grades, protocol tables, the verification verdict) |
| 08-21 → 08-30 | `Report_2026-08-22_SprintS2_Verdict.md` … `Report_2026-08-30_PhaseB_Rung2b_Verdict.md` · `Plan_2026-08-27_Rung2_Hardened.md` · `Plan_2026-08-29_Rung2b.md` · `Adversarial_Review_2026-08-27.md` | the Sudoku ladder on the native cell (d16 → d96): the price's sign arc (H-43 → H-48), the contractivity collapse, the funnel instruments |

## 3. ARC (the first program, 2026-07 → 08-21, and the analysis pass of 09-10)

| file | role |
|---|---|
| `Note_2026-09-10_ARC_Instruments.md` | the ARC analysis pass: every difference and similarity between the two regimes read operationally through one suite on our substrates and the field's ARC-1 TRM, each with interpretation, justification and physics |
| `Plan_2026-09-10_DEC-ARC_Build.md` | the plan for the last $400: one DEC-ARC build on a v6e-8, the pending instruments and gaps closed first (a plan, not a registration) |
| `Report_2026-08-21_Reflective_Pass.md` | the reflective pass that closed the ARC ladder and set the two-paper strategy |
| `Report_2026-08-19_Rung1_Verdict.md` · `_08-20_Rung1b_` · `_08-21_Rung1c_` · `Report_2026-08-13_Frontier_Consolidation.md` · `Report_2026-08-10_State_and_Program.md` | the seeded scale grid, the four ARC laws (the throat, count-vs-radius, the priced plateau, the transfer-specific dividend) |
| `QHRRN2_Architecture.md` · `Thesis_Information_Holography.md` · `S1_Floors.md` · `Gap_Analysis_CompressARC.md` · `Related_Work_Series.md` · `Related_Work_EqR_FPRM.md` | the original RG architecture (paper 2's lineage), the thesis statements, the analytic floors, the related work |

## 4. Where the numbers come from

`runs/analysis/` holds every analyzer's output (`champ_verdict.{txt,json}`, `champ_physics_20260909.*`, `corpus_lens_*.csv`, `mac_count_20260910.json`, the ARC readers' outputs, …). The frozen analyzers (`tools/analyze_<tag>.py --selftest`) adjudicate registered rules byte-untouched against their registration commit; the physics passes and lenses (`tools/analyze_<tag>_physics.py`, `tools/lens_*.py`) are descriptive and labeled. The ops record is `tools/HANDOFF.md` (the status blocks) and `tools/OPS_RUNBOOK.md`.
