# Documentation — what to read, in what order (index as of 2026-09-25)

The science of this repository lives in dated, append-only records. Nothing in a README is a claim; every number traces to an analysis script under `runs/analysis/` and a ledger entry. The records are kept in place and verbatim, because the ledger and the reports link to them by file name; a record whose framing was superseded carries a dated status banner under its title instead of being moved or edited. The current statement of the results is the root [`README.md`](../README.md) §2–§3.

## 1. The record of truth

| file | what it is |
|---|---|
| `Design_Ledger.md` | the epistemic source of truth: §0–§4 the rules, the components and the hypothesis register; §5 the append-only status-change log, newest first: one entry per registration, verdict, correction and decision. The 2026-09-19 "ADVERSARIAL PAPER AUDIT" entry and the 2026-09-24 documentation entry are the correction record behind this index |
| `Design_Ledger_Archive_2026-07-16_to_2026-08-21.md` | §5 entries of the ARC era, moved verbatim on 2026-09-02 |
| `Research_Brainstorm.md` · `Research_Brainstorm_Archive_2026-07_to_2026-08-13.md` | the freethinks that precede each registration; their decimation and commitment framings predate the audit and are not admitted claims |
| `Sudoku_vs_ARC_Instrument_Map.md` | the instrument-by-instrument status on both domains as of 2026-09-10 (historical; the paper's Sudoku-versus-ARC treatment is its operational boundary, root README §3 item 6) |

## 2. The paper's evidence runs (2026-09-13 → 09-21), newest first

The cycle is registration → run → verdict by a frozen analyzer → report; each report names its analyzer, artifacts and ledger entry. The paper's Table 1 rows, its interventions and its initialization series come from these records.

| date | file | role |
|---|---|---|
| 09-20 | `Handoff_2026-09-20_Attention_Rows_For_Paper.md` | the attention-arm rows, protocol strings, floors and wording handed to the paper session (consumed) |
| 09-20 | `Note_2026-09-20_Restart_Column.md` | the 128-restart column read as a registered letter with a seed floor: every arm inside the floor, no letter |
| 09-20 | `Report_2026-09-20_SA_Extension_Verdict.md` · `Plan_2026-09-19_SA_Extension.md` | Attention 128 / 192 / 256 extended 30k → 50k and run through the full battery: the paper's attention rows (95.45 / 97.31 / 97.93 at D16, 99.48 / 99.62 / 99.62 at D64 on all 422,786) |
| 09-19 | `Report_2026-09-19_Recipe_Ablation_Verdict.md` · `Plan_2026-09-19_Recipe_Ablation.md` | the Attention 256 recipe ablation at 30k: starts/anchors off, damping/noise off, batch/learning-rate changed (−12.82 / −8.10 points, the largest tested effect); audit addendum on the optimizer-identity wording |
| 09-19 | `Report_2026-09-19_Width_Ladder_Verdict.md` · `Plan_2026-09-18_Width_Ladder.md` · `Note_2026-09-19_Ladder_Adversarial_Pass.md` | the width ladder and the attention-mixer arms at 30k; the exploratory adversarial pass on it |
| 09-19 | `Note_2026-09-19_Commitment_Validity.md` | is "commitment" a property of the models or of the normalization? Outcome ARTIFACT: the earlier confidence readings were a softmax read of StableMax logits; first-iteration high-confidence share 22–40 %, not 85–99 % |
| 09-19 | `Note_2026-09-19_Corpus_Normalized.md` | the 512-puzzle, depth-32 trajectory corpus of nine checkpoints under the training normalization (the paper's correction-timing evidence) |
| 09-19 | `Note_2026-09-19_Fast_Slow_Probe.md` | the state-reset probe on four checkpoints (the paper's original reset study, `tools/lens_fast_slow.py`) |
| 09-19 | `Note_2026-09-19_First_Passage_Tests.md` · `Note_2026-09-19_Why_Decimation.md` | registered predictive tests on banked records; the second carries the audit addendum that retires its title mechanism |
| 09-17 | `Note_2026-09-17_Repair_Radius_Lens.md` | the matched-error repair lens: EqR-produced grids versus count-matched random corruptions on 438 pairs, re-encoded in MLP 192 and EqR (audit addendum on its confidence columns) |
| 09-17 | `Report_2026-09-17_X5_Long_Verdict.md` · `Plan_2026-09-17_X5_Long.md` | the TRM-cell control at MLP 192's parameter count trained to 960k: 60.26 % at D16, 62.40 % at D64 (audit addendum on training-rate arithmetic) |
| 09-17 | `Report_2026-09-17_Sudoku_Pending_Verdict.md` · `Plan_2026-09-17_Sudoku_Pending_Runs.md` | X5 at the 50k budget and the 128-restart banks of C7 / C8 / X5 |
| 09-15 | `Report_2026-09-15_W192_Long_Verdict.md` · `Plan_2026-09-14_W192_Long.md` | the seed-0 MLP 192 lineage continued to 150k with the 38-checkpoint initialization series: the fixed-start decline belongs to the fixed start (94k: 11 vs 124 / 124 of 128) |
| 09-15 | `Note_2026-09-15_FinalA_Reselect.md` | the Night-A arms re-selected on the 512-puzzle validation monitor (supersedes the Night-A augmentation and width contrasts) |
| 09-14 | `Report_2026-09-14_C8_Extension_Verdict.md` · `Plan_2026-09-14_C8_Extension.md` | the lowest MLP 192 seed extended to 50k; decision (B): the paper's MLP row is the budget-matched triple |
| 09-14 | `Report_2026-09-14_Paper_Final_Verdict.md` · `Plan_2026-09-13_Paper_Final_Runs.md` | the paper's final Sudoku runs: the MLP 192 seed triple (95.91 / 94.82 / 95.49 and 99.16 / 98.82 / 99.18) |
| 09-10 | `Plan_2026-09-10_FullSet_Evals.md` | the full-set evaluation plan (executed within the runs above) |

## 3. The audit and the corrections (2026-09-19 → 09-25)

| where | what |
|---|---|
| `Design_Ledger.md` §5, 2026-09-19 "ADVERSARIAL PAPER AUDIT" | the superseding scope and status correction: the softmax-on-StableMax commitment and calibration contrast withdrawn; decimation, "knows failure but not where", the irreversible sure set and the RG explanation not established; SE-RRM's 95.4 % at 16 (Table A6) added; AdamW versus AdamATan2; the SA256 4.9× training-cost claim corrected to about 2.45× |
| `Report_2026-09-19_Adversarial_Writing_Audit.md` · `Decimation_Definition_2026-09-19.md` | the audit itself and the definition note (local only, git-ignored at the PI's word) |
| the banners | `Report_2026-09-08_Frontier_Analysis.md`, `Report_2026-09-09_Champion_Verdict.md`, `Note_2026-09-10_Champion_Mechanisms.md`, `Comparison_Frontier_Compute_Params_2026-09-09.md`, `Sudoku_vs_ARC_Instrument_Map.md`, `Note_2026-09-10_ARC_Instruments.md` and the other records named in the 2026-09-24 ledger entry carry a status banner stating which of their framings were superseded |
| `Design_Ledger.md` §5, 2026-09-24 | this documentation pass: the root README rewritten, this index rebuilt, the banners, the paper-era notes and lens tools added to the repository |

## 4. The Sudoku campaign (2026-08-21 → 09-10), newest first — historical

These records built the digit-field cell and the instrument suite; their measured tables stand as observations of the listed runs, while the "seven laws", the DEC naming and the commitment / decimation vocabulary are superseded (the ledger §5 entries of 2026-09-19 and 2026-09-24).

| date | file | role |
|---|---|---|
| 09-10 | `Note_2026-09-10_Champion_Mechanisms.md` | the explanation-enrichment pass that closed the campaign (commitment and decimation explanations withdrawn) |
| 09-10 | `Comparison_Frontier_Compute_Params_2026-09-09.md` | the frontier tables by parameters and by the old XLA-cost compute measure (superseded by the paper's dense-MAC convention and 50k models) |
| 09-09 | `Report_2026-09-09_Champion_Verdict.md` · `Plan_2026-09-08_Champion_Night.md` | the champion night: the width-384 seed triple, four one-variable arms, the registered letters |
| 09-08 | `Report_2026-09-08_Frontier_Analysis.md` · `Plan_2026-09-07_Frontier_Registration.md` | the field's public Sudoku weights (TRM, CGAR, EqR) at the full protocol through our evaluator; the seven laws first stated (retired as summaries) |
| 09-07 | `Plan_2026-09-07_Instrument_Suite.md` | the instrument catalog (31 instruments) |
| 09-07 | `Report_2026-09-07_NightA_Verdict.md` | Night A: the digit-field cell reaches parity with the field's recipe without digit augmentation (its contrasts re-selected on 2026-09-15) |
| 09-05 / 06 | `Plan_2026-09-05_FinalPhase.md` · `Report_2026-09-06_FieldCheckpoints_Instruments.md` · `Plan_2026-09-05_FieldCheckpoints_Instruments.md` · `Note_2026-09-05_Field_KnowledgeBase.md` · `Note_2026-09-05_NightA_PreVerdict_Labels.md` | the cell's design, the field's checkpoints read through the suite, the field's recipes and literature, the pre-verdict labels |
| 09-02 → 09-05 | `Plan_2026-09-01_Champion_Track.md` · `Report_2026-09-02_ChampionPilot_Verdict.md` · `Plan_2026-09-02_Champion_sportC1.md` · `Report_2026-09-03_Champion_sportC1_Verdict.md` · `Plan_2026-09-04_sportC2.md` · `Report_2026-09-05_sportC2_Verdict.md` | the champion track: the d96 pilot, the d128 native round (our TRM implementation X0 is its A0 arm), the graft night |
| 08-31 → 09-03 | `Program_Review_2026-08-31.md` · `Program_Review_2026-09-02.md` · `Program_Review_2026-09-03.md` | the program reviews (claim grades, protocol tables, the verification question) |
| 08-21 → 08-30 | `Report_2026-08-22_SprintS2_Verdict.md` · `Report_2026-08-23_SprintS2_Wave2_Verdict.md` · `Report_2026-08-24_SprintS2_Wave3a_Verdict.md` · `Report_2026-08-27_PhaseB_Rung1_Verdict.md` · `Report_2026-08-29_PhaseB_Rung2_Verdict.md` · `Report_2026-08-30_PhaseB_Rung2b_Verdict.md` · `Plan_2026-08-27_Rung2_Hardened.md` · `Plan_2026-08-29_Rung2b.md` · `Adversarial_Review_2026-08-27.md` | the Sudoku sprints and phase-B rungs on the RG cell |

## 5. The ARC program (2026-07 → 08-21), the 2026-09-10 analysis pass and the DEC-ARC night — historical

| file | role |
|---|---|
| `Note_2026-09-10_ARC_Instruments.md` | the ARC analysis pass through one suite on our substrates and the released ARC-1 TRM (commitment vocabulary superseded; the flip rates of the later DEC-ARC records were corrected by a factor of 100) |
| `Plan_2026-09-10_DEC-ARC_Build.md` · `Note_2026-09-11_DEC-ARC_Pilot_Read.md` | the DEC-ARC build plan and pilot read; the night that followed (2026-09-15/16) is excluded from the paper and recorded in `Documentation/private/` (git-ignored) |
| `Report_2026-08-21_Reflective_Pass.md` | the reflective pass that closed the ARC ladder and set the two-paper strategy |
| `Report_2026-08-19_Rung1_Verdict.md` · `Report_2026-08-20_Rung1b_Verdict.md` · `Report_2026-08-21_Rung1c_Verdict.md` · `Report_2026-08-13_Frontier_Consolidation.md` · `Report_2026-08-10_State_and_Program.md` | the seeded scale grid and the ARC-era readings |
| `QHRRN2_Architecture.md` · `Thesis_Information_Holography.md` · `S1_Floors.md` · `Gap_Analysis_CompressARC.md` · `Related_Work_Series.md` · `Related_Work_EqR_FPRM.md` · `../README_PHYSICS.md` | the original RG architecture, the thesis statements, the floors, the CompressARC post-mortem, the related-work reads, the physics background |
| `HRRN.pdf` · `HRRN_Overview.pdf` · `Renormalization of Thought.pdf` · `Basin_Measurement_Note_v2.pdf` | the seed documents (December 2025 – January 2026) and the basin measurement note; each claim's status is in the ledger §1 |

## 6. Where the numbers come from

`runs/analysis/` holds every analyzer's output (`saext_verdict.*`, `paperfinal_verdict.*`, `x5long_*`, `champ_verdict.*`, the corpus and lens outputs, the ARC readers' outputs). The frozen analyzers (`tools/analyze_<tag>.py --selftest`) adjudicate byte-untouched against their registration commits; the lenses (`tools/lens_*.py`) and `tools/attention_transfer_study.py` are descriptive and say so. The paper's own recounts of the saved per-puzzle records, its numeric macros and its figures live in the local `paper/` tree with the frozen account (v4.3) and are not tracked here.

## 7. Not tracked in this repository

`paper/` (the manuscript, the frozen account, the code package), `Documentation/private/` (the DEC-ARC night and earlier private notes), `Documentation/Review logs/` (the professor's review-log PDFs), the two 2026-09-19 audit documents named above, `data/`, the run artifacts under `runs/*` other than `runs/analysis/`, and the GCP identity files.
