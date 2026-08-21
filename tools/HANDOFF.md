# HANDOFF — TPU ops phase (persistent; the "CURRENT CAMPAIGN" block is rewritten per campaign)

**Read this first, then `tools/OPS_RUNBOOK.md`.** The repo outranks conversation
memory. The ops model runs the campaign to completion and STOPS; the analysis
phase is reserved for the PI's next model switch.

## CURRENT CAMPAIGN — SPRINT S2 (Sudoku-Extreme wave 1; launched 2026-08-21; ledger 'SPRINT S2 LAUNCH REGISTRATION')

**What it is:** the series HRM→TRM→EqR→FPRM as an ablation ladder on our substrate on the
real Sudoku-Extreme benchmark (1k seeded train / FULL 423k test, exact accuracy). Tag `sport2`,
GCS `gs://qhrrn2-rescue/sport2/`, ONE spot v6e-8 pod, `tools/campaign.env` = `campaign_sport2.env`.
Arms S0 S1 S2 S3 S4 S5 S6 S7 (d16, 20k steps; S3/S7 T12, S4 T24), all 8 in ONE chip-pinned wave.
Chain `tools/chain_sport2.sh`: PHASE1 pretrain wave (~0.5–2 h; S4 is the long pole) → `PHASE1-OK` →
PHASE2 evals per arm on its own chip (stratified-512 @ t 6/64/256 k=16 + FULL test @ t 6/64;
~1–1.5 h) → `PHASE2-OK` → PHASE3 probes (~0.5 h) → `PHASE3-OK` → `RESCUE-OK` → `CHAIN-SPORT2-COMPLETE`
→ the loop tears down. ≈ 3–4 h ≈ $25–45 absent churn.

**Signatures:** `=== PRETRAIN S3 hh:mm === chip 3 flags: ...`, `PRETRAIN-S0-OK`, `SKIP-S0 (GCS complete)`,
`RESUME-S4 from live ckpt`, `RESTORE-S2 evals from GCS`, `EVAL-S1-OK`, `PROBE-S5-OK`.
`PRETRAIN-Sx-FAILED rc=` / `PROBE-Sx-FAILED` = one arm failed; the chain CONTINUES (others unaffected).
After the loop's `down rc=0`: §6 with `sport2` substituted — pull `gs://qhrrn2-rescue/sport2/sport2_final.tgz`
(expect `pretrainsport2_*` ×8 with metrics.jsonl, `sxeval_psport2*` ×8 (summary_all.json per depth),
`sudprobe_psport2*` ×8 at 512 rows), print-only inspection, ONE ops ledger line, ONE commit,
NO analysis (`tools/analyze_sport2.py` is reserved for the analysis pass). Then wave 2 / d96 / convert
per the launch entry. Everything in §1–§5 applies unchanged with tag sport2.

## PAUSED — bad TPU weather 2026-08-20 17:40Z (23:10 IST) — PI: "continue later"

Fleet ZERO verified (nodes + QRs, all 8 door zones); supervisor stopped; cron + all three
monitors stopped; launchd watchdog left running (billing backstop; no node = nothing to do).
Rung-1c PAUSED at 8/10 pretrains, NOT complete.

**State (banked in `gs://qhrrn2-rescue/rr1c/`):** 8 complete ckpts — all 4 H-40 corners
(A10s0/A10s1 d64@40k, A8s1/A8s2 d48@53k) AND all 4 A5-class d96 anchors (A11s0/A12s0/
A11s1/A12s1). **The two decisive datasets (width×budget 2×2 + d96 baseline) are safe.**
Remaining: A13s0/A13s1 (the β-rescue pair, H-41) + the 44-job battery phase.

**Why paused:** severe us-east1-d evening spot churn (~5 preemptions in ~2 h, several
mid-bring-up) + intermittent dry spells — the classic evening pattern (yesterday it cleared
by morning). Not a config problem. Deadline was extended to 09:01Z before pausing.

**RESUME (morning or when spot weather improves; ARMS order still load-bearing):**
```bash
cd /Users/aakash/Projects/HRRN
bash tools/pod.sh status                                        # expect node ABSENT, supervisor NOT RUNNING
echo $(( $(date -u +%s) + 16*3600 )) > runs/tpu_deadline.txt    # refresh the deadline
nohup bash tools/pod.sh supervise 16 >/dev/null 2>&1 &          # SKIPs 8 arms, resumes A13s0 from live ckpt, then A13s1 + batteries
tail -f runs/pod_qhrrn2-pod2.log
```
Then re-arm the hourly heartbeat cron + all three monitors (§2 verbatim — edge, watchdog-
inventory, and `tools/unstick_watch.sh`). Only ~2 pretrains + batteries remain (<3 h on a
stable pod). Everything else in this handoff is unchanged; ops-only, NO analysis
(tools/analyze_r1c.py reserved for the PI's Fable switch).

## COMPLETE 2026-08-21 08:29Z (13:59 IST) — rung-1c CHAIN-R0-COMPLETE; ops closed

Fleet ZERO; all 44 battery files 48/48 + 10 ckpts in `gs://qhrrn2-rescue/rr1c/` and local `runs/`; one ops-close ledger line committed. Finished in the first stable window after a ~9h overnight spot drought (covered by hourly resume timers; durability held everything). NEXT = ANALYSIS (fresh session): `tools/analyze_r1c.py` untouched → H-40/H-41 verdict + d96 anchors → d96 registration. This block is historical.

## HISTORICAL — RUNG 1c (launched 2026-08-20; COMPLETE; the WIDTH×BUDGET completion, the LAST rung-1 side quest before d96)

**What it is (ledger: 2026-08-20 RUNG-1c LAUNCH REGISTRATION):** completes the A4-class
2×2 (H-40: budget vs width), builds the A5-class 40k anchors (the d96 baseline row), and
tests the β-rescue (H-41). Tag `r1c`, GCS `gs://qhrrn2-rescue/rr1c/`, one spot v6e-8 pod.

**Arms in order (decisive corners FIRST, banked early against churn):**
A10s0 A8s1 A10s1 A8s2 (the 2×2: d64@40k ×2, d48@53k ×2, all A4-class) →
A11s0 A12s0 A11s1 A12s1 (A5-class 40k anchors d48/d64) → A13s0 A13s1 (d48@53k β=6e-5).
10 pretrains (@40k ≈ 29 min, @53k ≈ 38 min at ~23 it/s ⇒ ~5.4 h) → `PHASE1-OK` →
`PHASE2: 44 battery jobs` (lad/rg/rb/rt ×10 + e1e3 on A8s1 A8s2 A10s0 A10s1) in 6 waves
(~2.5 h) → `RESCUE-OK` → `CHAIN-R0-COMPLETE` → the loop tears down. ≈ 8 h ≈ $55 absent churn.

**Right now (07:55Z):** node CREATED in us-east1-d 07:47Z, bring-up in progress;
supervisor pid in `runs/pod_supervisor.pid`; deadline 01:43Z 21-Aug. THREE session
watches armed (survive model switches, NOT app restarts — re-arm per §2 below):
edge monitor, watchdog-inventory monitor, and the NEW self-healing stall watch
(`tools/unstick_watch.sh` under a Monitor: when a bring-up step hangs past its
normal duration on a POSITIVELY dead node, it kills the hung child so the loop
fails fast and re-hunts — the 08-19/08-20 silent-stall incidents cannot recur;
it emits an event only when it acts). ETA absent churn: PHASE1-OK ≈ 13:25Z,
COMPLETE + auto-teardown ≈ 15:45Z (21:15 IST); with yesterday's evening churn
pattern budget up to ~23:00 IST. Heartbeat cron at :23 carries a manual
unstick instruction as the backstop of the backstop.

**Known-good signatures:** every arm trains fresh (no SKIPs expected this campaign unless
resuming after a preemption — then SKIP/RESUME lines are the durability stack working).
Everything else in this handoff (§1-§6: first commands, re-arm prompts, log signatures,
decision table, never-list, completion procedure) applies unchanged with tag r1c — in §6
substitute `rr1c` for the bucket and `pretrainr1c_*`/`*_pr1c*` for the dirs (expect 10
ckpts + 44 battery files: 40 @48 rows + 4 e1e3). Analysis = `tools/analyze_r1c.py`,
reserved for the PI's Fable switch.

## 1. First three commands (any time, and after any app restart)

```bash
cd /Users/aakash/Projects/HRRN && bash tools/pod.sh status
tail -3 runs/pod_qhrrn2-pod2.log
launchctl list | grep qhrrn2          # "-  0  com.qhrrn2.tpuwatchdog" = watchdog loaded
```
If `status` says `supervisor: NOT RUNNING` and the campaign is not complete:
`nohup bash tools/pod.sh supervise 16 >/dev/null 2>&1 &` (adopts; refuses to
start twice; app restarts CAN kill it — the only manual action expected). Never
run two. If the deadline has passed, extend it FIRST
(`echo $(( $(date -u +%s) + 14*3600 )) > runs/tpu_deadline.txt`) or the loop exits
without creating.

## 2. Re-arm the session layers (they die with the session)

**Hourly heartbeat** — `CronCreate`, cron `17 * * * *`, prompt verbatim:

> HOURLY TPU HEARTBEAT (standing PI directive, generic — never run-specific). Do ALL of: (1) `cd /Users/aakash/Projects/HRRN && bash tools/pod.sh status` (node state across zones, supervisor alive?, deadline margin, GCS bank rr1b, remote job + per-arm DONE/partial + battery row counts). (2) From the remote detached.log progress (last `step N` line; ~23 it/s at d64 and d48) compute per-ARM completion fraction and ETA — this campaign (rung 1b) has 7 pretrains (A6s0 A7s0 A5s1 A6s1 A7s1 A5s2 A8s0, each 53,333 steps ≈ 38 min) then 33 battery jobs in 5 waves of 8 (~25 min/wave; the last wave has 1 job); rt-only arms A4s1 A4s2 A3s1 A9s1 A9s2 SKIP pretrain. (3) `tail -3 runs/tpu_status_log.txt` and `.venv/bin/python tools/spend_report.py --since 2026-08-19 | tail -3` for spend. (4) Report to the PI in 2-4 sentences + a table with EACH active process, completion fraction, and concrete ETA — EVEN IF UNEVENTFUL. (5) If the supervisor (runs/pod_supervisor.pid) is NOT alive while a node exists or the campaign is incomplete: restart it with `nohup bash tools/pod.sh supervise 16 >/dev/null 2>&1 &` and say so. (6) If the pod.sh log shows COMPLETE + `down rc=0`: verify ZERO TPUs + ZERO queued-resources in all zones, then do EXACTLY §6 of tools/HANDOFF.md (pull gs://qhrrn2-rescue/rr1b/r0_final.tgz with `gcloud storage cp`, list, extract, print-only inspection, one ops ledger line, ONE commit — NO analysis, NO verdicts: the analysis phase (tools/analyze_r1b.py) is reserved for the PI's next model switch) and delete this cron job (CronDelete). (7) If the hard-delete deadline is within 2h of expected completion, extend runs/tpu_deadline.txt and report it. Read tools/HANDOFF.md if anything is unclear.

**Event monitor** — `Monitor` (persistent), command verbatim:
```
cd /Users/aakash/Projects/HRRN && tail -n0 -F runs/pod_qhrrn2-pod2.log | grep --line-buffered -E '^2026' | awk -F'\|' '/CREATED in|canary FAILED|DOWN |COMPLETE|CHAIN-R0|NEEDS|died|sick|SSHFAIL|unreachable|all zones dry|EXIT|relaunch|not READY/{print;fflush();next} NF>=4{if($4!=last){print;fflush();last=$4}}'
```
It emits one line per state/phase change (steady 5-min polls are deduplicated).

**Watchdog-inventory monitor** — `Monitor` (persistent), command verbatim:
```
tail -n0 -F runs/tpu_status_log.txt | grep --line-buffered -vE 'PROBE-FAIL' | awk '{cur=$0; sub(/^[^|]*\| /,"",cur); if (cur!=last) {print; fflush(); last=cur}}'
```
Independent channel: fires on real fleet changes even while the supervisor is blocked mid-call.

**Self-healing stall watch** — `Monitor` (persistent), command verbatim:
```
cd /Users/aakash/Projects/HRRN && bash tools/unstick_watch.sh
```
Auto-unsticks a bring-up hung on a positively dead node (kills the hung CHILD only,
never the supervisor; never acts on READY/CREATING/unknown). An `UNSTUCK:` event =
it worked and the loop is re-hunting — no action needed. `SUPERVISOR DEAD` event =
restart per §1.

## 3. Normal log signatures (runs/pod_qhrrn2-pod2.log)

- Hunting: `node ABSENT everywhere — hunting` → `CREATE … in <zone>` → either `no capacity in <zone>` ×4 + `all zones dry — sleeping 8 min` (normal spot weather; repeats) or `CREATED in <zone>` → `UP (bootstrap+data)` (~7 min) → `CANARY` (~2 min) → `LAUNCH chain` → `launch: detached + verified` → `READY <zone> | RUNNING <pid> | …`.
- Phase 1: `READY … | RUNNING <pid> | === PRETRAIN A7s0 hh:mm === | step N … it/s`; `RESUMED from … at step N` after a preemption (≤5 min lost); `SKIP-<arm>` for completed or supplied ckpts.
- Phase 2: `PHASE1-OK`, `PHASE2: 33 battery jobs`, then `wave done rc=0 hh:mm` ×5. **`rc=1` = a job in that wave failed — the chain CONTINUES; do not intervene.**
- Completion: `COMPLETE (sentinel) in <zone>` → `final object present in GCS` → `DOWN … (campaign complete)` → `down rc=0` → supervisor exits 0; watchdog log then reads `none`.
- Preemption: `node PREEMPTED … — down` → `DOWN …` → next poll hunts. A preemption inside bring-up shows `canary FAILED` (the SSH to the dying node hangs up to ~13 min before the bounded call fails) → `DOWN … (bring-up failed …)` → hunt. **Nothing for you to do.**
- `read UNKNOWN (xN) — network? not acting` and watchdog `PROBE-FAIL` = API blips; nothing to do.

## 4. Decision table for YOU (the loop does everything in OPS_RUNBOOK §3)

| you see | do |
|---|---|
| supervisor NOT RUNNING, campaign incomplete | restart it (§1). Report. |
| supervisor exited 3 (`chain died 3x … EXITING`), node deleted | read the last relaunch's remote failure from `runs/cloud/qhrrn2-pod2-*.tgz` (down's rescue) `detached.log`. Transient (GCS/SSH hiccup) → restart `supervise`. A real crash in `pretrain.py`/probes → STOP, report the traceback to the PI, do NOT patch code. |
| the chain exits 1 right after the LAST wave (`wave done … rc=0` then `IDLE 1`) | this was the stager-kill bug; it is FIXED in `chain_r0.sh` (both `pkill -P … || true`). If it ever recurs: `r0_final.tgz` absent → the loop relaunches (resume replays waves in ~1 min each) — let it; if it fails 3×, run the chain's PHASE-3 rescue by hand exactly as the 08-19 ops entry describes (tar + `gsutil cp` to `$GCS/r0_final.tgz`), then the loop's GCS check tears down. |
| supervisor exited 2 (deadline) with campaign incomplete | only if the PI wants to continue: extend the deadline, restart `supervise`. Report spend. |
| `all zones dry` for > 2 h | report; keep waiting. Do NOT create on-demand, QRs, or a second pod. |
| any second node anywhere | should not exist; `.venv/bin/python tools/dispatcher.py down --name <name> --zone <zone>` and report. |
| anything confusing | PI standing order: `bash tools/pod.sh down` and wait. Work is banked; ≤5 min lost. |

## 5. Never (during this phase)

Two supervisors · editing `tools/campaign.env` (ARMS order is load-bearing) · touching
the running chain (`pod.sh stop/relaunch` only if the PI asks) · on-demand, queued
resources, a second pod · running `tools/analyze_r1b.py` for a verdict · patching
`pretrain.py`, `chain_r0.sh` or probes · deleting `runs/tpu_deadline.txt` · `gsutil -m`
pulls · more than ONE commit (the ops close).

## 6. Completion procedure — and where you STOP

After the loop's `down rc=0` and exit 0:
```bash
for z in us-east1-d us-east1-c us-east5-b us-central1-a us-central2-b us-west1-c us-west4-a asia-east1-c; do
  echo "$z: $(gcloud compute tpus tpu-vm list --zone=$z --project=quantum-llm --format='value(name,state)' 2>&1 | tr '\n' ' ')"
  gcloud compute tpus queued-resources list --zone=$z --project=quantum-llm --format='value(name,state)' 2>/dev/null
done                                                    # every line must be empty
mkdir -p runs/_rr1b_final && gcloud storage cp gs://qhrrn2-rescue/rr1b/r0_final.tgz runs/_rr1b_final/
tar tzf runs/_rr1b_final/r0_final.tgz | sed 's#^runs/##; s#/.*##' | sort | uniq -c   # expect pretrainr1b_* x12 (7 new + 5 supplied), lad/ladrg/ladrgb_pr1b* x7, ladrt_pr1b* x12, wave logs
tar xzf runs/_rr1b_final/r0_final.tgz -C .              # extracts into runs/ (note: pretrainr1b_A4s1 etc. are COPIES of supplied ckpts)
.venv/bin/python tools/inspect_ckpt.py runs/pretrainr1b_A6s0 runs/pretrainr1b_A7s0 runs/pretrainr1b_A5s1 runs/pretrainr1b_A6s1 runs/pretrainr1b_A7s1 runs/pretrainr1b_A5s2 runs/pretrainr1b_A8s0   # print only
for f in runs/*_pr1b*/results.jsonl; do echo "$f $(wc -l < $f)"; done   # row counts, print only (expect 48 each)
.venv/bin/python tools/spend_report.py --since 2026-08-19 | tail -4
```
Then ONE ledger line under §5 (newest first, right after the `## 5.` header): date,
"RUNG-1b CHAIN COMPLETE — ops close", completion time, preemptions/relaunches
(count `DOWN`/`CREATED`/`relaunching` lines since the `RUNG-1b CAMPAIGN START`
marker in the log), spend, "zero TPUs verified", "data local in runs/ + GCS rr1b;
analysis reserved for the next session". Commit (`git add Documentation/Design_Ledger.md`
only; `runs/` stays untracked) and push. Delete the cron. **STOP.** Tell the PI
the campaign is closed operationally and the analysis phase can start.

## 7. Facts you will be asked about

- Registration: 2026-08-19 RUNG-1b LAUNCH REGISTRATION (rules R1b-1..R1b-4, locked pre-data); analyzer `tools/analyze_r1b.py` self-test 15/15.
- Chain changes (arms A6–A9, `R0_RT_ALL`/`R0_RT_ONLY`) were harness-verified offline: exact per-arm flags, SKIPs, 33-job queue, 5 waves, sentinel rc=0.
- Supplied ckpts in `rr1b/`: A4s1/A4s2/A3s1 (rung 1, d64) and A9s1/A9s2 (= rung-0 d48 A4s1/A4s2, md5-verified).
- Rung-1 verdict (for context only): width ceiling confirmed at d48; H-38/H-39 are what this campaign tests.
