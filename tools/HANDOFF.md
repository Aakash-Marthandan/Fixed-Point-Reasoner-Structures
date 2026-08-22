# HANDOFF — TPU ops phase (persistent; the "CURRENT CAMPAIGN" block is rewritten per campaign)

**Read this first, then `tools/OPS_RUNBOOK.md`.** The repo outranks conversation
memory. The ops model (Opus) runs the campaign to completion and STOPS; the analysis
phase (Fable) is reserved for the PI's next model switch. **ALL ETAs and clock times
reported to the PI are in IST (UTC+5:30), with UTC in parentheses** (PI, 2026-08-21).
HARD BOUNDARY during ops: no `tools/analyze_sport2w2.py` (nor analyze_sport2.py), no reading/quoting accuracy
values from result files, no verdicts, no tuning, no chain/env edits, no second pod.

## CURRENT CAMPAIGN — SPRINT S2 WAVE 2 (Sudoku-Extreme; launched 2026-08-22; ledger 'SPRINT S2 WAVE 2 LAUNCH REGISTRATION')

**What it is:** the wave-1 verdict's levers on the PLAIN (β=0) base + the PRICE × SCALE surface
(PI 2026-08-22: price effects may be scale-dependent — width/budget/dose axes, not one cell), on ONE spot
pod: **v6e-16 (2 workers × 8 chips) hunted first, v6e-8 as the fallback** — `ACCEL_LIST="v6e-16 v6e-8"` in
`tools/campaign.env` (= `campaign_sport2w2.env`); the 16 is DEMOTED for the supervisor's life after
`BIG_MAX_STRIKES=2` bring-up failures/preemptions (PI: "fall back if 16 gets preempted too much"). Tag
`sport2w2`, GCS `gs://qhrrn2-rescue/sport2w2/`, chain `tools/chain_sport2w2.sh` (launched ONCE PER
WORKER with `CHAIN_WORKER=w CHAIN_WORKERS=n`; worker 0 runs `ARMS_W0`, worker 1 `ARMS_W1`; on a v6e-8 one
worker runs both lists, 2 jobs per chip). Jobs (16): W13 W2 W3 W1 W9 W5 W4 W8 | scan:S5:64:256
scan:S4:64:256 scan:S7:64:256 scan:S5:6:256 W6 W7 W1s1 W4s1 (arm meanings in the chain header). Each
training arm = pretrain → evals (strat t6/64/256 k16, full t6/t64, val t64, retention t8) → probe (not W7:
box4 layout) → GCS; then PHASE4 per worker (best-of-my-arms 20k-subsample k=128 t=64, 8 shards); then the
COMPLETION GUARD (the last worker to find every arm's artifacts in GCS builds `sport2w2_final.tgz` + emits
`CHAIN-SPORT2W2-COMPLETE`; a worker that finishes its share earlier emits `CHAIN-SPORT2W2-WORKER-DONE` and
exits 0 — pod.sh treats that as WDONE, not a crash). ETA absent churn ≈ 7–9 h on a v6e-16 (long poles:
W13 100k steps, the d32 arms' evals), ≈ 2× on a v6e-8; ≈ $13.6/h (16) or $6.8/h (8).

**Signatures (per worker log):** `=== SPORT2W2 START … worker=w/n my_jobs=[…]`, `chip c queue: …`,
`=== PRETRAIN W1 hh:mm === chip c flags: …`, `=== SCAN S5 t=64 k=256 …`, `PRETRAIN-Wx-OK`, `SKIP-Wx (GCS
complete)`, `RESUME-Wx from live ckpt`, `EVAL-Wx-OK`, `PROBE-Wx-OK`, `PROBE-SKIP-W7`, `SCAN-S5-t64-k256-OK`,
`QUEUES-DONE`, `PHASE4: best arm …`, `PHASE4-OK`, `RESCUE-OK`, `CHAIN-SPORT2W2-WORKER-DONE` /
`CHAIN-SPORT2W2-COMPLETE`. `PRETRAIN-Wx-FAILED rc=` = one arm failed, the others continue (the campaign
then never COMPLETEs: after 3 relaunches the supervisor exits 3 "needs eyes" — report which arm, do NOT
patch). Supervisor lines: `CREATE … (v6e-16 spot)`, `CREATED in <zone> (v6e-16, 2 worker(s))`,
`launch w0/w1: detached + verified`, `READY <zone> | RUNNING 2/2 worker(s) | [w0 …] [w1 …]`,
`(+1 done)`, `STRIKE n/2 against v6e-16`, `DEMOTE v6e-16`.

**Per-worker progress (heartbeat):** replace the wave-1 ssh loop with, for each worker w in 0..n-1
(n from `runs/pod_workers.txt`): `gcloud compute tpus tpu-vm ssh qhrrn2-pod2 --zone=<zone> --worker=$w
--project=quantum-llm --command "cd ~/qhrrn2 && for a in W13 W2 W3 W1 W9 W5 W4 W8 W6 W7 W1s1 W4s1 W5gen;
do [ -f runs/wave_pre_$a.log ] || continue; printf '%s: ' $a; grep -E 'step |DONE' runs/wave_pre_$a.log |
tail -1 | cut -c1-80; echo; done; grep -E 'SCAN-.*-OK|EVAL-.*-OK|PROBE-.*-OK|PHASE4|QUEUES-DONE|WORKER-DONE|
COMPLETE' runs/detached.log | tail -8"` (bounded; perl alarm 150). Per-arm steps: W13 100k, W1/W9/W1s1
50k, W2/W3 30k, others 20k (W5 = W5gen 20k then W5 20k).

**Analysis boundary unchanged:** `tools/analyze_sport2w2.py` (self-test 21/21) is reserved for the analysis
pass; no accuracy values quoted during ops. §6 completion (sport2w2 version): pull
`gs://qhrrn2-rescue/sport2w2/sport2w2_final.tgz`; expect `pretrainsport2w2_*` ×13 (12 arms + W5gen, ckpt
at the registered steps), `sxeval_psport2w2*` ×12 (strat_t6/t64/t256 + full_t6/t64 + val_t64 + ret_t8 =
7 summary files each), `sudprobe_psport2w2*` ×11 (no W7), `sxbreadth_*` ×4, `sxbreadth20k_*` ×1–2;
print-only `inspect_ckpt.py` on a few arms; ONE ops ledger line; ONE commit; STOP.

## PREVIOUS CAMPAIGN (historical) — CURRENT CAMPAIGN — SPRINT S2 (Sudoku-Extreme wave 1; launched 2026-08-21; ledger 'SPRINT S2 LAUNCH REGISTRATION')

## ⏸ PAUSED — overnight spot drought 2026-08-22 00:56 IST (19:26Z 08-21) — PI: "pause till morning, resume 7am IST, 1h gaps"

Fleet ZERO verified (all 8 door zones, nodes + QRs). Supervisor KILLED, meter loop stopped, PHASE1 watcher
+ all 3 monitors stopped, heartbeat cron deleted. launchd watchdog left running (billing backstop; no node = nothing to do).
**State: 7/8 arms DONE + banked in `gs://qhrrn2-rescue/sport2/` (S0/S1/S2/S3/S5/S6/S7 all `_ckpt.pkl`).
Only S4 (T24) remains — resumes from `S4_ckpt_live.pkl` (18:13Z, ~step 14,200 = ~71%).** PHASE2/PHASE3 not started.
**Why paused:** us-east1-d evening churn escalated to preempting nodes DURING bring-up (18:25Z + 19:01Z both canary-FAILED)
plus intermittent all-zones-dry — no node could survive its ~8-min bring-up. Classic overnight trough (clears by morning; cf 08-20).

**RESUME** (fires via session cron `0 7-11 * * *` IST = 7,8,9,10,11 AM IST — the 7am start + hourly fallback; each fire self-checks and self-deletes when done). A fresh session does it manually:
```bash
cd /Users/aakash/Projects/HRRN
echo $(( $(date -u +%s) + 14*3600 )) > runs/tpu_deadline.txt          # 1. extend deadline FIRST
bash tools/pod.sh status                                              # 2. expect node ABSENT, supervisor NOT RUNNING (unless already resumed)
# if CHAIN-SPORT2-COMPLETE already in gs://qhrrn2-rescue/sport2/ (final object) -> skip to §6 close instead
nohup bash tools/pod.sh supervise 14 >/dev/null 2>&1 &                # 3. resume: SKIPs the 7 done arms, resumes S4 from live ckpt, then PHASE2/PHASE3
nohup .venv/bin/python tools/sport2_meter.py --interval 300 >runs/sport2_meter.log 2>&1 &   # 4. meter back
```
Then re-arm §2 (heartbeat cron + edge/inventory/unstick monitors) and, optionally, a fresh PHASE1 watcher bounded to the NEW log tail. If morning is STILL churny (nodes preempt at bring-up), the supervisor keeps hunting on its own; the hourly cron just re-verifies it's alive. On `CHAIN-SPORT2-COMPLETE` → §6 close (SPORT2 version) → STOP. Delete the resume cron once PHASE1-OK is reached or the campaign completes.


**INCIDENT + RECOVERY 11:30–11:50Z / 17:00–17:20 IST (Opus):** `pod.sh stop` killed the workers but
the chain script itself survived (v_stop's pkill pattern was r0-specific), raced through its empty
phases and uploaded a VACUOUS `sport2_final.tgz` + sentinel at 11:31:08Z → the supervisor read COMPLETE,
tore the node down and exited. No data lost: the six running arms' ckpts were pushed to GCS-live at
11:31:13–20Z (S0/S1/S2/S5/S6 ≈ step 11.8k, S4 ≈ 1.8k). FIXES (harness-verified): (i) `pod.sh` v_stop patterns
generalized to any `tools/chain_*.sh` + `eval_*.py`; (ii) `chain_sport2.sh` completion GUARD — sentinel +
final object only when every arm has pretrain .done + full-test summaries + probe rows, else
`CHAIN-SPORT2-INCOMPLETE missing:…` exit 1 (supervisor relaunches → resume). The bogus object was moved
to `gs://qhrrn2-rescue/sport2/_bogus_final_1131Z.tgz`. Supervisor RESTARTED 11:50Z: hunts a fresh node →
resumes S0/S1/S2/S4/S5/S6 from GCS-live, S3/S7 fresh with `--remat`. New ETA in the next heartbeat
(add ≈ 25 min hunt/bring-up + the S3/S7 restart ≈ 1.5 h to the original plan → COMPLETE ≈ 21:30–22:00 IST).
If the supervisor ever logs COMPLETE while arms are incomplete: check `gsutil ls gs://qhrrn2-rescue/sport2/`
for a stray `sport2_final.tgz`, quarantine it (`gsutil mv`), and restart `supervise`.

**OPS INTERVENTION 11:30Z / 17:00 IST (Opus, PI: "fix things and make a decision"):** S3 and S7 (the
T12 arms WITHOUT `--remat`) OOM'd HBM at their first step (`HLO temporaries 33.02G > 31.24G`; the P11-EXT
lesson — S4/T24 already carried --remat and trains fine at 2.26 it/s). Fix = `--remat` added to S3/S7 in
`tools/chain_sport2.sh` (flags re-verified), chain stopped, patched chain shipped, relaunched: the six
running arms RESUME from their ckpts (≤5 min lost), S3/S7 retrain with remat. Campaign still 8/8 arms;
ETAs shift by the S3/S7 restart (T12+remat ≈ 1.3× T12 ≈ 1.5 h) — new ETA in the next heartbeat.

**LIVE (as of 10:35Z / 16:05 IST 2026-08-21):** node READY in us-east1-d (created 10:14Z, first
attempt), chain launched 10:27Z (remote pid 12042), supervisor pid 43606 (`runs/pod_supervisor.pid`),
deadline +14 h (`runs/tpu_deadline.txt`), all 8 arms training in parallel. ETA absent churn:
T6 arms ≈ 16:35 IST (11:05Z), T12 arms ≈ 17:00 IST, S4 T24 ≈ 18:00 IST (12:30Z) = PHASE1-OK,
PHASE2-OK ≈ 19:30 IST, COMPLETE + auto-teardown ≈ **20:00 IST (14:30Z)**. Heartbeat cron at :23
and the three monitors are armed in the launching session (they survive model switches, NOT app
restarts — re-arm per §2).

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

**Hourly heartbeat** — `CronCreate`, cron `23 * * * *`, prompt verbatim:

> HOURLY TPU HEARTBEAT (standing PI directive, generic — never run-specific). Report ALL times/ETAs in IST (UTC+5:30) with UTC in parentheses. Do ALL of: (1) `bash tools/pod.sh status` (node state across zones, supervisor alive?, deadline margin, GCS bank sport2, remote job + per-arm DONE/partial). (2) Per-arm progress: `gcloud compute tpus tpu-vm ssh qhrrn2-pod2 --zone=us-east1-d --project=quantum-llm --command "cd ~/qhrrn2 && for a in S0 S1 S2 S3 S4 S5 S6 S7; do printf '%s: ' \$a; grep -E 'step |PRETRAIN-|EVAL-|PROBE-' runs/wave_pre_\$a.log runs/detached.log 2>/dev/null | grep \$a | tail -1 | cut -c1-100; echo; done"` (bounded; use perl alarm 150) — compute per-ARM completion fraction and ETA from the measured it/s. This campaign (SPRINT S2, Sudoku-Extreme wave 1) has 8 chip-pinned pretrains IN PARALLEL (S0 S1 S2 S5 S6 at T6, S3/S7 at T12, S4 at T24 = the long pole), then PHASE2 evals per arm on its own chip, then PHASE3 probes, then RESCUE-OK + CHAIN-SPORT2-COMPLETE. (3) `tail -3 runs/tpu_status_log.txt` and `.venv/bin/python tools/spend_report.py --since 2026-08-21 | tail -3` for spend. (4) Report to the PI in 2–4 sentences + a table with EACH active process, completion fraction, and concrete ETA in IST — EVEN IF UNEVENTFUL. (5) If the supervisor (runs/pod_supervisor.pid) is NOT alive while a node exists or the campaign is incomplete: restart it with `nohup bash tools/pod.sh supervise 14 >/dev/null 2>&1 &` and say so. (6) If the pod.sh log shows COMPLETE + `down rc=0`: verify ZERO TPUs + ZERO queued-resources in all zones, then do EXACTLY §6 of tools/HANDOFF.md (sport2 version) and delete this cron job (CronDelete). NO analysis, NO accuracy values quoted, NO verdicts — tools/analyze_sport2.py is reserved for the analysis pass. (7) If the hard-delete deadline (runs/tpu_deadline.txt) is within 2h of expected completion, extend it and report. (8) If a bring-up step (UP/CANARY/LAUNCH) has sat unchanged for >25 min and a fresh describe shows the node PREEMPTED/STOPPED/absent and the stall watch has not acted, kill the supervisor's hung CHILD (never the supervisor) per tools/unstick_watch.sh and report. Read tools/HANDOFF.md if anything is unclear.

**Event monitor** — `Monitor` (persistent), command verbatim:
```
cd /Users/aakash/Projects/HRRN && tail -n0 -F runs/pod_qhrrn2-pod2.log | grep --line-buffered -E '^2026' | awk -F'\|' '/CREATED in|canary FAILED|DOWN |COMPLETE|CHAIN-SPORT2|PHASE|NEEDS|died|sick|SSHFAIL|unreachable|all zones dry|EXIT|relaunch|not READY|FAILED/{print;fflush();next} NF>=4{if($4!=last){print;fflush();last=$4}}'
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

**SPORT2 (current):** `=== SPORT2 START … arms=… ===` → `PHASE1: pretrain` → eight `=== PRETRAIN Sx hh:mm === chip c flags: …` lines at once (parallel) → per-arm `PRETRAIN-Sx-OK` (or `PRETRAIN-Sx-FAILED rc=… (see runs/wave_pre_Sx.log)` — the chain CONTINUES) → `wave done` → `PHASE1-OK` → `PHASE2: eval` → per-arm `EVAL-Sx-OK` → `PHASE2-OK` → `PHASE3: probes` → `PROBE-Sx-OK` → `PHASE3-OK` → `RESCUE-OK` → `CHAIN-SPORT2-COMPLETE`. After a preemption: `SKIP-Sx (GCS complete)` / `RESUME-Sx from live ckpt` / `RESTORE-Sx evals from GCS` = the durability stack working. pod.sh's PROGRESS field shows only the LAST matching line (with 8 parallel arms it is not a progress bar — use the per-arm ssh in §2 for fractions). The rung-era signatures below still describe the supervisor itself.

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

Two supervisors · editing `tools/campaign.env` / `tools/campaign_sport2.env` · touching
the running chain (`pod.sh stop/relaunch` only if the PI asks) · on-demand, queued
resources, a second pod · running `tools/analyze_sport2.py` (or any analyzer) for a verdict ·
reading or quoting accuracy/exact numbers from `summary_*.json` / `results.jsonl` (the verdict
pass does that) · patching `pretrain.py`, `chain_sport2.sh`, `eval_sudoku_extreme.py` or probes ·
deleting `runs/tpu_deadline.txt` · `gsutil -m` pulls · more than ONE commit (the ops close) ·
reporting ETAs in any timezone but IST.

## 6. Completion procedure — and where you STOP (SPORT2 version)

After the loop's `down rc=0` and exit 0:
```bash
cd /Users/aakash/Projects/HRRN
for z in us-east1-d us-east1-c us-east5-b us-central1-a us-central2-b us-west1-c us-west4-a asia-east1-c; do
  echo "$z: $(gcloud compute tpus tpu-vm list --zone=$z --project=quantum-llm --format='value(name,state)' 2>&1 | tr '\n' ' ')"
  gcloud compute tpus queued-resources list --zone=$z --project=quantum-llm --format='value(name,state)' 2>/dev/null
done                                                    # every line must be empty
mkdir -p runs/_sport2_final && gcloud storage cp gs://qhrrn2-rescue/sport2/sport2_final.tgz runs/_sport2_final/
tar tzf runs/_sport2_final/sport2_final.tgz | sed 's#^runs/##; s#/.*##' | sort | uniq -c   # expect pretrainsport2_S0..S7 (x8), sxeval_psport2S0..S7 (x8), sudprobe_psport2S0..S7 (x8), wave logs
tar xzf runs/_sport2_final/sport2_final.tgz -C .              # extracts into runs/
for a in S0 S1 S2 S3 S4 S5 S6 S7; do echo "$a: ckpt $([ -f runs/pretrainsport2_$a/ckpt_latest.pkl ] && echo ok || echo MISSING)  metrics $(wc -l < runs/pretrainsport2_$a/metrics.jsonl 2>/dev/null) rows  eval-files $(ls runs/sxeval_psport2$a/*/summary_all.json 2>/dev/null | wc -l)  probe-rows $(wc -l < runs/sudprobe_psport2$a/results.jsonl 2>/dev/null)"; done   # PRESENCE + COUNTS ONLY — do not cat summary files
.venv/bin/python tools/inspect_ckpt.py runs/pretrainsport2_S0 runs/pretrainsport2_S3 runs/pretrainsport2_S4   # print-only (d16, step 20000, T per arm)
.venv/bin/python tools/spend_report.py --since 2026-08-21 | tail -4
```
Expected per arm: 1 ckpt at step 20,000; eval-files 5 (strat_t6/t64/t256 + full_t6/t64); probe 512 rows.
Then ONE ledger line under §5 (newest first, right after the `## 5.` header): date,
"SPRINT S2 CHAIN COMPLETE — ops close", completion time (IST + UTC), preemptions/relaunches
(count `DOWN`/`CREATED`/`relaunching` lines since `SPORT2 START` in the log), spend, "zero TPUs
verified", "data local in runs/ + GCS sport2; analysis reserved for the analysis pass
(tools/analyze_sport2.py)". Commit (`git add Documentation/Design_Ledger.md` only; `runs/` stays
untracked) and push. Delete the cron. **STOP.** Tell the PI the campaign is closed operationally and
the analysis pass can start. If arms are MISSING (a `PRETRAIN-Sx-FAILED` in the log): report which,
with the last 20 lines of `runs/wave_pre_Sx.log` — do NOT relaunch or patch; the PI decides.

## 7. Facts you will be asked about

- SPORT2: registration = ledger 'SPRINT S2 LAUNCH REGISTRATION' (2026-08-21; rules pre-data in
  tools/analyze_sport2.py, self-test 12/12); protocol = sapientinc/sudoku-extreme, seeded 1k train ×100
  group-aug, FULL 423k test exact accuracy; arms S0 base / S1 +RI / S2 +NI / S3 T12 / S4 T24 / S5 plain /
  S6 +digit-aug / S7 RI+NI+T12; harness-verified offline (4 runs) + CPU smoke end-to-end; the chain
  pulls the benchmark npz from GCS itself. Rung-1c (tag r1c) is COMPLETE and ledgered (7b79641); its
  block below is historical.

- Registration: 2026-08-19 RUNG-1b LAUNCH REGISTRATION (rules R1b-1..R1b-4, locked pre-data); analyzer `tools/analyze_r1b.py` self-test 15/15.
- Chain changes (arms A6–A9, `R0_RT_ALL`/`R0_RT_ONLY`) were harness-verified offline: exact per-arm flags, SKIPs, 33-job queue, 5 waves, sentinel rc=0.
- Supplied ckpts in `rr1b/`: A4s1/A4s2/A3s1 (rung 1, d64) and A9s1/A9s2 (= rung-0 d48 A4s1/A4s2, md5-verified).
- Rung-1 verdict (for context only): width ceiling confirmed at d48; H-38/H-39 are what this campaign tests.
