# HANDOFF — TPU ops phase (persistent; the "CURRENT CAMPAIGN" block is rewritten per campaign)

**Read this first, then `tools/OPS_RUNBOOK.md`.** The repo outranks conversation
memory. The ops model runs the campaign to completion and STOPS; the analysis
phase is reserved for the PI's next model switch.

## CURRENT CAMPAIGN — RUNG 1b (launched 2026-08-19 09:18Z / 14:48 IST)

**What it is (ledger: 2026-08-19 RUNG-1b LAUNCH REGISTRATION):** the cells the
rung-1 verdict owes before rung 2 / d96 — H-38 β-rescale at d64 (A6 β=1e-5, A7
β=3e-6, A5-class, seeds 0/1), H-39 A5 seeds (A5s1, A5s2), A8 = A4-class d48@53,333
(η/budget), and rt-48 on every arm including rt-ONLY for ckpts supplied in GCS
(A4s1 A4s2 A3s1 from rung 1; A9s1 A9s2 = rung-0 d48 A4 aliases). Tag `r1b`,
GCS `gs://qhrrn2-rescue/rr1b/`, one spot v6e-8 pod `qhrrn2-pod2`.

**Expected shape:** 7 pretrains (≈38 min each at ~23 it/s: A6s0 A7s0 A5s1 A6s1
A7s1 A5s2 A8s0) → `PHASE1-OK` → `PHASE2: 33 battery jobs` → 5 × `wave done rc=0`
(≈25 min each; the 5th wave has one job) → `RESCUE-OK` → `CHAIN-R0-COMPLETE` →
the loop tears down. ≈6.3 pod-h ≈ $43 absent preemption. Deadline 01:18Z Aug 20
(16 h margin). The five rt-only arms print `SKIP-<arm> (completed in an earlier
life; final ckpt restored)` — that is correct, their ckpts were staged on purpose.

**Right now:** supervisor pid in `runs/pod_supervisor.pid`, hunting from 09:18Z
(first CREATE in us-east1-d). Monitor + hourly cron (:17) armed in this session.

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
