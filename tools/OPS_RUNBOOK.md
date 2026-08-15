# TPU OPS RUNBOOK — rung-0 campaign (written 2026-08-15 for model/session handoff)

The science lives in `Documentation/Design_Ledger.md`. This file is ONLY the
operational state machine, so that any session — after a restart, a model
switch, or a cold start — can supervise the fleet mechanically without
re-deriving anything from conversation memory. **The repo outranks memory.**

## 1. What is running and what it produces

Two v6e-8 spot pods in us-east1-d run `tools/chain_r0.sh` (12 pretrains,
d48/T6/40k, arms A1 floors / A2 global / A3 plain / A4 floors+NI, seeds 0-2,
+ 50 battery jobs). Split by `R0_ARMS`:

| pod | R0_ARMS | done when |
|---|---|---|
| qhrrn2-pod  | `A1s0 A1s1 A1s2 A2s0 A2s1 A2s2` | `CHAIN-R0-COMPLETE` in its detached.log |
| qhrrn2-pod2 | `A3s0 A3s1 A3s2 A4s0 A4s1 A4s2` | same |

Every result stages to `gs://qhrrn2-rescue/r0/` (ckpts live every 5 min +
per-arm; batteries per wave as `partial_results.tgz`; final `r0_final.tgz`).
**A preemption costs <=5 min of compute, never the campaign** — the chain
resumes from GCS on relaunch (`RESUME-<arm>-FROM-GCS`, `RESUMED ... at step N`).

## 2. The four supervision layers (who survives what)

| layer | survives | job |
|---|---|---|
| **launchd watchdog** `com.qhrrn2.tpuwatchdog` (15-min) | EVERYTHING (session death, restarts, sleep) | inventory log `runs/tpu_status_log.txt`; **HARD-DELETES all nodes past `runs/tpu_deadline.txt`** (UTC epoch). Zones: east1-d/c, east5-b, central1-a, central2-b, west1-c, west4-a, asia-east1-c |
| **hunters** `tools/r0_retry_loop.sh <pod> "<arms>" <hours>` | this Mac's process table (NOT app restarts) | create -> up -> canary -> launch -> verify -> SUPERVISE; on node not-READY: teardown + resume hunt; exit only on CHAIN-R0-COMPLETE or hours elapsed. Log: `runs/r0_retry_<pod>.log` |
| **completion watchers** `tools/completion_watch.sh <pod> <zone> CHAIN-R0-COMPLETE <hours>` | same as hunters | on sentinel: rescue + `dispatcher down`. Tolerates transient absence. Log: `runs/complwatch_<pod>.log` |
| **session monitors / cron heartbeat** | this session only | convenience; re-arm after any restart |

Rule: **exactly one hunter and one watcher per pod.** Overlapping instances
write one log and become unreadable (bitten 08-15). Check with
`ps -eo pid,args | grep -E "r0_retry_loop|completion_watch" | grep -v grep`.

## 3. The 5-command health check (run this first, always)

```bash
gcloud compute tpus tpu-vm list --zone=us-east1-d --format="table(name,state,health)"
ps -eo pid,args | grep -E "r0_retry_loop|completion_watch" | grep -v grep
tail -2 runs/tpu_status_log.txt
python3 -c "import time;d=int(open('runs/tpu_deadline.txt').read());print(f'deadline in {(d-time.time())/3600:.1f}h')"
.venv/bin/python tools/spend_report.py --since 2026-08-14 | grep -E "qhrrn2|TOTAL"
```
Then per live pod (chain progress):
```bash
gcloud compute tpus tpu-vm ssh <POD> --zone=us-east1-d --project=quantum-llm --command="cd ~/qhrrn2 && (kill -0 \$(cat runs/detached.pid) 2>/dev/null && echo RUNNING || echo IDLE); echo arms=\$(ls runs/pretrain13f_*/.done 2>/dev/null|wc -l)/6; grep -E 'PHASE1-OK|wave done|CHAIN-R0' runs/detached.log | tail -2"
```

## 4. Decision table

| you see | it means | do |
|---|---|---|
| node PREEMPTED, hunter alive | normal churn | nothing — hunter clears it and re-hunts |
| node PREEMPTED, **no hunter** | supervision gap | `dispatcher down --name <pod> --zone us-east1-d`, then relaunch ONE hunter (cmd in §5) |
| node READY, chain IDLE, no CHAIN-R0-COMPLETE | chain died (ceiling / crash) | relaunch chain: `dispatcher run --name <pod> --zone us-east1-d --detach --wall-time 30600 --cmd "R0_ARMS='<arms>' bash tools/chain_r0.sh $VH $RG $RB $RT"` (VH/RG/RB/RT: build from `runs/lad_p1248c40k` task list + `data/re_gate48`, `data/re_gateb48`, `data/re_train48` stems). It resumes. |
| CHAIN-R0-COMPLETE seen, node still up | watcher missed it | pull `gs://qhrrn2-rescue/r0/r0_final.tgz`, then `dispatcher down` |
| stale hunter polling a dead node | pre-fix hunter | `pkill -f r0_retry_loop.sh`, relaunch per §5 |
| deadline < remaining work | would kill good work | extend: write new epoch to `runs/tpu_deadline.txt` |
| anything confusing / out of hand | PI standing order | **delete all VMs and wait**: `for p in qhrrn2-pod qhrrn2-pod2; do .venv/bin/python tools/dispatcher.py down --name $p --zone us-east1-d; done` — work is banked, nothing is lost |

## 5. Relaunch commands (exact)

```bash
# hunter (one per pod; 8h budget)
nohup bash tools/r0_retry_loop.sh qhrrn2-pod  "A1s0 A1s1 A1s2 A2s0 A2s1 A2s2" 8 >/dev/null 2>&1 &
nohup bash tools/r0_retry_loop.sh qhrrn2-pod2 "A3s0 A3s1 A3s2 A4s0 A4s1 A4s2" 8 >/dev/null 2>&1 &
# watcher (one per pod)
nohup bash tools/completion_watch.sh qhrrn2-pod  us-east1-d CHAIN-R0-COMPLETE 12 >/dev/null 2>&1 &
nohup bash tools/completion_watch.sh qhrrn2-pod2 us-east1-d CHAIN-R0-COMPLETE 12 >/dev/null 2>&1 &
```
The hunter carries the canary ckpt (`runs/pretrain6_d24/ckpt_latest.pkl`,
mkdir-before-scp) and passes `dispatcher canary` before any launch. `up`
is idempotent on an existing node.

## 6. When both chains complete

1. Pull `gs://qhrrn2-rescue/r0/r0_final.tgz` (both pods stage the same
   object name — pull each pod's before teardown, or read the per-arm
   dirs from the tarballs). Extract into `runs/`.
2. Verify zero TPUs + zero QRs across all door zones.
3. `.venv/bin/python tools/analyze_r0.py` — R1-R6 as registered
   (2026-08-14 rung-0 launch entry). Ledger the verdict. Do NOT tune.

## 7. Spend

`tools/spend_report.py` derives billed READY-hours from the watchdog log
(+-15 min). Rung-0 registered estimate $130; expected all-in with the
overnight churn ~$140-160. Program-to-date at 08-15 morning ~$335.
