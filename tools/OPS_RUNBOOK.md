# TPU OPS RUNBOOK — one-pod campaign (rewritten 2026-08-18 night)

The science lives in `Documentation/Design_Ledger.md`. This file is ONLY the
operational state machine, so any session — after a restart, a model switch,
a cold start — supervises mechanically from the repo. **The repo outranks
conversation memory.** PI directive (2026-08-18): ONE stable spot v6e-8 pod
finishes the rung; no multi-pod, no queued resources, no on-demand.

## 1. The three files that matter

| file | role |
|---|---|
| `tools/campaign.env` | THE source of truth: pod name, zones, accelerator, `R_TAG/R_D/R_STEPS`, `ARMS` (order load-bearing), chain script, wall ceiling, GCS bucket, sentinel |
| `tools/pod.sh` | THE ops tool: `supervise` (one loop) · `status` · `relaunch` · `stop` · `down` · `log` |
| `runs/tpu_deadline.txt` | THE knob: UTC epoch past which the launchd watchdog DELETES every node — and past which the supervisor stops creating (it re-reads this file every poll) |

Task lists (VH/RG/RB/RT CSVs) live in `tools/r0_tasks.sh` (repo copy — never a scratchpad).

## 2. Layers — who survives what

| layer | survives | job |
|---|---|---|
| **`pod.sh supervise`** (Mac process, log `runs/pod_<POD>.log`, pid `runs/pod_supervisor.pid`) | this Mac's process table — NOT app restarts: restart it, it ADOPTS whatever is running | the state table below; exits 0 on completion (after teardown), 3 if the chain dies 3× on one node (after teardown, "needs eyes"), 2 at the deadline (no creates) |
| **launchd watchdog** `com.qhrrn2.tpuwatchdog` (`tools/tpu_watchdog.sh`, 15 min) | EVERYTHING | inventory log `runs/tpu_status_log.txt` + snapshot `runs/tpu_status.txt`; macOS notify on change; **hard-deletes all nodes past the deadline** |
| session heartbeat (hourly cron) + Monitor on the supervisor log | this session only | PI-facing status; re-arm after any restart |

The chain (`tools/chain_r0.sh`) is resume-complete from GCS: live ckpt every
5 min, complete ckpt per arm (`<ARM>_ckpt.pkl` = proof of completion → SKIP),
battery rows every 5 min + per wave (`partial_<firstarm>.tgz`, sanitized on
restore), final `r0_final.tgz` then the sentinel `CHAIN-R0-COMPLETE`. **Any
teardown costs ≤5 min of compute, never the campaign.**

## 3. State table (one poll = one row; never acts on an UNKNOWN read)

| supervisor sees | does |
|---|---|
| GCS `r0_final.tgz` present (checked first, node-independent) | down wherever the node is → exit 0 |
| node ABSENT in every zone (all read positively) | create(spot) zone by zone → `up --with-data` (retry ×1) → canary ckpt + `canary` → `launch_detached` → verify; any failure after create → down, next zone; all dry → sleep 8 min |
| CREATING / STOPPING / REPAIRING / DELETING | wait 2 min |
| PREEMPTED / STOPPED / TERMINATED (positively observed) | down (rescue + delete); next poll hunts |
| READY + sentinel in remote log | down → exit 0 |
| READY + job RUNNING | log one progress line |
| READY + job IDLE (crash / `timeout` ceiling / kill) | relaunch (resumes) — ≤3 per node life, then down + exit 3 loudly |
| READY + SSH unreachable ×6 (30 min) | down (sick node); next poll hunts |
| list/describe FAILED (network) | nothing; count; keep polling |
| past `runs/tpu_deadline.txt` | exit 2 without creating (watchdog deletes what is up) |

Verified offline end-to-end (stub gcloud/gsutil, real dispatcher +
launch_detached underneath) on every row above before it ran live.

## 4. Health check — 3 commands, always first

```bash
bash tools/pod.sh status            # node (all zones), supervisor, deadline, GCS bank, remote job + per-arm + rows
tail -3 runs/pod_qhrrn2-pod2.log    # what the loop last saw / did
launchctl list | grep qhrrn2        # watchdog loaded? (reinstall: bash tools/install_watchdog.sh)
```

## 5. After an app restart / cold start

1. `git log --oneline -3` — has a completion/verdict entry landed?
2. `bash tools/pod.sh status` — if `supervisor: NOT RUNNING` and the campaign
   is not complete: `nohup bash tools/pod.sh supervise 14 >/dev/null 2>&1 &`
   (it adopts a live chain; refuses to start twice).
3. Re-arm the session heartbeat + Monitor on `runs/pod_qhrrn2-pod2.log`.

## 6. Manual verbs (rare — the loop does these itself)

```bash
bash tools/pod.sh relaunch          # READY node, chain idle → relaunch (resumes)
bash tools/pod.sh stop              # controlled stop of the remote chain, verified (kill tree by pgid; pkill patterns cannot self-match)
bash tools/pod.sh down              # rescue + delete the node = the PI standing order ("delete all and wait")
echo $(( $(date -u +%s) + 12*3600 )) > runs/tpu_deadline.txt   # extend the deadline (supervisor + watchdog both follow it)
```
Never run two supervisors. Never edit `ARMS` order mid-campaign.

## 7. Completion

The loop tears the node down itself. Then (bucket/tag from `tools/campaign.env`):
`gcloud storage cp $GCS/r0_final.tgz .` (never `gsutil -m`) → extract into `runs/` →
`.venv/bin/python tools/inspect_ckpt.py runs/pretrain${R_TAG}_*`
(every arm admitted at artifact level: d/T/steps/flags) → verify zero TPUs +
zero queued-resources in all zones → the campaign's pre-registered analyzer
(`tools/analyze_<tag>.py`, written before the data, with `--selftest`; admits only
artifact-verified cells) — in the ANALYSIS phase, never the ops phase. Do NOT tune.
The per-campaign handoff (`tools/HANDOFF.md`) carries the exact commands.

## 8. Spend

`.venv/bin/python tools/spend_report.py --since <date>` derives billed READY
hours from the watchdog log (±15 min) at $6.82/pod-h spot v6e-8.

## 9. Retired (2026-08-18; in git history before this commit)

`r0_retry_loop.sh` (hunter), `completion_watch.sh` (watcher),
`r1_merge_into_pod2.sh` (merge; its STOP command `pkill -f`-matched its own
ssh shell and died before reporting — the last of the day's incidents). Their
jobs are the rows of §3 in one loop.
