# HANDOFF — TPU ops phase (persistent; the "CURRENT CAMPAIGN" block is rewritten per campaign)

**Read this first, then `tools/OPS_RUNBOOK.md`.** The repo outranks conversation
memory. The ops model runs the campaign to completion and STOPS; the analysis
phase (Fable) is reserved for the PI's next switch. **ALL ETAs and clock times
reported to the PI are in IST (UTC+5:30), with UTC in parentheses** (PI, 2026-08-21).
HARD BOUNDARY during ops: no `tools/analyze_sportB.py` (nor analyze_sport3a.py / analyze_sport2w2.py / analyze_sport2.py), no reading/quoting accuracy
values from result files, no verdicts, no tuning, no chain/env edits, no second pod.

## CURRENT CAMPAIGN — PHASE B RUNG 1 "the parity ladder, d64" (launched 2026-08-24 evening IST; ledger 'PHASE B RUNG 1 LAUNCH REGISTRATION')

**What it is:** d64 FULL-WIDTH (`--width-scale 4`, n_bulk 959,482) versions of the measured d16/d32
reference recipes — WIDTH is the only variable. ONE spot pod, v6e-16 first / v6e-8 fallback (same
ladder/strikes as waves 2/3a). Tag `sportB`, GCS `gs://qhrrn2-rescue/sportB/`, chain
`tools/chain_sportB.sh` (per worker, `CHAIN_WORKER/CHAIN_WORKERS`, chips self-detected; each arm
pretrains DP over its worker's WHOLE host — no chip pinning during pretrain).
**Arms:** W0 = B1 (plain T12 RI.5 NI.01 @50k) · W1 = B2 (plain T12 FPA k4 @50k) · W2 = B3 (priced
T12 @50k) · W3 = B4 + B4s1 + B5 sequential (plain T6 @20k ×2 seeds; priced T6 @20k). Monitor every
5k, banked ckpts every 5k, live-sync 5 min. After each arm: `VALBEST-Bx`, cheap evals
(`EVALCHEAP-Bx-OK`). **PHASE2 = a GLOBAL claim queue** (GCS `claim_*` markers; any worker runs any
ready task): breadth SCREENS per arm (strat-512 k=256 on the val-best AND mid banked ckpts —
`SCREEN-Bx-vb-OK step=NNNNN` / `SCREEN-Bx-mid-OK`), full-test evals (B1–B4 t6+t64+valbest;
B4s1/B5 t64 only — `FULL-Bx-t64-OK`), `PROBES4-OK` (B1–B4 parallel), then `PHASE4: screen winner
Bx` → 20k k=128 (`PHASE4-OK`; `PHASE4-MID` too iff the winner's screens differ ≥ 5 pp).
Completion: `CHAIN-SPORTB-WORKER-DONE` (waits) / `CHAIN-SPORTB-COMPLETE` + `SELF-TEARDOWN`.
ETA ≈ 9–12 h on a v6e-16 (INFERRED — canary + first arm measure it; wall pace expected ≥ 2 it/s
T12 d64 DP-4; the 8.5 h wall ceiling mid-pretrain recycles + resumes = NORMAL).

**Signature notes:** `PRETRAIN-Bx-REMAT-RETRY` = HBM OOM auto-retry with --remat (numerics-equivalent,
tested) — normal, report it. `PRETRAIN-Bx-DIVERGED last_loss=…` = report to the PI, do NOT patch
(the registered contingency — ONE relaunch at lr 5e-4 — is the PI's call). `SHARD-WAIT` = chip
transiently busy, retrying, normal. Claim lines are silent (GCS markers); a task another worker
claimed shows nothing locally. `PHASE2-DONE` per worker precedes the guard.

**Heartbeat progress (per worker w):** `gcloud compute tpus tpu-vm ssh qhrrn2-pod2 --zone=<zone>
--worker=$w --project=quantum-llm --command "cd ~/qhrrn2 && for a in B1 B2 B3 B4 B4s1 B5; do [ -f
runs/wave_pre_$a.log ] || continue; printf '%s: ' $a; grep -E 'step |DONE' runs/wave_pre_$a.log |
tail -1 | cut -c1-70; echo; grep MONITOR runs/wave_pre_$a.log | tail -1 | cut -c1-120; done; grep -E
'PRETRAIN-.*-OK|VALBEST|EVALCHEAP|SCREEN-.*-OK|FULL-.*-OK|PROBES4|PHASE4|PHASE2-DONE|QUEUES-DONE|WORKER-DONE|COMPLETE|FAILED|DIVERGED|REMAT'
runs/detached.log | tail -12"` (bounded, perl alarm 150). Steps: B1/B2/B3 50k, B4/B4s1/B5 20k.
MONITOR lines (val@t64/ret_final/lam) are the trajectory — report, do NOT interpret.

**§6 completion (sportB version):** pull `gs://qhrrn2-rescue/sportB/sportB_final.tgz`; expect 6
`pretrainsportB_*` dirs (ckpt_latest + banked ckpt_0*.pkl [10 for 50k arms, 4 for 20k arms] +
metrics.jsonl + val_best.txt), 6 `sxeval_psportB*` dirs (strat_t6/t64/t256 + val_t64 + ret_t8 +
retfm_t8 everywhere; full_t64 everywhere; full_t6 + full_t64_valbest on B1–B4 where vb differs), 12
`sxscreen_psportB*_{vb,mid}` dirs (some may be skip-empty if vb=final had no distinct mid), 4
`sudprobe_psportB*` (B1–B4), 1–2 `sxbreadth20k_psportB*`; counts only; ONE ops ledger line; ONE
commit; STOP. Analysis = `tools/analyze_sportB.py` untouched (self-test 16/16), reserved.

## ✅ COMPLETE — SPRINT S2 WAVE 3a (2026-08-24 08:48Z / 14:18 IST) — ops closed, fleet ZERO, node self-torn-down, zero preemptions, all 16 arms + scans banked (sport3a_final.tgz 356 MB, extracted into runs/); VERDICT LEDGERED same day (analyze_sport3a.py 18/18; report Report_2026-08-24_SprintS2_Wave3a_Verdict.md). Mid-run supervisor counter fix 89f154b validated live. The blocks below are the historical launch/ops record.

## ▶ LIVE — OPS HANDOFF (Opus) as of 2026-08-23 23:20 IST (17:50Z) — v6e-16 READY us-east1-d (since 16:57 IST), 4 workers × 4 chips, 16/16 chips busy, no preemption since launch

**State:** chain launched 17:09–17:10 IST; relaunched per worker after each worker's breadth scan to load the +8 addendum arms (w0 13:37Z, w2 13:42Z, w3 13:48Z, w1 15:01Z — these timestamps set the 8.5 h WALL-CEILING relaunches: **w0 ≈ 03:37 IST, w2 ≈ 03:42, w3 ≈ 03:48, w1 ≈ 05:01 IST** → `worker(s) N IDLE — relaunching` + resume from live ckpts = NORMAL, not incidents). All three S5 breadth scans BANKED (`breadth20000_S5_k128/k256`, `breadth_S5_t64_k1024`). Per-worker arms (one per chip): W0 A2 A3 A2s1 A3s1 · W1 A4 A5 A4s1 A5s1 · W2 A6 A7 A9 A9s1 · W3 A7s1 A8(DONE 16:25Z, cheap evals OK) A10 A6s1. Progress at 22:19 IST: T12 arms 8.5k–25k/50k (≈1.2 it/s wall), A6 d32 32k, A6s1 21k; the pole is worker 1 (arms started 20:31 IST). **ETA COMPLETE ≈ 12:30–13:30 IST Aug 24**; deadline `runs/tpu_deadline.txt` = 16:43 IST Aug 24 (heartbeat rule 7 extends it if completion slips within 2 h). Idle-chip FILLER (`tools/idle_filler.sh`, detached on w0/w2/w3; `runs/filler.log`; claims `breadth_*.claim` in GCS) fills free chips with breadth k=256 scans of W3 (done 17:36Z, banked), W2 (running on w3 chip 1), W8, W9 — `FILLER-*-OK` lines; the chain's sharded phases retry on a busy chip (`SHARD-WAIT … retrying` = normal). Supervisor pid in `runs/pod_supervisor.pid` (+ caffeinate pinned), meter, heartbeat cron :23 (id a553275f in THIS app session; re-arm per §2 if the app restarts, using the wave-3a prompt in the CURRENT CAMPAIGN block + the chip-census/utilization-policy paragraph), edge + unstick + watchdog-inventory monitors. Spend today ≈ $170 (v6e-16 ≈ $13.6/h).
**OPS FIX 2026-08-24 03:47 IST (Opus, mid-run):** the supervisor's relaunch crash-loop counter (`RELAUNCHES`) did not reset on healthy running polls, so the 4 staggered 8.5h wall-ceiling resumes climbed it 1 per recycle toward the exit-3 'needs eyes' teardown of a HEALTHY run. Fixed in `pod.sh` (commit 89f154b): reset the counter on any poll with no worker needing relaunch and >=1 running; cap 6->10. Supervisor RESTARTED (new pid in the pidfile) with the fix + zeroed counter; it re-adopted the live node and now logs wall-ceiling relaunches as `attempt 1/10` (a fast crash loop keeps a worker IDLE every poll and is still caught). No data impact.

**Decision table additions:** `RUNNING 3/4` on one poll = usually an SSH miss under load, check the worker directly before acting; `PRETRAIN-Ax-FAILED rc=134` with a libtpu `SLICE_FAILURE` = transient, the relaunch resumes it (W4 precedent) — report, do not patch; utilization policy: report the chip census each heartbeat, RECOMMEND (never execute) v6e-8/v6e-1 migrations per the rule; filler jobs never run on worker 1. `WORKER-DONE` = that worker finished its share (waits); `COMPLETE` = global, then SELF-TEARDOWN + supervisor down ("already gone" OK).
**§6 completion (sport3a version, see the CURRENT CAMPAIGN block for counts):** 16 pretrain dirs (ckpt_latest + ckpt_0*.pkl ×10 + metrics.jsonl + val_best.txt), 16 eval dirs (strat t6/t64/t256, val_t64, ret_t8, retfm_t8, full_t6, full_t64 [+ full_t64_valbest]), 14 probes (no A4/A4s1), scans: 3 S5 + the filler W2/W3/W8/W9 k=256 (whatever finished) + 1–4 `sxbreadth20k_psport3a*`; presence+counts only; ONE ops ledger line; ONE commit (working tree is clean at handoff); STOP. **The analysis pass (Fable, next session): `tools/analyze_sport3a.py` untouched (self-test 18/18) → BREADTH-CONFIRM / COLLAPSE CONTROL / FIX arms / PRICE CEILING / REGULARISER-OF-CHOICE / TRAJECTORY LAW / RECIPE DECISION → Phase B build (full-width scaling flag, DP ladder chain d64→d96→d128) + launch.**

## CURRENT CAMPAIGN — SPRINT S2 WAVE 3a "fix, confirm, instrument" (launched 2026-08-23 evening IST; ledger 'SPRINT S2 WAVE 3a LAUNCH REGISTRATION')

**What it is:** Phase A of the wave-3 plan (the wave-2 verdict: H-45 contractivity collapse of plain maps with
budget/capacity; H-44 price = contractive regulariser; S5 breadth 70.7 % @k128 strat-512). ONE spot pod, v6e-16
first / v6e-8 fallback (same ladder, strikes, demotion as wave 2). Tag `sport3a`, GCS `gs://qhrrn2-rescue/sport3a/`,
chain `tools/chain_sport3a.sh` (same launch contract as wave 2: per worker, `CHAIN_WORKER/CHAIN_WORKERS`, chips
self-detected). **Jobs:** PHASE0 (worker 0, sharded over its chips) = BREADTH CONFIRMATION on the banked wave-1 S5
map: `bscan:S5:20000:128`, `bscan:S5:20000:256`, `bscan:S5:strat:1024` (≈ 2.75 h on 8 chips, ≈ 5.5 h on 4);
then 16 training arms (v6e-16: one per chip, `ARMS_W0..W3` = "A2 A3 A2s1 A3s1" "A4 A5 A4s1 A5s1" "A6 A7 A9 A9s1" "A7s1 A8 A10 A6s1"; the +8 addendum of 18:45 IST; on a v6e-8 fallback two per chip):
A2 plain T12 RI.5 NI.01 · A3 plain T12 FPA k4 · A4 plain T12 eq_coupled · A5 priced T12 · A6 priced d32 T6 ·
A7 plain T12 (W2 → 50k) · A7s1 seed 1 · A8 plain T6 aug500 wd1e-3 · A9 plain T12 RI/NI+FPA · A10 priced T12 RI/NI · seeds A2s1 A3s1 A4s1 A5s1 A6s1 A9s1 — all 50k steps, MONITOR every 5k (val@t64,
retention sched/final-map, η, λ_max), ckpts banked every 5k (`ckpt_0*.pkl`, live-synced with metrics). Per arm:
pretrain → `VALBEST-Ax <step>` → cheap evals (strat t6/64/256 k16, val t64, ret_t8, retfm_t8) → probe (skipped
for A4: eq_coupled). Then PHASE-F = full-test evals SHARDED per arm (t6 + t64 final; t64 val-best), PHASE4 =
best-arm 20k k=128, completion guard → `sport3a_final.tgz` + `CHAIN-SPORT3A-COMPLETE` (+ self-teardown).
ETA ≈ 14 h on a v6e-8 (T12 @50k ≈ 9 h is the pole), ≈ 16 h on a v6e-16 (worker 0 carries PHASE0).

**Chip-utilization policy (PI 2026-08-23):** report busy/idle chips per worker each heartbeat; ≤8 busy for >30 min (no sharded phase imminent) → migrate to v6e-8 (stop → `ACCEL_LIST=v6e-8` in campaign.env → restart supervisor → resume); one long arm alone → v6e-1. Wave 3a: `tools/idle_filler.sh` runs detached on workers 0/2/3 (`runs/filler.pid`, `runs/filler.log`) filling free chips with eval-only breadth k=256 scans of W3/W2/W8/W9 (claims `breadth_*.claim` in GCS; results `breadth_<arm>_t64_k256.tgz`); the chain's sharded phases retry on a busy chip (`SHARD-WAIT … retrying` = normal). Don't start a second filler; if a filler job looks stuck >3 h, kill just that python process.

**Signatures:** `=== SPORT3A START … chips=N`, `=== BSCAN bscan:S5:20000:128 …`, `BSCAN-…-OK`, `PHASE0-DONE`,
`chip c queue: Ax`, `=== PRETRAIN Ax …`, `MONITOR step N: val@t64 … lam_max …` (in `runs/wave_pre_Ax.log`),
`VALBEST-Ax`, `PRETRAIN-Ax-OK`, `EVALCHEAP-Ax-OK`, `PROBE-Ax-OK` / `PROBE-SKIP-A4`, `QUEUES-DONE`, `FULL-Ax-t64-OK`,
`FULLVB-Ax-t64-OK step=…`, `EVAL-Ax-OK`, `PHASE4: best arm …`, `PHASE4-OK`, `RESCUE-OK`, `CHAIN-SPORT3A-WORKER-DONE` /
`CHAIN-SPORT3A-COMPLETE`, `SELF-TEARDOWN…`. `PRETRAIN-Ax-FAILED rc=` = report the last 20 log lines, do NOT patch
(the wave-2 W4 crash was a transient libtpu SLICE_FAILURE; the relaunch resumes). Supervisor lines as in wave 2.

**Heartbeat progress (per worker w):** `gcloud compute tpus tpu-vm ssh qhrrn2-pod2 --zone=<zone> --worker=$w
--project=quantum-llm --command "cd ~/qhrrn2 && for a in A2 A3 A4 A5 A6 A7 A7s1 A8 A2s1 A3s1 A4s1 A5s1 A6s1 A9 A9s1 A10; do [ -f runs/wave_pre_$a.log ]
|| continue; printf '%s: ' $a; grep -E 'step |DONE' runs/wave_pre_$a.log | tail -1 | cut -c1-70; echo; grep MONITOR
runs/wave_pre_$a.log | tail -1 | cut -c1-120; done; grep -E 'BSCAN-.*-OK|PHASE0-DONE|PRETRAIN-.*-OK|EVALCHEAP|FULL|PHASE4|
QUEUES-DONE|WORKER-DONE|COMPLETE|FAILED' runs/detached.log | tail -12"` (bounded). All arms 50k steps; T12 wall ≈ 1.5–1.7
it/s, T6 d16 ≈ 4 it/s, d32 ≈ 2.2 it/s; MONITOR lines are the trajectory (val@t64 / ret_final / lam_max) — report them
but do NOT interpret (no verdicts; `tools/analyze_sport3a.py` is reserved).

**§6 completion (sport3a version):** pull `gs://qhrrn2-rescue/sport3a/sport3a_final.tgz`; expect 16 `pretrainsport3a_*`
dirs each with `ckpt_latest.pkl` + `ckpt_0*.pkl` (10) + `metrics.jsonl` + `val_best.txt`; 16 `sxeval_psport3a*` dirs
(strat_t6/t64/t256, val_t64, ret_t8, retfm_t8, full_t6, full_t64, + full_t64_valbest where it differs); 14 probes (no A4/A4s1);
`sxbreadth20000_S5_k128`, `sxbreadth20000_S5_k256`, `sxbreadth_S5_t64_k1024`, one `sxbreadth20k_psport3a*`; counts only;
ONE ops ledger line; ONE commit; STOP.

## ✅ COMPLETE — SPRINT S2 WAVE 2 (2026-08-23 09:52Z / 15:22 IST) — ops closed, fleet ZERO, analysis pending

CHAIN-SPORT2W2-COMPLETE on qhrrn2-pod2/us-east1-d; node self-torn-down + supervisor down (both paths, clean); all
8 zones ZERO (nodes + QRs). `gs://qhrrn2-rescue/sport2w2/sport2w2_final.tgz` (68MB): 13 ckpts (12 arms + W5gen),
12 eval arms × 7 summaries (84 summary_all.json), 11 probes (bar W5gen + box4-W7), 4 scans, 1 breadth20k (PHASE4
best-arm W2). Ops fixes + shard helpers committed. Spend ≈ $130 (wave-2). ALL LAYERS STOPPED (supervisor exited 0,
meter/caffeinate/monitors/heartbeat-cron down). **The analysis pass has NOT run** — `tools/analyze_sport2w2.py`
(untouched, self-test 21/21) is ready: M0-W2 / bands / breadth / P1-P7 / H-44. Overnight incidents (all handled,
in the ledger close): 4 preemptions + v6e-16 demotion; W4 transient libtpu SLICE_FAILURE crash → resumed; W4/W9
evals sharded over idle chips + merged (validity verified). The blocks below are the (now historical) launch spec.

## PREVIOUS CAMPAIGN (historical) — SPRINT S2 WAVE 2 (Sudoku-Extreme; launched 2026-08-22; COMPLETE 2026-08-23; ledger 'SPRINT S2 WAVE 2 LAUNCH REGISTRATION')

## ▶ LIVE — OPS HANDOFF (Opus) as of 2026-08-23 00:15 IST (18:45Z) — node v6e-8 READY us-east1-d since 17:41Z, chain launched 17:49Z, 1 worker × 8 chips, no preemption since

**State:** v6e-16 DEMOTED for this supervisor's life (strikes 2/2: preempted at 13:57Z while running, and at ~17:26Z during bring-up) → the v6e-8 fallback runs ALL FOUR job lists on one host, two jobs per chip: chip0 W13→W9 · chip1 W2→W5 · chip2 W3→W4 · chip3 W1→W8 · chip4 [scan S5@t64 ✓]→W6 · chip5 [scan S4 ✓]→W7 · chip6 [scan S7 ✓]→W1s1 · chip7 [scan S5@t6 ✓]→W4s1. **All 4 breadth scans COMPLETE + banked** (`breadth_*.tgz`). Training at 18:45Z (wall pace ≈ 4.1 it/s on T6 d16, ≈ 1.7 it/s T12, ≈ 2.2 it/s d32; the printed it/s overstates ~2.5×): W13 15.8k/100k · W1 15.8k/50k · W2 5.75k/30k · W3 5.8k/30k · W6 13.85k/20k · W7 7.3k/20k · W1s1 8.45k/50k · W4s1 5.9k/20k; W9 W5 W4 W8 queued behind W13 W2 W3 W1. **ETA absent churn:** long pole = chip 0 (W13 done ≈ 06:00 IST → evals → W9 → evals ≈ 12:30 IST) → PHASE4 (best-of-all-arms 20k-subsample k=128, 8-way, ~1 h) → COMPLETE + teardown ≈ **13:30–14:30 IST Aug 23**; d32 arms (W4/W8 evals ~3 h each) finish ≈ 10–11 IST. Deadline `runs/tpu_deadline.txt` = 17:06Z (22:36 IST) Aug 23 — extend if completion slips past ~20:00 IST. The remote wall ceiling (8.5 h) kills the chain ≈ 02:19Z (07:49 IST) → the supervisor relaunches (resume, SKIPs done work, re-arms the DMS) — `worker(s) 0 IDLE — relaunching` is NORMAL, not an incident. Spend since 08-22: $57.61 (incl. wave-1's morning; the two v6e-16 landings billed ~1.3 h).
**12:55 IST Aug 23 — ops acceleration in flight (PI: "prioritize W4's relaunch + evals, do the rest in parallel"):** W4 crashed once at step 19,550 with a libtpu `SLICE_FAILURE_WORKER_UNAVAILABLE` SIGABRT (transient runtime fault; resumable) → controlled `pod.sh stop` + supervisor relaunch at 07:21Z (W9 + W4 resumed from live ckpts). `tools/w4_shard_evals.sh` runs DETACHED on the pod beside the chain (pid `runs/shard_helper.pid`, log `runs/shard_helper.log`): for W4 then W9 it waits for the final ckpt, runs the full-test evals (t6, t64) with the SAME evaluator/flags sharded 5-way over idle chips 1 4 5 6 7 and `--merge`s them into the chain's eval dir (verified: sharded+merged == single run, every summary key and every per-puzzle record identical); the chain's eval_one then SKIPs those files. If the chain's own single-chip eval of that file had already started, the helper retires that ONE redundant process after the merged result is in place — `EVAL-W4-full64-FAILED` / `EVAL-W9-full64-FAILED` in detached.log is then COSMETIC (the merged `summary_all.json` + `records_all.npz` are present and ship in the eval tgz/final object; summaries carry `shard: "merged"`). Mention the sharded execution in the ops-close ledger line. Do not start a second helper. RESOLVED 14:04 IST: the helper's in-line merge lacked `JAX_PLATFORMS=cpu` and failed on `/dev/vfio busy` (shard COMPUTES were valid) — fixed in `tools/w4_shard_evals.sh`, and `tools/shard_merge_wait.sh` (merge-only, CPU) merged W4-t64 + W9-t6/t64 (each n=422,786 == sum of shards, asserted; tagged `merged`). All 13 arms' eval+probe OK; the `EVAL-W4-full64-FAILED`/`EVAL-W9-full64-FAILED` lines are the cosmetic retirements. ADD `tools/shard_merge_wait.sh` to the ops-close commit list. Wave-3 lesson: d32 full-test eval ≈ 2.75 h single-chip — shard it from the start.
**Layers live:** supervisor pid in `runs/pod_supervisor.pid` (+ caffeinate pinned to it), meter (`runs/sport2_meter.txt` / `.html`), heartbeat cron :23, edge + watchdog-inventory + unstick monitors (session-scoped — re-arm per §2 after any app restart, using the wave-2 prompt/patterns in this block). **Teardown:** supervisor on COMPLETE (primary) · the finalizing worker's SELF-TEARDOWN (backstop; `down` then reports "already gone — OK") · watchdog deadline (last resort).
**Decision-table additions for this campaign:** `WDONE`/`WORKER-DONE` = that worker finished its share (multi-host only; on the v6e-8 the single worker emits COMPLETE directly) · `STRIKE n/2` / `DEMOTE v6e-16` = the accelerator ladder working; to retry the 16 on a later hunt: `rm runs/pod_strikes.txt` and restart the supervisor (PI decision) · `PRETRAIN-Wx-FAILED` = report which arm + last 20 lines of `runs/wave_pre_Wx.log`, do NOT patch (the campaign then never COMPLETEs: the supervisor would relaunch up to 6 rounds, then exit 3 "needs eyes" after tearing down) · `SELF-TEARDOWN` lines = normal at the end.
**Uncommitted working tree (ops fixes + hardening, harness-verified): commit them WITH the ops-close ledger line** (`git add tools/chain_sport2w2.sh tools/pod.sh tools/dispatcher.py tools/HANDOFF.md tools/sport2_meter.py tools/w4_shard_evals.sh tools/shard_merge_wait.sh Documentation/Design_Ledger.md`).
**Analysis boundary unchanged:** no `tools/analyze_sport2w2.py`, no accuracy values quoted; §6 (sport2w2 version, below in the CURRENT CAMPAIGN block) then STOP — the analysis pass is Fable's on the next switch.

## ▶ RESUMED 2026-08-22 22:38 IST (17:08Z) — PI: "let's continue"; the deferred hardening is now APPLIED (chain self-teardown on COMPLETE gated by SELF_TEARDOWN/SELF_POD/SELF_ZONE from the pod.sh launch line; MAX_RELAUNCH=6; dispatcher `down` tolerant of an already-deleted node; live-sync sleeper cleanup) and harness-verified; supervisor + caffeinate + meter + heartbeat + 3 monitors re-armed; deadline +24h. The block below is the pause record.

## ⏸ PAUSED — 2026-08-22 19:45 IST (14:15Z) — PI: "pod is gone, let's pause the work for now"

Fleet ZERO verified (all 8 door zones, nodes + QRs). Supervisor KILLED (it had NOT re-created a node), meter +
caffeinate stopped, heartbeat cron deleted, all 3 monitors stopped. launchd watchdog left running (billing
backstop; deadline file 2026-08-23 13:58Z — refresh before any resume). The first v6e-16 (us-east1-d, 4
workers × 4 chips) was PREEMPTED at 13:57Z after ~45 min (STRIKE 1/2 recorded in runs/pod_strikes.txt —
delete that file to give the 16 a fresh 2-strike budget on resume).
**Banked in `gs://qhrrn2-rescue/sport2w2/`:** breadth scans S5@t6 and S5@t64 (k=256) COMPLETE; LIVE ckpts
(≤5 min old at preemption) for W13 W1 W1s1 W2 W3 W5gen W6 W7 W9 (resume from them); W4 W8 W4s1 (d32) and
the S4/S7 scans restart fresh. No arm complete yet; nothing to analyze.
**Uncommitted ops fixes in the working tree (commit at the next checkpoint):** chain chip auto-detect (v6e-16 =
4 hosts × 4 chips), 4-way ARMS_W0..W3 lists in both env files, pod.sh accel_workers v6e-16→4 + `stop`
archives worker logs, meter wave-2-aware. **Deferred hardening (not applied):** chain self-teardown on COMPLETE
(SELF_TEARDOWN/SELF_POD/SELF_ZONE via the pod.sh launch line + dispatcher `down` tolerant of an already-deleted
node) and MAX_RELAUNCH 3→6 for 4-worker wall-ceiling resumes — apply + harness before a long unattended run.

**RESUME (any session):**
```bash
cd /Users/aakash/Projects/HRRN
echo $(( $(date -u +%s) + 24*3600 )) > runs/tpu_deadline.txt      # 1. deadline FIRST (supervisor + watchdog follow it)
rm -f runs/pod_strikes.txt                                        # 2. optional: fresh strike budget for the v6e-16
bash tools/pod.sh status                                          # 3. expect node ABSENT, supervisor NOT RUNNING
nohup bash tools/pod.sh supervise 14 >/dev/null 2>&1 &            # 4. hunts v6e-16 then v6e-8; chain SKIPs done scans, RESUMEs partial arms
nohup caffeinate -i -s -w $(cat runs/pod_supervisor.pid) >/dev/null 2>&1 &   # 5. keep the Mac awake while it runs
nohup .venv/bin/python tools/sport2_meter.py --interval 300 >runs/sport2_meter.log 2>&1 &   # 6. meter (optional)
```
Then re-arm §2 (heartbeat cron :23 with the wave-2 per-worker progress loop described in the CURRENT CAMPAIGN
block; edge monitor with the `CHAIN-SPORT2W2|WORKER-DONE|STRIKE|DEMOTE|launch w` patterns; watchdog-inventory
monitor; `tools/unstick_watch.sh`). On `CHAIN-SPORT2W2-COMPLETE` → §6 (sport2w2 version) → STOP.


**What it is:** the wave-1 verdict's levers on the PLAIN (β=0) base + the PRICE × SCALE surface
(PI 2026-08-22: price effects may be scale-dependent — width/budget/dose axes, not one cell), on ONE spot
pod: **v6e-16 (4 workers × 4 chips — measured live 2026-08-22) hunted first, v6e-8 (1 × 8) as the fallback** — `ACCEL_LIST="v6e-16 v6e-8"` in
`tools/campaign.env` (= `campaign_sport2w2.env`); the 16 is DEMOTED for the supervisor's life after
`BIG_MAX_STRIKES=2` bring-up failures/preemptions (PI: "fall back if 16 gets preempted too much"). Tag
`sport2w2`, GCS `gs://qhrrn2-rescue/sport2w2/`, chain `tools/chain_sport2w2.sh` (launched ONCE PER
WORKER with `CHAIN_WORKER=w CHAIN_WORKERS=n`; worker w runs `ARMS_W$w`, the chip count per host is
self-detected from /dev/vfio; on a v6e-8 one worker runs all four lists, 2 jobs per chip). Jobs (16, one
per chip): W0 = W13 W2 W3 W1 | W1 = scan:S5:64:256 scan:S4:64:256 scan:S7:64:256 scan:S5:6:256 | W2 = W9 W5
W4 W8 | W3 = W6 W7 W1s1 W4s1 (arm meanings in the chain header). Each
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
patch). Supervisor lines: `CREATE … (v6e-16 spot)`, `CREATED in <zone> (v6e-16, 4 worker(s))`,
`launch w0/w1: detached + verified`, `READY <zone> | RUNNING 4/4 worker(s) | [w0 …] … [w3 …]`,
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
