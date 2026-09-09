# FULL-SET EVALUATIONS of the champion models (2026-09-10; eval-only; the PI: "get the best models we have for the numbers and evaluate them against the full eval set where we think the numbers would best reflect our work in the paper"; the ARC analysis in one pass once the pod's ARC runs land)

**Object.** The champion night's numbers on the paper's headline rows are on subsamples: D64 on 100k (binomial SE 0.04 pp), D128 on 20k (0.07), D256 on 5k (0.09). The field reports its D64 headlines on the full 422,786 (EqR 93.0, FPRM 94.2 pass@1), so the full-set D64 row is the one that makes our headline directly comparable at the convention level; D128 and D256 are depths the field does not report, where a larger subsample buys the precision and the full set buys nothing the paper needs. **The models:** the seed triple C0 C1 C2 (the claim-bearing seeded headline) and C5 (w192, the labeled best; the accuracy-per-MAC point); C4 (set attention, one seed, one variable) stays on its 100k row unless the wall allows it.

**Paces (measured from the chain's walls on the Mumbai v6e-16, four chips per worker; the cold evaluator runs at the US pace):** 109 puzzle-steps per second per chip at w384 (mean coupling), 82 with the attention coupling, 239 at w192. A full-test D64 row = 4.23× the chain's 100k wall.

| row | set | C0 / C1 / C2 (w384) | C5 (w192) | C4 (attention) |
|---|---|---|---|---|
| D64 FULL | 422,786 | 17.2 h each on 4 chips (69 chip-h) | 7.9 h (32 chip-h) | 22.8 h (91 chip-h) |
| D128 on 50k | 50,000 (seed 20260822) | 4.0 h | 1.9 h | 5.4 h |
| D256 on 50k | 50,000 | 8.0 h | 3.7 h | 10.8 h |
| D128 / D256 FULL | 422,786 | 34 / 68 h | 16 / 32 h | 46 / 91 h |

**The pod bills by wall, not by busy chips** (Mumbai v6e-16 ≈ $16/h), so the cost is the longest lane.

**Plan A (recommended): the seeded D64 row on the full set + the labeled best's rows.** Four lanes, one per worker: w2 C0 D64 full (already running under the old filler since ≈ 19:25Z Sep 9; resumes from its 300 s partials under the new one) · w0 C1 D64 full · w1 C2 D64 full · w3 C5 D64 full → C5 D128 on 50k → C5 D256 on 50k → the ARC suite (≈ 1 h; p13Dri / p13C53 × {val-hard, dev-30}, k = 32). Pole = 17.2 h from ≈ 20:00Z Sep 9 → ≈ 13:15Z Sep 10; w3 finishes its rows at ≈ 09:30Z and the ARC suite by ≈ 10:30Z; the pod comes down ≈ 13:30Z Sep 10. **Cap → 2026-09-10T16:00:00Z** (a 2.5 h margin; the knob, the node guard re-planted, the DMS keeper follows). **Cost from 19:40Z Sep 9 ≈ 18 h × $16 ≈ $290; the night ≈ $870; the program ≈ $1,000 of the $1,000 approved** — nothing left for the ARC-d96 rung without new credits. The k128 rows for C1 C2 C4 (in flight at 19:40Z, ≈ 3 h each) bank before the workers switch; the old filler's d64full C6 / C3 (next in its order) are abandoned unstarted.

**Plan B: A + C4's D64 full** on w3 after the ARC suite (22.8 h → the pole ≈ 09:30Z + 22.8 h ≈ 08:00Z Sep 11; cap → 10:00Z Sep 11; ≈ +$300 over A).

**Plan C: no full-set rows.** The ARC suite now on the first idle worker (≈ 1 h), the pod down by ≈ 21:30Z Sep 9 (≈ $30 more); the paper carries D64 on the 100k subsample (SE 0.04 pp), D128 on 20k, D256 on 5k, labeled; ≈ $290 stays for the ARC-d96 rung.

**Predictions (descriptive; each full-set / 50k number is expected inside its subsample row ± 2 SE):** C0 D64 full ∈ [98.30, 98.54], C1 ∈ [97.36, 97.56], C2 ∈ [98.55, 98.79], C5 D64 full ∈ [98.98, 99.22]; C5 D128 on 50k ∈ [99.30, 99.54], C5 D256 on 50k ∈ [99.42, 99.78]. Regressions 0 on every row. The triple's full-set D64 mean within 0.1 pp of 98.18. A row outside its band is reported as such (a subsample artifact would be the reading, never a correction).

**Mechanics.** `tools/filler_full.sh` (a retargeted copy of the champion filler: env-driven arm order and job list per worker; jobs d64full / d128sub / d256sub / arcsuite; the ARC gate a separate list, default none; idempotent GCS markers `filler/{d64full,d128sub,d256sub}_<arm>_OK`; 4-way sharded per worker; 300 s partials; PID-based liveness). Per worker: stop the old filler (`runs/filler_w<W>.pid`) and its evaluator shards AFTER its in-flight k128 job banks, then `W=<W> MY_ARMS=<its chain arms> FILLER_ARMS=<lane> JOBS=<lane jobs> nohup bash tools/filler_full.sh > runs/filler_full_w<W>.log 2>&1 &`; verify by artifact (the pidfile, the log's `FILLER-JOB-START`, the claim marker). The reader for the paper's table: `tools/analyze_champ_full.py` (descriptive; the new rows beside the registered ones, paired on the 100k ∩ full where both exist). Close: `bash tools/pod.sh down` after `arcsuite_OK` and the last `_OK`, fleet zero verified, the filler tarballs pulled with crc32c.

**Not launched. The PI chooses A / B / C and the cap; the launch is on the go.**
