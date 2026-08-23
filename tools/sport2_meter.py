# Ledger: SPRINT S2 progress METER (2026-08-21, PI request) — a read-only
# "download meter" for the running campaign: ONE bounded ssh per tick, zero model
# calls, zero writes on the pod. Pace is WALL-CLOCK measured between ticks (the
# trainer's printed it/s is per-step compute and overstates throughput ~3-4x on
# remat arms — the root of the evening's ETA slips), smoothed by an EMA.
"""
  .venv/bin/python tools/sport2_meter.py            # refresh every 300 s; Ctrl-C to stop
  .venv/bin/python tools/sport2_meter.py --once     # one table, exit
  .venv/bin/python tools/sport2_meter.py --interval 120
Also mirrors each table to runs/sport2_meter.txt.
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "tools" / "campaign.env"
LOG = ROOT / "runs" / "pod_qhrrn2-pod2.log"
OUT = ROOT / "runs" / "sport2_meter.txt"
HTML = ROOT / "runs" / "sport2_meter.html"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
PROJECT = "quantum-llm"
# phase-duration priors (min) for the COMPLETE estimate after PHASE1-OK — labeled "est"
PHASE2_MIN, PHASE3_MIN = 75, 30

def env():
    v = {}
    for line in ENV.read_text().splitlines():
        m = re.match(r'^([A-Z_]+)=(.*)$', line.strip())
        if m: v[m.group(1)] = m.group(2).split("#")[0].strip().strip('"')
    return v

def arm_steps(chain_path, arms, default_steps):
    """WAVE 2 (2026-08-22): per-arm step budgets parsed from the chain's arm_flags
    (single source of truth); seed replicates inherit their base arm; W5 = two stages."""
    steps = {}
    try: txt = Path(chain_path).read_text()
    except Exception: txt = ""
    for m in re.finditer(r'^\s*([A-Z]\w*)\)\s+echo "([^"]*)"', txt, re.M):
        sm = re.search(r'--steps (\d+)', m.group(2))
        if sm: steps[m.group(1)] = int(sm.group(1))
    out = {}
    for a in arms:
        base = re.sub(r's[12]$', '', a)
        out[a] = steps.get(a) or steps.get(base) or default_steps
    return out

def workers():
    try: return max(int((ROOT / "runs" / "pod_workers.txt").read_text().strip()), 1)
    except Exception: return 1

def pod_zone():
    """Latest zone the supervisor saw the node in (READY/CREATED lines), or None."""
    try: lines = LOG.read_text().splitlines()[-400:]
    except Exception: return None
    for l in reversed(lines):
        m = re.search(r'\| READY (\S+) \|', l) or re.search(r'CREATED in (\S+)', l) or re.search(r'READY (\S+)', l)
        if m: return m.group(1)
        if "ABSENT" in l or "DOWN " in l or "PREEMPTED" in l: return None
    return None

def ssh(pod, zone, cmd, timeout=70, worker=0):
    args = ["gcloud", "compute", "tpus", "tpu-vm", "ssh", pod, f"--zone={zone}", f"--project={PROJECT}"]
    if worker: args.append(f"--worker={worker}")
    r = subprocess.run(args + ["--command", cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout

def gcs_done(gcs, arms):
    try:
        r = subprocess.run(["gsutil", "ls", gcs + "/"], capture_output=True, text=True, timeout=40)
        names = set(os.path.basename(x) for x in r.stdout.split())
        return {a: (f"{a}_ckpt.pkl" in names) for a in arms}, names
    except Exception:
        return {a: False for a in arms}, set()

def bar(f, w=28):
    n = int(round(max(0.0, min(1.0, f)) * w)); return "█" * n + "░" * (w - n)

def fmt_eta(ts):
    return "-" if ts is None else dt.datetime.fromtimestamp(ts, IST).strftime("%H:%M IST")

def render_html(now_ist, now_utc, reach, stale, phases, rows, arms, steps, T, p1_eta, complete_eta, interval):
    def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;")
    ph = [x for x in phases.split() if x] if phases else []
    chips = "".join(f'<span class="chip {"ok" if ("OK" in x or "COMPLETE" in x) else "ev"}">{esc(x)}</span>' for x in ph) or '<span class="chip run">PHASE1 in progress</span>'
    trs = []
    for arm in arms:
        r = rows[arm]; st = r.get("step") or 0; tot = steps[arm] if isinstance(steps, dict) else steps; f = st / tot; done = r.get("done")
        status = "DONE · banked" if done else ("training" if r.get("step") else "queued / compiling")
        if r.get("stale") and not done: status += " (stale)"
        cls = "done" if done else ("run" if r.get("step") else "wait")
        rate = f'{r["rate"]:.2f}' if r.get("rate") else "–"
        trs.append(f'<tr class="{cls}"><td class="arm">{arm}<span class="t">T{T.get(re.sub(r"s[12]$", "", arm), "?")}</span></td>'
                   f'<td class="num">{st:,}<span class="of">/{tot:,}</span></td>'
                   f'<td class="barcell"><div class="bar"><div class="fill" style="width:{f*100:.1f}%"></div></div></td>'
                   f'<td class="num">{f*100:.1f}%</td><td class="num">{rate}</td><td class="num">{esc(fmt_eta(r.get("eta")))}</td><td class="st">{esc(status)}</td></tr>')
    p1 = fmt_eta(p1_eta) if p1_eta else ("reached" if "PHASE1-OK" in (phases or "") else "measuring…")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="{max(30, interval // 5)}">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>S2 meter</title>
<style>
:root{{--bg:#f7f7f5;--fg:#1b1b1b;--mut:#6b6b6b;--card:#fff;--line:#e6e6e2;--fill:#2f6df6;--done:#1f9d55;--wait:#b7b7b2;--chip:#eef2ff;--chipok:#e6f6ec;--chipev:#fff7e6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#101214;--fg:#e9e9e6;--mut:#9a9a96;--card:#181b1f;--line:#2a2e34;--fill:#5b8cff;--done:#3cc07a;--wait:#4a4f57;--chip:#1e2536;--chipok:#17302a;--chipev:#342b17}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:20px}}
.card{{max-width:860px;margin:0 auto;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px}}
h1{{font-size:16px;margin:0 0 2px;font-weight:650}} .sub{{color:var(--mut);font-size:12.5px;margin-bottom:12px}}
.chips{{margin:8px 0 14px;display:flex;flex-wrap:wrap;gap:6px}} .chip{{font-size:11.5px;padding:2px 8px;border-radius:999px;background:var(--chip)}} .chip.ok{{background:var(--chipok)}} .chip.ev{{background:var(--chipev)}}
table{{width:100%;border-collapse:collapse}} th{{text-align:left;color:var(--mut);font-weight:500;font-size:11.5px;padding:4px 6px;border-bottom:1px solid var(--line)}}
td{{padding:7px 6px;border-bottom:1px solid var(--line);vertical-align:middle;font-variant-numeric:tabular-nums}} tr:last-child td{{border-bottom:0}}
.arm{{font-weight:650}} .t{{color:var(--mut);font-weight:400;font-size:11.5px;margin-left:6px}} .num{{text-align:right;white-space:nowrap}} .of{{color:var(--mut)}}
.barcell{{width:36%}} .bar{{height:10px;border-radius:999px;background:var(--line);overflow:hidden}} .fill{{height:100%;background:var(--fill);border-radius:999px;transition:width .6s}}
tr.done .fill{{background:var(--done)}} tr.wait .fill{{background:var(--wait)}} .st{{color:var(--mut);white-space:nowrap;font-size:12.5px}}
.foot{{margin-top:14px;display:flex;gap:18px;flex-wrap:wrap;font-size:13px}} .foot b{{font-weight:650}} .note{{color:var(--mut);font-size:11.5px;margin-top:10px}}
</style></head><body><div class="card">
<h1>SPRINT S2 — Sudoku-Extreme (wave 2)</h1>
<div class="sub">updated {esc(now_ist)} ({esc(now_utc)}) · pod: {esc(reach)}{" · <b>pod down / hunting — last known + GCS bank</b>" if stale else ""} · data refreshes every {interval // 60} min</div>
<div class="chips">{chips}</div>
<table><thead><tr><th>arm</th><th style="text-align:right">step</th><th>progress</th><th style="text-align:right">%</th><th style="text-align:right">it/s (wall)</th><th style="text-align:right">ETA</th><th>status</th></tr></thead>
<tbody>{"".join(trs)}</tbody></table>
<div class="foot"><span>PHASE1-OK ETA: <b>{esc(p1)}</b></span><span>COMPLETE (est): <b>{esc(fmt_eta(complete_eta))}</b> <span style="color:var(--mut)">(+{PHASE2_MIN}m eval, +{PHASE3_MIN}m probes after PHASE1)</span></span></div>
<div class="note">pace = wall-clock steps/s measured between ticks (EMA). The trainer's printed it/s is per-step compute and reads ~3–4× high on remat arms. Read-only: one bounded ssh per tick, no model calls.</div>
</div></body></html>"""
    try: HTML.write_text(page)
    except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300); ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    e = env(); pod, gcs, tag = e["POD"], e["GCS"], e.get("R_TAG", "sport2")
    raw = [x for x in e["ARMS"].split() if not (x.startswith("scan:") or x.startswith("bscan:"))]
    arms = []
    for arm_ in raw:                   # W5 = two stages (W5gen then W5) — both rows (never rebind `a` = argparse namespace)
        if re.sub(r's[12]$', '', arm_) == "W5": arms.append(arm_ + "gen")
        arms.append(arm_)
    STEPS = arm_steps(ROOT / e.get("CHAIN_SCRIPT", "tools/chain_sport2.sh"), arms, int(e.get("R_STEPS", "20000")))
    steps = max(STEPS.values())        # legacy scalar (bars use per-arm STEPS)
    state = {}   # arm -> dict(t, step, rate)
    last_rows, last_phases, last_zone = {}, "", None
    while True:
        t_now = time.time(); zone = pod_zone(); rows = {}; phases = ""; reach = "unreachable"
        if zone:
            cmd = ("cd ~/qhrrn2 2>/dev/null || exit 3; for a in " + " ".join(arms) + "; do printf '%s|' $a; "
                   f"[ -f runs/pretrain{tag}_$a/.done ] && printf 'DONE|' || printf -- '-|'; "
                   "grep 'it/s' runs/wave_pre_$a.log 2>/dev/null | tail -1 | grep -oE 'step +[0-9]+' | awk '{printf $2}'; echo; done; "
                   "echo PHASES; grep -E 'SCAN-.*-OK|PRETRAIN-.*-OK|EVAL-.*-OK|PROBE-.*-OK|PROBE-SKIP|QUEUES-DONE|PHASE4|RESCUE-OK|CHAIN-SPORT2W?2?-[A-Z-]+|PHASE1-OK|PHASE2-OK|PHASE3-OK' "
                   "runs/detached.log 2>/dev/null | sed -E 's/ [0-9]{2}:[0-9]{2}$//; s/ 2026-.*$//; s/ worker=.*$//' | awk '!s[$0]++' | tr '\\n' ' '")
            for w in range(workers()):
                try:
                    rc, out = ssh(pod, zone, cmd, worker=w)
                    if rc == 0 and "PHASES" in out:
                        reach = f"READY {zone} ({workers()}w)"
                        head, _, ph = out.partition("PHASES")
                        phases = (phases + " " + ph.strip()).strip()
                        for l in head.splitlines():
                            p = l.strip().split("|")
                            if len(p) >= 3 and p[0] in arms and (p[1] == "DONE" or p[2].isdigit()):
                                st = int(p[2]) if p[2].isdigit() else None
                                rows[p[0]] = dict(done=(p[1] == "DONE"), step=st)
                except Exception:
                    pass
        if not rows:   # node down / hunting: fall back to the GCS bank + last known
            done, _ = gcs_done(gcs, arms)
            for arm in arms:
                prev = last_rows.get(arm, {})
                rows[arm] = dict(done=done[arm] or prev.get("done", False), step=prev.get("step"), stale=True)
            phases = last_phases
        for arm in arms:   # never KeyError on a partially parsed reply: carry last known
            if arm not in rows:
                prev = last_rows.get(arm, {}); rows[arm] = dict(done=prev.get("done", False), step=prev.get("step"), stale=True)
        # wall-clock pace
        for arm, r in rows.items():
            if r.get("done"): r["step"] = STEPS[arm]; r["rate"] = None; continue
            s = state.get(arm); st = r.get("step")
            if st is not None and s and s.get("step") is not None and st > s["step"] and t_now > s["t"]:
                inst = (st - s["step"]) / (t_now - s["t"])
                r["rate"] = inst if s.get("rate") is None else 0.5 * s["rate"] + 0.5 * inst
            else:
                r["rate"] = (s or {}).get("rate")
            if st is not None and not r.get("stale"): state[arm] = dict(t=t_now, step=st, rate=r["rate"])
        # ETAs
        p1_eta = None; any_training = False
        for arm, r in rows.items():
            if r.get("done") or r.get("step") is None: r["eta"] = None; continue
            any_training = True
            if r.get("rate"):
                r["eta"] = t_now + (STEPS[arm] - r["step"]) / r["rate"]; p1_eta = max(p1_eta or 0, r["eta"])
            else: r["eta"] = None
        complete_eta = None
        if "COMPLETE" in phases and "WORKER-DONE" not in phases.replace("CHAIN-SPORT2W2-WORKER-DONE", ""): complete_eta = t_now
        if re.search(r"CHAIN-SPORT2W?2?-COMPLETE", phases): complete_eta = t_now
        elif "PHASE1-OK" in phases or not any_training:
            base = t_now; complete_eta = base + (PHASE2_MIN + PHASE3_MIN) * 60 if "PHASE2-OK" not in phases else base + PHASE3_MIN * 60
        elif p1_eta: complete_eta = p1_eta + (PHASE2_MIN + PHASE3_MIN) * 60
        # render
        now_ist = dt.datetime.fromtimestamp(t_now, IST).strftime("%H:%M:%S IST"); now_utc = dt.datetime.fromtimestamp(t_now, dt.timezone.utc).strftime("%H:%MZ")
        L = [f"SPRINT S2 — Sudoku-Extreme {tag}   {now_ist} ({now_utc})   pod: {reach}" + ("   [pod down/hunting — showing last known + GCS bank]" if any(r.get('stale') for r in rows.values()) else ""),
             f"phases: {phases or '(queues in progress)'}", ""]
        L.append(f"{'arm':5s} {'T':>3s} {'step':>7s}/{'budget':<7s} {'progress':28s} {'%':>5s} {'it/s(wall)':>10s} {'ETA':>10s}  status")
        T = {"S0": 6, "S1": 6, "S2": 6, "S3": 12, "S4": 24, "S5": 6, "S6": 6, "S7": 12,
             "W1": 6, "W13": 6, "W2": 12, "W3": 12, "W4": 6, "W8": 6, "W9": 6, "W6": 6, "W5": 6, "W5gen": 6, "W7": 6}
        for arm in arms:
            r = rows[arm]; st = r.get("step"); f = (st or 0) / STEPS[arm]
            status = "DONE (banked)" if r.get("done") else ("training" if st else "queued/compiling")
            if r.get("stale") and not r.get("done"): status += " [stale]"
            L.append(f"{arm:5s} {T.get(re.sub(r's[12]$', '', arm), '?'):>3} {(st if st is not None else 0):>7d}/{STEPS[arm]:<7d} {bar(f)} {f*100:5.1f} "
                     f"{(f'{r['rate']:.2f}' if r.get('rate') else '-'):>10s} {fmt_eta(r.get('eta')):>10s}  {status}")
        L += ["", f"PHASE1-OK ETA: {fmt_eta(p1_eta) if p1_eta else ('reached' if 'PHASE1-OK' in phases else 'measuring…')}"
                  f"    COMPLETE (est, +{PHASE2_MIN}m eval +{PHASE3_MIN}m probes): {fmt_eta(complete_eta)}",
              "pace = wall-clock steps/s measured between ticks (EMA); the trainer's printed it/s is per-step compute and runs ~3-4x higher on remat arms."]
        txt = "\n".join(L)
        sys.stdout.write("\033[2J\033[H" + txt + "\n"); sys.stdout.flush()
        try: OUT.write_text(txt + "\n")
        except Exception: pass
        render_html(now_ist, now_utc, reach, any(r.get('stale') for r in rows.values()), phases, rows, arms, STEPS, T, p1_eta, complete_eta, a.interval)
        last_rows, last_phases, last_zone = rows, phases, zone
        if a.once or re.search(r"CHAIN-SPORT2W?2?-COMPLETE", phases): return
        time.sleep(a.interval)

if __name__ == "__main__":
    main()
