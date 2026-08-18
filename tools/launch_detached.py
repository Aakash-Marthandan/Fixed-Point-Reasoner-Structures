# Ledger: detached launcher WITHOUT a local poller (2026-08-18).
#
# WHY THIS EXISTS: `dispatcher run --detach` correctly detaches the remote
# job but then the LOCAL process polls SSH until the job exits. Every hunter
# wrapper today failed on that: (1) unbounded -> hunter blocked for the chain's
# life, never supervising (wedge #1); (2) wall-clock alarm -> killed the call
# mid code-sync on a slow link, before the remote nohup fired -> "chain did
# not start" -> teardown loop (wedge #2). Timing hacks around a poller are the
# wrong layer. This tool does exactly the durable part and RETURNS:
#     sync code -> re-arm DMS -> ONE bounded SSH that launches under
#     setsid nohup and writes runs/detached.pid -> verify pid alive -> exit 0
# It reuses dispatcher's own functions (sync_code, arm_dms, gssh, sh) so the
# remote launch line is byte-identical to the proven one, incl. the
# double-launch guard. Supervision is the hunter's job, not this tool's.
#
# Exit codes: 0 launched+verified | 7 remote busy (double-launch guard) |
#             3 launch ssh failed | 4 verify failed (no live pid)
"""
  .venv/bin/python tools/launch_detached.py --name POD --zone Z --cmd "..." [--wall-time S]
"""
from __future__ import annotations
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dispatcher as D  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--wall-time", type=int, default=30600,
                    help="remote ceiling (s): the job is wrapped in `timeout` "
                         "on the VM so a runaway never outlives the DMS")
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args()

    D.TPU_NAME = a.name           # dispatcher helpers read the module global
    D.guard_identity(False)
    D.arm_dms(a.zone, False)
    if not a.no_sync:
        D.sync_code(a.zone, False, False)

    # remote job: same shape as dispatcher's, plus a REMOTE wall ceiling so a
    # hung chain cannot outlive the DMS window without any local poller.
    inner = shlex.quote(
        f"export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src; "
        f"timeout {a.wall_time} bash -c {shlex.quote(a.cmd)}; "
        "echo $? > runs/detached.exit")
    launch = (f"cd {D.REMOTE_PROJECT} && mkdir -p runs && "
              "if test -f runs/detached.pid && kill -0 $(cat runs/detached.pid) "
              "2>/dev/null; then echo detached-BUSY; exit 7; fi && "
              "rm -f runs/detached.exit runs/detached.pid && "
              f"(setsid nohup sh -c {inner} < /dev/null > runs/detached.log 2>&1 & "
              "echo $! > runs/detached.pid) && echo detached-launch-ok")
    r = subprocess.run(D.gssh(launch, a.zone), shell=True, capture_output=True,
                       text=True, timeout=180)
    if "detached-BUSY" in r.stdout:
        print("launch: remote job already running (double-launch guard) — treating as launched")
        return 7
    if r.returncode != 0 or "detached-launch-ok" not in r.stdout:
        print(f"launch: ssh failed rc={r.returncode}\n{r.stdout[-400:]}\n{r.stderr[-400:]}")
        return 3

    # VERIFY: the pid is alive (one more bounded ssh)
    v = subprocess.run(D.gssh(f"cd {D.REMOTE_PROJECT} && test -f runs/detached.pid && "
                              "kill -0 $(cat runs/detached.pid) 2>/dev/null && echo ALIVE",
                              a.zone),
                       shell=True, capture_output=True, text=True, timeout=120)
    if "ALIVE" not in v.stdout:
        print("launch: pid not alive after launch — verify FAILED")
        return 4
    print("launch: detached + verified alive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
