# Ledger: §1.4 staged deploy / teardown discipline [P-M], amended twice:
#   2026-07-27 (ledger §5): --spot default; dead man's switch ALWAYS armed;
#     identity guard hard-coded; uv-managed Python 3.14 (shakedown-1 lesson).
#   2026-07-27 night (PI-directed policy change): April's per-run
#     provision→teardown cycle replaced by a SESSION-PERSISTENT workflow —
#     `up` once, many fast `run`s, `down` at session end. The dead man's
#     switch (re-armed on every run, default +10 h) is the forgetting
#     backstop, not the workflow. `cycle` keeps the old unattended
#     one-shot semantics for pretraining. Comfort > cents; the retained
#     protections are the ones that cost no comfort: identity guard, DMS,
#     rescue, and gates-before-training-spend.
"""QHRRN-2 TPU dispatcher.

  up      provision (spot default) + arm DMS + upload + bootstrap (idempotent)
  run     re-arm DMS + sync code + execute --cmd (wall ceiling) + rescue;
          --detach = reset-proof: remote nohup + short-SSH polling (REQUIRED
          for anything longer than ~15 min — attached SSH resets, and gcloud
          auto-retry re-runs the command)
  down    rescue + delete
  status  describe the VM (read-only)
  cycle   up + run + unconditional down — unattended mode (pretraining)

Typical day:  up  →  run --cmd "..."  (× many)  →  down
`--dry-run` on any subcommand prints the gcloud commands without executing.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

# Identity comes from the environment (or the git-ignored tools/.gcp_identity,
# two lines: account then project) so no personal account lands in the public
# repo; the guard refuses to run without both.
def _identity():
    acct = os.environ.get("QHRRN_GCP_ACCOUNT", "")
    proj = os.environ.get("QHRRN_GCP_PROJECT", "")
    if not (acct and proj):
        try:
            lines = (Path(__file__).parent / ".gcp_identity").read_text().split()
            acct, proj = acct or lines[0], proj or lines[1]
        except Exception:
            pass
    return acct, proj


ACCOUNT, PROJECT = _identity()
DEFAULT_ZONE = "us-east1-c"
DEFAULT_ACCEL = "v5litepod-1"
TPU_NAME = "qhrrn2-tpu"
TPU_VERSION = "tpu-ubuntu2204-base"
REMOTE_PROJECT = "~/qhrrn2"
UPLOADS = ["src", "tests", "tools", "requirements.txt", "pyproject.toml"]
DMS_MINUTES = 600  # dead man's switch: 10 h, re-armed on every `run`


def sh(cmd: str, *, dry: bool, check: bool = True, timeout: int | None = 600):
    """Every call bounded (2026-07-29: an unbounded ceiling-kill SSH hung ~7 h
    overnight and delayed teardown past the DMS window — billing ran on)."""
    print(f"  $ {cmd}", flush=True)
    if dry:
        return None
    return subprocess.run(cmd, check=check, timeout=timeout, shell=True)


def gssh(command: str, zone: str) -> str:
    return (f"gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone={zone} "
            f"--project={PROJECT} --command={shlex.quote(command)}")


def guard_identity(dry: bool):
    """Refuse to act against the wrong account/project (memory-rule, enforced)."""
    if not ACCOUNT or not PROJECT:
        print("identity guard: set QHRRN_GCP_ACCOUNT and QHRRN_GCP_PROJECT "
              "(the guard refuses to run cloud commands without them)")
        sys.exit(2)
    acct = subprocess.run("gcloud config get-value account", shell=True,
                          capture_output=True, text=True).stdout.strip()
    proj = subprocess.run("gcloud config get-value project", shell=True,
                          capture_output=True, text=True).stdout.strip()
    ok = acct == ACCOUNT and proj == PROJECT
    print(f"identity guard: account={acct} project={proj} -> "
          + ("OK" if ok else f"MISMATCH (need {ACCOUNT} / {PROJECT})"), flush=True)
    if not ok and not dry:
        sys.exit(2)


def vm_state(zone: str) -> str | None:
    r = subprocess.run(
        f"gcloud compute tpus tpu-vm describe {TPU_NAME} --zone={zone} "
        f"--project={PROJECT} --format='value(state)'",
        shell=True, capture_output=True, text=True)
    return r.stdout.strip() or None if r.returncode == 0 else None


def arm_dms(zone: str, dry: bool, minutes: int = DMS_MINUTES):
    print(f">>> Dead man's switch: re-arm +{minutes} min")
    sh(gssh(f"sudo shutdown -c 2>/dev/null || true; sudo shutdown -h +{minutes}",
            zone), dry=dry, check=False)


def _stream(process):
    try:
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
    except (ValueError, OSError):
        pass  # pipe closed during kill — expected


def sync_code(zone: str, dry: bool, with_data: bool):
    print(">>> Sync code")
    sh(gssh(f"mkdir -p {REMOTE_PROJECT}", zone), dry=dry)
    for item in UPLOADS + (["data/ARC-AGI/data"] if with_data else []):
        sh(f"gcloud compute tpus tpu-vm scp --recurse {item} "
           f"{TPU_NAME}:{REMOTE_PROJECT}/ --zone={zone} --project={PROJECT}",
           dry=dry)


def rescue(zone: str, dry: bool):
    print(">>> Results rescue")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("runs/cloud", exist_ok=True)
    try:
        sh(gssh(f"cd {REMOTE_PROJECT} && tar czf /tmp/qhrrn2_runs.tgz runs "
                "2>/dev/null || true", zone), dry=dry, check=False, timeout=120)
        sh(f"gcloud compute tpus tpu-vm scp {TPU_NAME}:/tmp/qhrrn2_runs.tgz "
           f"runs/cloud/{stamp}.tgz --zone={zone} --project={PROJECT}",
           dry=dry, check=False, timeout=120)
    except Exception as e:  # rescue must never block anything
        print(f"  WARNING: rescue failed: {e}")


def cmd_up(args) -> int:
    guard_identity(args.dry_run)
    state = None if args.dry_run else vm_state(args.zone)
    if state:
        print(f">>> up: '{TPU_NAME}' already exists (state={state}) — reusing")
    else:
        spot = "" if args.on_demand else " --spot"
        print(f">>> up: provisioning {args.accelerator}{spot or ' (ON-DEMAND)'}")
        sh(f"gcloud compute tpus tpu-vm create {TPU_NAME} --zone={args.zone} "
           f"--project={PROJECT} --accelerator-type={args.accelerator} "
           f"--version={TPU_VERSION}{spot}", dry=args.dry_run)
    arm_dms(args.zone, args.dry_run)
    sync_code(args.zone, args.dry_run, args.with_data)
    # Idempotent bootstrap: uv-managed CPython 3.14 (parity with local venv),
    # exact pins incl. jax[tpu]==0.10.2 (shakedown-1: system Python too old).
    print(">>> Bootstrap (skipped if .venv/.boot_ok present)")
    sh(gssh(f"export PATH=~/.local/bin:$PATH && cd {REMOTE_PROJECT} && "
            "test -f .venv/.boot_ok && echo 'bootstrap: already done' || ("
            "python3 -m pip install -q uv && "
            "uv python install 3.14 && "
            "uv venv --python 3.14 .venv && "
            "uv pip install --python .venv/bin/python -q -r requirements.txt "
            "'jax[tpu]==0.10.2' "
            "-f https://storage.googleapis.com/jax-releases/libtpu_releases.html "
            "&& touch .venv/.boot_ok)", args.zone), dry=args.dry_run)
    print(f">>> up: READY. DMS fires in {DMS_MINUTES} min unless re-armed; "
          f"`down` when finished.")
    return 0


def _run_detached(args) -> int:
    """Reset-proof execution (ledger night-3 (g)): the job runs under setsid+
    nohup on the VM writing to a remote log; the local side POLLS with short
    SSH calls, so a dropped connection can neither kill the job nor re-run it
    (the gcloud auto-retry hazard for non-idempotent training)."""
    # Launch forensics (2026-07-28): `A && B & C` parses as `(A && B) & C` —
    # the pid write ran OUTSIDE the project cwd and the background group held
    # the SSH channel, hanging the launch until a reset (which gcloud then
    # retried, double-launching). Fix: subshell keeps cwd for the pid write,
    # and stdin/out/err are ALL detached so sshd closes immediately.
    inner = shlex.quote(f"PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src {args.cmd}; "
                        "echo $? > runs/detached.exit")
    # Double-launch guard (2026-07-28: a pkill'd LOCAL dispatcher left its
    # REMOTE job running; the next launch collided on the TPU): refuse if the
    # pid-file process is still alive.
    launch = (f"cd {REMOTE_PROJECT} && mkdir -p runs && "
              "if test -f runs/detached.pid && kill -0 $(cat runs/detached.pid) "
              "2>/dev/null; then echo detached-BUSY; exit 7; fi && "
              "rm -f runs/detached.exit runs/detached.pid && "
              f"(setsid nohup sh -c {inner} < /dev/null > runs/detached.log 2>&1 & "
              "echo $! > runs/detached.pid) && "
              "echo detached-launch-ok")
    sh(gssh(launch, args.zone), dry=args.dry_run, timeout=120)
    if args.dry_run:
        print("  (dry-run: poll loop skipped)")
        return 0
    print(f">>> polling every {args.poll_interval}s (ceiling {args.wall_time}s); "
          "SSH drops are tolerated; periodic rescue every ~10 min", flush=True)
    offset, t0, misses, polls = 0, time.time(), 0, 0
    while True:
        polls += 1
        if polls % max(600 // args.poll_interval, 1) == 0:
            rescue(args.zone, False)  # reflection rule 3: data survives any death
        if time.time() - t0 > args.wall_time:
            print(f"FATAL: ceiling {args.wall_time}s hit; rescuing then killing remote job")
            rescue(args.zone, False)  # data first — a killed sweep's rows are still rows
            try:
                sh(gssh(f"cd {REMOTE_PROJECT} && kill -- -$(cat runs/detached.pid) "
                        "2>/dev/null || true", args.zone), dry=False, check=False,
                   timeout=120)
            except Exception as e:
                print(f"  WARNING: remote kill failed ({e}); teardown will handle it")
            return 124
        time.sleep(args.poll_interval)
        poll = (f"cd {REMOTE_PROJECT} && "
                "S=$(test -f runs/detached.exit && cat runs/detached.exit || echo RUNNING); "
                "B=$(wc -c < runs/detached.log); "
                f"echo \"@@STATUS $S $B\" && tail -c +{offset + 1} runs/detached.log")
        try:
            r = subprocess.run(gssh(poll, args.zone), shell=True,
                               capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            misses += 1
            print(f"  (poll ssh timeout x{misses} — job unaffected, retrying)", flush=True)
            if misses >= 10:
                print("FATAL: 10 consecutive poll failures; job may still be "
                      "running on the VM — check manually before re-running")
                return 5
            continue
        if r.returncode != 0:
            misses += 1
            print(f"  (poll ssh failed x{misses} — job unaffected, retrying)", flush=True)
            if misses >= 10:
                print("FATAL: 10 consecutive poll failures; job may still be "
                      "running on the VM — check manually before re-running")
                return 5
            continue
        misses = 0
        head, _, chunk = r.stdout.partition("\n")
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        parts = head.split()
        status, nbytes = parts[1], int(parts[2])
        offset = max(offset, nbytes)
        if status != "RUNNING":
            return int(status)


def cmd_run(args) -> int:
    guard_identity(args.dry_run)
    if args.wall_time > (DMS_MINUTES - 90) * 60:
        print(f"run: REFUSED — wall-time {args.wall_time}s does not fit inside the "
              f"dead-man's-switch window ({DMS_MINUTES} min) with 90 min margin. "
              "The DMS blocks SSH when it fires (and does NOT stop billing); "
              "everything must finish, rescue, and tear down before it.")
        return 6
    if not args.dry_run and vm_state(args.zone) is None:
        print(f"run: no VM '{TPU_NAME}' — `up` first")
        return 3
    arm_dms(args.zone, args.dry_run)
    if not args.no_sync:
        sync_code(args.zone, args.dry_run, args.with_data)
    print(f">>> run (ceiling {args.wall_time}s"
          + (", detached" if args.detach else "") + f"): {args.cmd}")
    if args.detach:
        code = _run_detached(args)
    else:
        full = gssh(f"cd {REMOTE_PROJECT} && "
                    f"PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src {args.cmd}", args.zone)
        print(f"  $ {full}", flush=True)
        code = 0
        if not args.dry_run:
            process = subprocess.Popen(full, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, shell=True)
            threading.Thread(target=_stream, args=(process,), daemon=True).start()
            try:
                code = process.wait(timeout=args.wall_time)
            except subprocess.TimeoutExpired:
                print(f"FATAL: wall-clock ceiling {args.wall_time}s hit; killing SSH")
                process.kill()
                process.wait()
                code = 124
    rescue(args.zone, args.dry_run)
    print(f">>> run: exit {code}. VM stays up (DMS backstop armed); `down` to stop.")
    return code


def cmd_down(args) -> int:
    guard_identity(args.dry_run)
    rescue(args.zone, args.dry_run)
    print(">>> down: teardown")
    try:
        sh(f"gcloud compute tpus tpu-vm delete {TPU_NAME} --quiet "
           f"--zone={args.zone} --project={PROJECT}", dry=args.dry_run, timeout=300)
        print("  teardown OK")
    except Exception as e:
        print(f"  CRITICAL: teardown failed ({e}). DELETE '{TPU_NAME}' "
              f"MANUALLY in the GCP console NOW — it is billing.")
        return 4
    return 0


def cmd_status(args) -> int:
    guard_identity(True)
    state = vm_state(args.zone)
    print(f"status: {TPU_NAME} in {args.zone}: "
          + (f"state={state}" if state else "not found"))
    return 0


def cmd_cycle(args) -> int:
    """Unattended one-shot: up + run + unconditional down (pretraining mode)."""
    code = 1
    try:
        code = cmd_up(args)
        if code == 0:
            code = cmd_run(args)
    finally:
        down_code = cmd_down(args)
        code = code or down_code
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--zone", default=DEFAULT_ZONE)
    common.add_argument("--dry-run", action="store_true")
    up_like = argparse.ArgumentParser(add_help=False)
    up_like.add_argument("--accelerator", default=DEFAULT_ACCEL)
    up_like.add_argument("--on-demand", action="store_true", help="disable --spot")
    up_like.add_argument("--with-data", action="store_true",
                         help="upload vendored ARC data")
    run_like = argparse.ArgumentParser(add_help=False)
    run_like.add_argument("--cmd", default="python3 -m pytest -q")
    run_like.add_argument("--wall-time", type=int, default=7200)
    run_like.add_argument("--no-sync", action="store_true")
    run_like.add_argument("--detach", action="store_true",
                          help="reset-proof: run remotely under nohup, poll via short SSH")
    run_like.add_argument("--poll-interval", type=int, default=30)

    sub.add_parser("up", parents=[common, up_like])
    sub.add_parser("run", parents=[common, up_like, run_like])
    sub.add_parser("down", parents=[common])
    sub.add_parser("status", parents=[common])
    sub.add_parser("cycle", parents=[common, up_like, run_like])

    args = ap.parse_args()
    sys.exit({"up": cmd_up, "run": cmd_run, "down": cmd_down,
              "status": cmd_status, "cycle": cmd_cycle}[args.verb](args))


if __name__ == "__main__":
    main()
