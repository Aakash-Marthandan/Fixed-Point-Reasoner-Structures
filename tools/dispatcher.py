# Ledger: §1.4 staged deploy / teardown discipline [P-M], amended twice:
#   2026-07-27 (ledger §5): --spot default; dead man's switch ALWAYS armed;
#     identity guard hard-coded; uv-managed Python 3.14 (shakedown-1 lesson).
#   2026-07-27 night (PI-directed policy change): the prior per-run
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
    for item in UPLOADS:
        sh(f"gcloud compute tpus tpu-vm scp --recurse {item} "
           f"{TPU_NAME}:{REMOTE_PROJECT}/ --zone={zone} --project={PROJECT}",
           dry=dry)
    if with_data:
        # 2026-08-01: recursive scp of ~800 small JSONs blew the 600 s sh()
        # ceiling (per-file overhead) AND landed at ~/qhrrn2/data — the loader
        # wants data/ARC-AGI/data relative to repo root. One tar stream fixes
        # both: single connection, relative paths preserved end-to-end.
        # Same day, second bite: macOS bsdtar writes mac-metadata PAX entries
        # that bsdtar's OWN listing hides but GNU tar extracts as literal
        # ._*.json files (AppleDouble), which the *.json glob then loads.
        # COPYFILE_DISABLE=1 suppresses them at create; --exclude at extract
        # is defense in depth against a foreign-made archive.
        print(">>> Sync data (tar stream)")
        extra = " data/ConceptARC" if os.path.isdir("data/ConceptARC") else ""
        sh(f"COPYFILE_DISABLE=1 tar czf - data/ARC-AGI/data{extra} | "
           + gssh(f"rm -rf {REMOTE_PROJECT}/data && "
                  f"tar xzf - -C {REMOTE_PROJECT} --exclude \"._*\"",
                  zone), dry=dry, timeout=300)


def rescue(zone: str, dry: bool):
    print(">>> Results rescue")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("runs/cloud", exist_ok=True)
    try:
        sh(gssh(f"cd {REMOTE_PROJECT} && tar czf /tmp/qhrrn2_runs.tgz runs "
                "2>/dev/null || true", zone), dry=dry, check=False, timeout=120)
        # fleet mode: rescue files carry the VM name so lanes never collide
        sh(f"gcloud compute tpus tpu-vm scp {TPU_NAME}:/tmp/qhrrn2_runs.tgz "
           f"runs/cloud/{TPU_NAME}-{stamp}.tgz --zone={zone} --project={PROJECT}",
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
            # libtpu OVERRIDE pin (2026-08-07): jax 0.10.2 resolves
            # libtpu==0.0.42.1, which stack-overflows in the XLA fusion cost
            # estimator (FusedSpatialMajorConvolution) on fresh VMs built from
            # the rolled tpu-ubuntu2204-base image — identical wheels ran clean
            # on the 08-05/06 fleet (env-rot data point #2; ulimit -s no cure:
            # fiber stacks are fixed-size). 0.0.43.1 (released 08-04) fixes it.
            "&& uv pip install --python .venv/bin/python -q --no-deps "
            "libtpu==0.0.43.1 "
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
    # export form (2026-08-05): `VAR=x cmd1 && cmd2` binds VAR to cmd1 ONLY —
    # a chained cmd2 silently ran under system python (no numpy).
    inner = shlex.quote(f"export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src; {args.cmd}; "
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
            # 2026-07-30: SSH handshake to a loaded 4-wide VM can take minutes;
            # 120s misread latency as death (rescues at 600s succeeded through
            # the same window). Generous per-poll cap; misses still bounded.
            r = subprocess.run(gssh(poll, args.zone), shell=True,
                               capture_output=True, text=True, timeout=420)
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
            backoff = min(args.poll_interval * (2 ** max(misses - 5, 0)), 300)
            print(f"  (poll ssh failed x{misses} — job unaffected; next try in "
                  f"~{backoff}s)", flush=True)
            if misses >= 20:  # ~45+ min of outage with backoff (2026-07-30:
                # a spot host dropped SSH for minutes and 10 fast misses
                # aborted supervision of a healthy job)
                print("FATAL: 20 consecutive poll failures; job may still be "
                      "running on the VM — check manually before re-running")
                return 5
            time.sleep(max(backoff - args.poll_interval, 0))
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
                    f"export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src; {args.cmd}", args.zone)
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
          + (f"state={state}" if state else "state=not found"))
    # fleet awareness: teardown vigilance must see EVERY lane, not just --name
    r = subprocess.run(
        f"gcloud compute tpus tpu-vm list --zone={args.zone} --project={PROJECT} "
        "--format='value(name,state)'", shell=True, capture_output=True, text=True)
    fleet = [l for l in r.stdout.strip().splitlines() if l]
    print(f"fleet: {len(fleet)} VM(s) in {args.zone}"
          + (": " + "; ".join(fleet) if fleet else ""))
    if state and args.jobs:
        # Detached-job forensics (2026-08-07 incident: manual kills left local
        # pollers aimed at a REUSED pid file; job state must be inspectable
        # before any launch/kill decision).
        try:
            sh(gssh(f"cd {REMOTE_PROJECT} 2>/dev/null && "
                    "if test -f runs/detached.pid && "
                    "kill -0 $(cat runs/detached.pid) 2>/dev/null; then "
                    "echo \"job: RUNNING pid=$(cat runs/detached.pid)\"; "
                    "tail -1 runs/detached.log 2>/dev/null | cut -c1-100; else "
                    "echo \"job: idle exit=$(cat runs/detached.exit 2>/dev/null "
                    "|| echo none)\"; fi; "
                    "for f in runs/*/results.jsonl; do test -f \"$f\" && "
                    "echo \"  $f: $(wc -l < \"$f\") rows\"; done",
                    args.zone), dry=False, timeout=120)
        except Exception as e:
            print(f"  job probe failed: {e}")
    return 0


def cmd_canary(args) -> int:
    """Known-good-command gate (STANDING PRACTICE, ledger 2026-08-07): after
    any provision/re-provision and BEFORE any campaign, run one short
    single-bulk population fit and verify the completion sentinel. The 08-07
    incident (host image rolled under identical wheels; 96 cells lost) is the
    reason this exists. Exit 0 = lane certified; anything else = STOP."""
    guard_identity(args.dry_run)
    if vm_state(args.zone) is None:
        print(f"canary: no VM '{TPU_NAME}' — `up` first")
        return 3
    arm_dms(args.zone, args.dry_run)
    marker = "CANARY-PASS"
    cmd = (f"rm -rf runs/_canary && python tools/eval_pop.py "
           f"--ckpt {args.canary_ckpt} --tasks {args.canary_task} "
           f"--steps 60 --val-every 30 --out runs/_canary "
           f"&& echo {marker}")
    full = gssh(f"cd {REMOTE_PROJECT} && "
                f"export PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src; {cmd}",
                args.zone)
    print(f">>> canary ({args.canary_task}, 60 steps): {TPU_NAME}")
    if args.dry_run:
        print(f"  $ {full}")
        return 0
    r = subprocess.run(full, shell=True, capture_output=True, text=True,
                       timeout=1800)
    passed = marker in r.stdout
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-3:])
    print(tail)
    print(f">>> canary: {'PASS' if passed else 'FAIL'} "
          f"(sentinel {'found' if passed else 'MISSING'}, ssh exit {r.returncode})")
    return 0 if passed else 5


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
    global TPU_NAME
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--zone", default=DEFAULT_ZONE)
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--name", default=TPU_NAME,
                        help="VM name (fleet mode 2026-08-02: run several "
                             "lanes concurrently, e.g. qhrrn2-a/-b/-c)")
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
    status_p = sub.add_parser("status", parents=[common])
    status_p.add_argument("--jobs", action="store_true",
                          help="also probe detached-job state + results rows")
    canary_p = sub.add_parser("canary", parents=[common])
    canary_p.add_argument("--canary-ckpt",
                          default="runs/pretrain6_d24/ckpt_latest.pkl")
    canary_p.add_argument("--canary-task", default="ca_AboveBelow5")
    sub.add_parser("cycle", parents=[common, up_like, run_like])

    args = ap.parse_args()
    TPU_NAME = args.name  # every helper reads the module global
    sys.exit({"up": cmd_up, "run": cmd_run, "down": cmd_down,
              "status": cmd_status, "cycle": cmd_cycle,
              "canary": cmd_canary}[args.verb](args))


if __name__ == "__main__":
    main()
