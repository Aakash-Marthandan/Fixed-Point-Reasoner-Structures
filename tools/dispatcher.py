# Ledger: §1.4 staged deploy / teardown discipline [P-M] — the one April
# subsystem that worked as designed — carried forward with the ledger's own
# amendments: --spot (ledger disposition "Kept; add --spot"), dead man's
# switch armed ALWAYS (not only --keep-alive: a slept laptop must never leave
# a TPU billing), account/project guard hard-coded (multiple Google accounts
# exist; only aakashemailbox@gmail.com / quantum-llm carries the credits).
"""QHRRN-2 TPU dispatcher: provision -> upload -> run -> rescue -> teardown.

Every phase is budget-defensive:
  - provisioning defaults to --spot v5litepod-1 in us-east1-c;
  - a remote `sudo shutdown -h +N` is armed immediately after provisioning,
    so the VM self-terminates even if this process (or the laptop) dies;
  - the run has a hard local wall-clock ceiling;
  - results are tarred and rescued in `finally` — always attempted;
  - teardown runs in `finally` unless --keep-alive, with a loud manual
    fallback instruction if the delete itself fails.

Usage (nothing executes without --cloud; --dry-run prints every gcloud
command without running anything):

  .venv/bin/python tools/dispatcher.py --cloud --dry-run \
      --cmd "python3 tools/run_gates.py --gate all --steps 600"
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time

ACCOUNT = "aakashemailbox@gmail.com"
PROJECT = "quantum-llm"
DEFAULT_ZONE = "us-east1-c"
DEFAULT_ACCEL = "v5litepod-1"
TPU_NAME = "qhrrn2-tpu"
TPU_VERSION = "tpu-ubuntu2204-base"
REMOTE_PROJECT = "~/qhrrn2"
UPLOADS = ["src", "tests", "tools", "requirements.txt", "pyproject.toml"]


def sh(cmd: str, *, dry: bool, check: bool = True, timeout: int | None = None):
    print(f"  $ {cmd}", flush=True)
    if dry:
        return None
    return subprocess.run(cmd, check=check, timeout=timeout, shell=True)


def guard_identity(dry: bool):
    """Refuse to run against the wrong account/project (memory-rule, enforced)."""
    acct = subprocess.run("gcloud config get-value account", shell=True,
                          capture_output=True, text=True).stdout.strip()
    proj = subprocess.run("gcloud config get-value project", shell=True,
                          capture_output=True, text=True).stdout.strip()
    ok = acct == ACCOUNT and proj == PROJECT
    print(f"identity guard: account={acct} project={proj} -> "
          + ("OK" if ok else f"MISMATCH (need {ACCOUNT} / {PROJECT})"), flush=True)
    if not ok and not dry:
        sys.exit(2)


def _stream(process):
    try:
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
    except (ValueError, OSError):
        pass  # pipe closed during kill — expected


def gssh(command: str, zone: str) -> str:
    return (f"gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone={zone} "
            f"--project={PROJECT} --command={shlex.quote(command)}")


def run_cloud(args):
    dry = args.dry_run
    guard_identity(dry)
    zone = args.zone
    dms_minutes = max(2 * args.wall_time // 60, 180)
    process = None

    try:
        if args.reuse_existing:
            print(f">>> Phase 1: SKIPPED — reusing '{TPU_NAME}' in {zone}")
        else:
            spot = " --spot" if not args.on_demand else ""
            print(f">>> Phase 1: Provisioning {args.accelerator}{spot or ' (ON-DEMAND)'}")
            sh(f"gcloud compute tpus tpu-vm create {TPU_NAME} --zone={zone} "
               f"--project={PROJECT} --accelerator-type={args.accelerator} "
               f"--version={TPU_VERSION}{spot}", dry=dry)

        print(f">>> Phase 1.5: Dead man's switch — ALWAYS armed (+{dms_minutes} min)")
        sh(gssh(f"sudo shutdown -h +{dms_minutes}", zone), dry=dry)

        print(">>> Phase 2: Upload")
        sh(gssh(f"mkdir -p {REMOTE_PROJECT}", zone), dry=dry)
        for item in UPLOADS + (["data/ARC-AGI/data"] if args.with_data else []):
            sh(f"gcloud compute tpus tpu-vm scp --recurse {item} "
               f"{TPU_NAME}:{REMOTE_PROJECT}/ --zone={zone} --project={PROJECT}",
               dry=dry)

        # Shakedown 1 lesson (2026-07-27): the VM image's system Python is too
        # old for the pinned jax==0.10.2 (pip cannot resolve it). Bootstrap an
        # exact modern interpreter with uv — parity with the local 3.14 venv,
        # no version skew (April E8).
        print(">>> Phase 3: Bootstrap (uv-managed Python 3.14 + pinned deps)")
        sh(gssh(f"export PATH=~/.local/bin:$PATH && cd {REMOTE_PROJECT} && "
                "python3 -m pip install -q uv && "
                "uv python install 3.14 && "
                "uv venv --python 3.14 .venv && "
                "uv pip install --python .venv/bin/python -q -r requirements.txt "
                "'jax[tpu]==0.10.2' "
                "-f https://storage.googleapis.com/jax-releases/libtpu_releases.html",
                zone), dry=dry)

        print(f">>> Phase 4: Run (ceiling {args.wall_time}s): {args.cmd}")
        run_inner = (f"cd {REMOTE_PROJECT} && "
                     f"PATH=$PWD/.venv/bin:$PATH PYTHONPATH=src {args.cmd}")
        full = gssh(run_inner, zone)
        print(f"  $ {full}")
        if not dry:
            process = subprocess.Popen(full, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, shell=True)
            threading.Thread(target=_stream, args=(process,), daemon=True).start()
            try:
                code = process.wait(timeout=args.wall_time)
                if code != 0:
                    print(f"WARNING: remote run exited {code}")
            except subprocess.TimeoutExpired:
                print(f"FATAL: wall-clock ceiling {args.wall_time}s hit; killing SSH")
                process.kill()
                process.wait()

    except Exception as e:
        print(f"FATAL: dispatcher crashed: {e}")
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()

    finally:
        print(">>> Phase 4.5: RESULTS RESCUE (always attempted)")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs("runs/cloud", exist_ok=True)
        try:
            sh(gssh(f"cd {REMOTE_PROJECT} && tar czf /tmp/qhrrn2_runs.tgz runs "
                    "2>/dev/null || true", zone), dry=dry, check=False, timeout=120)
            sh(f"gcloud compute tpus tpu-vm scp {TPU_NAME}:/tmp/qhrrn2_runs.tgz "
               f"runs/cloud/{stamp}.tgz --zone={zone} --project={PROJECT}",
               dry=dry, check=False, timeout=120)
        except Exception as e:  # rescue must never block teardown
            print(f"  WARNING: rescue failed: {e}")

        if args.keep_alive:
            print(f">>> Phase 5: TEARDOWN SKIPPED (--keep-alive); dead man's switch "
                  f"fires in <= {dms_minutes} min. Manual delete:\n"
                  f"  gcloud compute tpus tpu-vm delete {TPU_NAME} --quiet "
                  f"--zone={zone} --project={PROJECT}")
        else:
            print(">>> Phase 5: TEARDOWN (unconditional)")
            try:
                sh(f"gcloud compute tpus tpu-vm delete {TPU_NAME} --quiet "
                   f"--zone={zone} --project={PROJECT}", dry=dry, timeout=300)
                print("  teardown OK")
            except Exception as e:
                print(f"  CRITICAL: teardown failed ({e}). DELETE '{TPU_NAME}' "
                      f"MANUALLY in the GCP console NOW — it is billing.")


def main():
    ap = argparse.ArgumentParser(description="QHRRN-2 TPU dispatcher")
    ap.add_argument("--cloud", action="store_true", help="required to do anything")
    ap.add_argument("--cmd", default="python3 -m pytest -q",
                    help="remote command, run from repo root with PYTHONPATH=src")
    ap.add_argument("--accelerator", default=DEFAULT_ACCEL)
    ap.add_argument("--zone", default=DEFAULT_ZONE)
    ap.add_argument("--wall-time", type=int, default=7200, help="local SSH ceiling, s")
    ap.add_argument("--on-demand", action="store_true", help="disable --spot")
    ap.add_argument("--with-data", action="store_true", help="upload vendored ARC data")
    ap.add_argument("--keep-alive", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every gcloud command; execute nothing")
    args = ap.parse_args()
    if not args.cloud:
        ap.error("nothing to do: pass --cloud (optionally with --dry-run)")
    run_cloud(args)


if __name__ == "__main__":
    main()
