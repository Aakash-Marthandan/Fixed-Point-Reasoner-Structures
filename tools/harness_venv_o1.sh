#!/bin/bash
# Offline stub harness for the O1 venv-tarball half (dispatcher.py venv_url /
# venv_restore_cmd / venv_bank_cmd), built 2026-08-27 during the rung-2 ride.
# Runs the EXACT remote command strings in a sandboxed $HOME with a stub gsutil
# — no cloud, no node, no side effects outside the sandbox. Scenarios:
#   S1 restore-ok        (object present, venv validates -> boot_ok, RESTORE-OK)
#   S2 already-boot      (boot_ok present -> short-circuit, gsutil never called)
#   S3 object-missing    (pull fails -> MISS, sandbox left clean for bootstrap)
#   S4 bad-validate      (tarball extracts, python import fails -> wiped + MISS)
#   S5 bank-when-absent  (boot_ok + stat-miss -> tar + upload + VENV-BANKED)
#   S6 bank-when-present (stat-hit -> no upload)
#   S7 url-derivation    (explicit var wins; else code-archive dir + sha8; else None)
set -uo pipefail
cd "$(dirname "$0")/.."
SB=$(mktemp -d /tmp/h_venv_o1.XXXXXX)
trap 'rm -rf "$SB"' EXIT
FH="$SB/home"; mkdir -p "$FH/qhrrn2" "$SB/bin" "$SB/gcs" "$SB/fix"
PASS=0; FAIL=0
say () { echo "  $1"; }
chk () { if eval "$2"; then PASS=$((PASS+1)); say "PASS $1"; else FAIL=$((FAIL+1)); say "FAIL $1"; fi; }

# ---- stub gsutil (mode via files in $SB) ----
cat > "$SB/bin/gsutil" <<EOF
#!/bin/bash
echo "\$*" >> "$SB/gsutil.log"
args=(); for a in "\$@"; do [ "\$a" = "-q" ] && continue; args+=("\$a"); done
case "\${args[0]}" in
  cp)
    if [[ "\${args[1]}" == gs://* ]]; then   # download
      [ -f "$SB/mode_have" ] || exit 1
      cp "\$(cat "$SB/mode_have")" "\${args[2]}"; exit 0
    else                                      # upload
      cp "\${args[1]}" "$SB/gcs/\$(basename "\${args[2]}")"; exit 0
    fi;;
  stat) [ -f "$SB/mode_stat_present" ] && exit 0 || exit 1;;
esac
exit 1
EOF
chmod +x "$SB/bin/gsutil"

# ---- fixtures: GOOD and BAD venv tarballs (linux-shaped relative layout) ----
mkfix () { # $1=name $2=python-exit-code
  local R="$SB/fix/$1"; rm -rf "$R"
  mkdir -p "$R/qhrrn2/.venv/bin" "$R/.local/share/uv/python/cpython-stub/bin"
  printf '#!/bin/sh\nexit %s\n' "$2" > "$R/qhrrn2/.venv/bin/python"
  chmod +x "$R/qhrrn2/.venv/bin/python"
  touch "$R/.local/share/uv/python/cpython-stub/bin/python3.14"
  (cd "$R" && tar czf "$SB/fix/$1.tgz" qhrrn2/.venv .local/share/uv/python)
}
mkfix good 0
mkfix bad 1

URL="gs://stub-bucket/sportBr2/venv_v6e_deadbeef.tgz"
RESTORE=$(.venv/bin/python -c "import sys; sys.path.insert(0,'tools'); import dispatcher; print(dispatcher.venv_restore_cmd('$URL'))")
BANK=$(.venv/bin/python -c "import sys; sys.path.insert(0,'tools'); import dispatcher; print(dispatcher.venv_bank_cmd('$URL'))")
run_remote () { env -i HOME="$FH" PATH="$SB/bin:/usr/bin:/bin" bash -c "$1" 2>&1; }

echo "== S1 restore-ok =="
rm -rf "$FH/qhrrn2/.venv"; : > "$SB/gsutil.log"; echo "$SB/fix/good.tgz" > "$SB/mode_have"
OUT=$(run_remote "$RESTORE")
chk "S1 echoes VENV-RESTORE-OK"        "grep -q VENV-RESTORE-OK <<<\"\$OUT\""
chk "S1 boot_ok created"                "[ -f '$FH/qhrrn2/.venv/.boot_ok' ]"
chk "S1 interpreter tree extracted"     "[ -f '$FH/.local/share/uv/python/cpython-stub/bin/python3.14' ]"

echo "== S2 already-boot =="
: > "$SB/gsutil.log"
OUT=$(run_remote "$RESTORE")
chk "S2 short-circuits"                 "grep -q 'already bootstrapped' <<<\"\$OUT\""
chk "S2 gsutil never called"            "[ ! -s '$SB/gsutil.log' ]"

echo "== S3 object-missing =="
rm -rf "$FH/qhrrn2/.venv" "$SB/mode_have"; : > "$SB/gsutil.log"
OUT=$(run_remote "$RESTORE")
chk "S3 echoes MISS"                    "grep -q VENV-RESTORE-MISS <<<\"\$OUT\""
chk "S3 no venv left behind"            "[ ! -e '$FH/qhrrn2/.venv' ]"
chk "S3 exits 0 (non-fatal)"            "run_remote \"$RESTORE\" >/dev/null"

echo "== S4 bad-validate =="
rm -rf "$FH/qhrrn2/.venv" "$FH/.local"; echo "$SB/fix/bad.tgz" > "$SB/mode_have"; : > "$SB/gsutil.log"
OUT=$(run_remote "$RESTORE")
chk "S4 echoes MISS"                    "grep -q VENV-RESTORE-MISS <<<\"\$OUT\""
chk "S4 broken venv wiped"              "[ ! -e '$FH/qhrrn2/.venv' ]"

echo "== S5 bank-when-absent =="
rm -rf "$FH/qhrrn2/.venv" "$SB/mode_stat_present"
mkdir -p "$FH/qhrrn2/.venv/bin" "$FH/.local/share/uv/python/cpython-stub"
touch "$FH/qhrrn2/.venv/.boot_ok"; : > "$SB/gsutil.log"
OUT=$(run_remote "$BANK")
chk "S5 echoes VENV-BANKED"             "grep -q VENV-BANKED <<<\"\$OUT\""
chk "S5 object uploaded"                "[ -f '$SB/gcs/venv_v6e_deadbeef.tgz' ]"

echo "== S6 bank-when-present =="
touch "$SB/mode_stat_present"; rm -f "$SB/gcs/venv_v6e_deadbeef.tgz"; : > "$SB/gsutil.log"
OUT=$(run_remote "$BANK")
chk "S6 no upload"                      "[ ! -f '$SB/gcs/venv_v6e_deadbeef.tgz' ]"
chk "S6 bank exits 0"                   "run_remote \"$BANK\" >/dev/null"

echo "== S7 url-derivation =="
D7=$(.venv/bin/python - <<'PY'
import sys, os, hashlib
sys.path.insert(0, "tools")
import dispatcher
h = hashlib.sha256(open("requirements.txt", "rb").read()).hexdigest()[:8]
os.environ.pop("QHRRN_VENV_TGZ", None)
os.environ["QHRRN_CODE_TGZ"] = "gs://b/sportBr2/code_abc.tgz"
r1 = dispatcher.venv_url("v6e-16") == f"gs://b/sportBr2/venv_v6e_{h}.tgz"
os.environ["QHRRN_VENV_TGZ"] = "gs://explicit/x.tgz"
r2 = dispatcher.venv_url("v6e-8") == "gs://explicit/x.tgz"
os.environ.pop("QHRRN_VENV_TGZ"); os.environ.pop("QHRRN_CODE_TGZ")
r3 = dispatcher.venv_url("v6e-8") is None
print("OK" if (r1 and r2 and r3) else f"BAD {r1} {r2} {r3}")
PY
)
chk "S7 derivation (derived/explicit/none)" "[ \"$D7\" = OK ]"

echo "== SUMMARY: PASS=$PASS FAIL=$FAIL =="
[ "$FAIL" -eq 0 ]
