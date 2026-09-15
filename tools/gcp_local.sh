# Ledger: THE ARC ERA ops (2026-09-15). `source tools/gcp_local.sh` loads the git-ignored identity + project policy
# (tools/.gcp_local.env; template tools/gcp_local.env.example; GCP_LOCAL_ENV overrides the path for the offline harness) and
# defines the shared-project predicates every ops tool uses. Returns 1 (sets nothing) when the file is missing — callers refuse.
# It never sets PROJECT or QHRRN_GCP_PROJECT (the campaign env and the caller decide those).
_gcp_local_d=${BASH_SOURCE[0]:+$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}   # bash; other shells fall back to ./tools (every ops tool runs from the repo root)
_gcp_local_f=${GCP_LOCAL_ENV:-${_gcp_local_d:-tools}/.gcp_local.env}
if [ ! -f "$_gcp_local_f" ]; then
  echo "GCP-LOCAL-MISSING: $_gcp_local_f (copy tools/gcp_local.env.example and fill it in)" >&2
  return 1 2>/dev/null || exit 1
fi
set -a; . "$_gcp_local_f"; set +a
for _v in QHRRN_GCP_ACCOUNT SUDOKU_PROJECT OWN_PREFIX; do   # eval, not ${!v}: the loader must work when sourced from zsh as well as bash
  eval "_gcp_val=\${$_v:-}"
  if [ -z "$_gcp_val" ]; then echo "GCP-LOCAL-INCOMPLETE: $_v unset in $_gcp_local_f" >&2; return 1 2>/dev/null || exit 1; fi
done
is_shared_project () { case " ${SHARED_PROJECTS:-} " in *" $1 "*) return 0 ;; esac; return 1; }
is_spot_only_project () { case " ${SPOT_ONLY_PROJECTS:-} " in *" $1 "*) return 0 ;; esac; return 1; }
is_own_name () { [ -n "$1" ] && [ "${1#"$OWN_PREFIX"}" != "$1" ]; }
