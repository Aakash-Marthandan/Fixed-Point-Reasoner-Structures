#!/bin/bash
# G1 FULL ROW on the node — RULE (ops incident 2026-09-10): runs ONLY after CHAIN-DECARC-COMPLETE with the supervisor STOPPED,
# on the TPU CHIPS (one pinned process per chip, <= 8), never on the host CPUs (32 JAX-CPU shards thrashed the host at load 405
# and starved the chain's battery for 80 min). The natives' colour x dihedral invariance: 48 val-hard tasks x 6 transforms x 2 maps,
# 8 shards per map, the two maps in sequence (8 chips each). Usage on the node: nohup bash tools/g1_node.sh > runs/g1_node.log 2>&1 &
cd "$HOME/qhrrn2" || exit 1
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src
SETS=gs://qhrrn2-rescue/decarc/sets; OUTB=${G1_OUT:-gs://qhrrn2-rescue/decarc/g1}
NCHIP=$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]' || true); [ "$NCHIP" -ge 1 ] 2>/dev/null || NCHIP=8
if pgrep -f "tools/chain_decarc.sh" >/dev/null || pgrep -f "tools/pretrain.py" >/dev/null || pgrep -f "tools/eval_decarc.py" >/dev/null; then
  echo "G1-REFUSED: the chain is running on this node (the node host and chips belong to the chain until COMPLETE)"; exit 3
fi
pin () { local c=$1; shift; TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$c "$@"; }
mkdir -p runs/pretrain13_Dri runs/pretrain13_C53
[ -f runs/pretrain13_Dri/ckpt_053333.pkl ] || gsutil -q cp $SETS/p13Dri.pkl runs/pretrain13_Dri/ckpt_053333.pkl
[ -f runs/pretrain13_C53/ckpt_053333.pkl ] || gsutil -q cp $SETS/p13C53.pkl runs/pretrain13_C53/ckpt_053333.pkl
echo "G1-START $(date -u +%FT%TZ) chips=$NCHIP"
for N in Dri C53; do
  pids=()
  for i in $(seq 0 $((NCHIP - 1))); do
    pin "$i" nohup python3 tools/lens_orbit_invariance_arc.py --ckpt runs/pretrain13_$N/ckpt_053333.pkl --set valhard --dump-z --shard "$i/$NCHIP" --out runs/orbit_arc_full_p13$N > runs/g1_p13${N}_$i.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  python3 tools/lens_orbit_invariance_arc.py --out runs/orbit_arc_full_p13$N --summarize --ckpt x > runs/orbit_arc_full_p13$N/summary.log 2>&1
  echo "G1-MAP-DONE $N $(date -u +%FT%TZ)"
done
tar czf /tmp/g1_full.tgz runs/orbit_arc_full_p13Dri runs/orbit_arc_full_p13C53 runs/g1_*.log && gsutil -q cp /tmp/g1_full.tgz "$OUTB/g1_full.tgz"
echo "G1-DONE $(date -u +%FT%TZ)"
