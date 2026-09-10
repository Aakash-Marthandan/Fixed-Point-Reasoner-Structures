#!/bin/bash
# G1 FULL ROW on the pilot node's CPUs (the TPU chips run the chain): the natives' colour x dihedral invariance,
# 48 val-hard tasks x 6 transforms x 2 maps, 16 shards per map split over the two hosts (W = this host: 0 -> shards 0-7, 1 -> 8-15).
# Usage on the node: W=<host index> bash g1_node.sh   (detached: nohup ... > runs/g1_node.log 2>&1 &)
cd "$HOME/qhrrn2" || exit 1
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
W=${W:-0}; SETS=gs://qhrrn2-rescue/decarc/sets; OUTB=gs://qhrrn2-rescue/decarc_pilot/g1
mkdir -p runs/pretrain13_Dri runs/pretrain13_C53
[ -f runs/pretrain13_Dri/ckpt_053333.pkl ] || gsutil -q cp $SETS/p13Dri.pkl runs/pretrain13_Dri/ckpt_053333.pkl
[ -f runs/pretrain13_C53/ckpt_053333.pkl ] || gsutil -q cp $SETS/p13C53.pkl runs/pretrain13_C53/ckpt_053333.pkl
echo "G1-START host=$W $(date -u +%FT%TZ) nproc=$(nproc)"
pids=()
for N in Dri C53; do
  for i in $(seq $((W * 8)) $((W * 8 + 7))); do
    nohup python3 tools/lens_orbit_invariance_arc.py --ckpt runs/pretrain13_$N/ckpt_053333.pkl --set valhard --dump-z --shard $i/16 --out runs/orbit_arc_full_p13$N > runs/g1_p13${N}_$i.log 2>&1 &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait $p; done
tar czf /tmp/g1_host$W.tgz runs/orbit_arc_full_p13Dri runs/orbit_arc_full_p13C53 runs/g1_*.log && gsutil -q cp /tmp/g1_host$W.tgz $OUTB/g1_host$W.tgz
echo "G1-DONE host=$W $(date -u +%FT%TZ)"
