#!/bin/sh
# Population gate on val-hard (ConceptARC holdout): $1 = ckpt, $2 = out.
set -e
V=$(python -c "import json; print(','.join(json.load(open('tools/valhard.json'))['valhard']))")
python tools/eval_pop.py --ckpt "$1" --tasks "$V" --steps 600 --out "$2"
echo "POP VALHARD DONE: $2"
