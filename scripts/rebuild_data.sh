#!/usr/bin/env bash
# End-to-end: clean the parallel corpus, retrain SPM on the clean data,
# then back up the old SPM model.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Step 1: cleaning training data ==="
python scripts/clean_data.py \
    --src data/train.zh \
    --tgt data/train.en \
    --out-src data/train.clean.zh \
    --out-tgt data/train.clean.en

echo
echo "=== Step 2: backing up old SPM model ==="
if [ -f data/spm.model ]; then
    mv data/spm.model data/spm.model.old
    mv data/spm.vocab data/spm.vocab.old
    echo "  old spm moved to data/spm.model.old"
fi

echo
echo "=== Step 3: training new SPM on cleaned data ==="
python scripts/train_tokenizer.py \
    --inputs data/train.clean.zh data/train.clean.en \
    --model-prefix data/spm \
    --vocab-size 32000

echo
echo "=== Done ==="
echo "Next: update configs/base.yaml to point at the cleaned files:"
echo "  train_src: data/train.clean.zh"
echo "  train_tgt: data/train.clean.en"
echo "Then start a fresh training run (delete old checkpoints/ first)."
