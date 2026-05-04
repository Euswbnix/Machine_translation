#!/usr/bin/env bash
# Multi-seed training launcher for WMT26 upgrade.
#
# Runs 3 additional seeds (1, 2, 3) for each cell in the 2×2 ablation.
# Original runs used seed=42; those results are kept as-is.
#
# Usage:
#   bash scripts/wmt26_upgrade/run_all_seeds.sh           # run all
#   bash scripts/wmt26_upgrade/run_all_seeds.sh base_v1   # run one cell
#
# Each run appends _s{seed} to checkpoint dir and experiment name.
# Estimated total GPU time: ~84h on a single 5090.

set -euo pipefail
cd "$(dirname "$0")/../.."   # cd to Machine_translation root

SEEDS=(1 2 3)

declare -A CONFIGS
CONFIGS[base_v1]="configs/base_en_fr_v1_redo.yaml"
CONFIGS[big_v1]="configs/big_en_fr_v1_redo.yaml"
CONFIGS[base_v2]="configs/base_en_fr.yaml"
CONFIGS[big_v2]="configs/big_en_fr.yaml"
CONFIGS[base_ende]="configs/base_en_de.yaml"
CONFIGS[big_ende]="configs/big_en_de.yaml"

# Estimated hours per run (for progress reporting)
declare -A HOURS
HOURS[base_v1]=2.5
HOURS[big_v1]=6
HOURS[base_v2]=14
HOURS[big_v2]=9
HOURS[base_ende]=3
HOURS[big_ende]=7

FILTER="${1:-all}"

for cell in base_v1 big_v1 base_v2 big_v2 base_ende big_ende; do
    if [[ "$FILTER" != "all" && "$FILTER" != "$cell" ]]; then
        continue
    fi
    for seed in "${SEEDS[@]}"; do
        suffix="_s${seed}"
        echo "============================================"
        echo "  Cell: ${cell}  Seed: ${seed}  (~${HOURS[$cell]}h)"
        echo "  Config: ${CONFIGS[$cell]}"
        echo "  Suffix: ${suffix}"
        echo "============================================"
        python train.py \
            --config "${CONFIGS[$cell]}" \
            --seed "$seed" \
            --suffix "$suffix"
        echo "  >>> ${cell} seed=${seed} DONE"
        echo ""
    done
done

echo "All multi-seed runs complete."
