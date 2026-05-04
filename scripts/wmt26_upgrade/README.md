# WMT26 Upgrade Pipeline

## Overview

Upgrades the WMT paper from single-seed/BLEU-only to multi-seed + COMET + en-zh + domain-conditional FT.

## Execution Order

### Phase 1: Multi-seed pretraining (~84 GPU-hours)

```bash
# Run from Machine_translation/ root
bash scripts/wmt26_upgrade/run_all_seeds.sh

# Or run individual cells:
bash scripts/wmt26_upgrade/run_all_seeds.sh base_v1
bash scripts/wmt26_upgrade/run_all_seeds.sh big_v1
# ... etc
```

### Phase 2: COMET evaluation (~2-3 hours)

```bash
# Evaluate all checkpoints (original seed=42 + new seeds 1,2,3)
python scripts/wmt26_upgrade/eval_comet.py \
    --checkpoint-glob "checkpoints/*/best.pt" \
    --out-dir results/
```

### Phase 3: en-zh language pair (~40-60 GPU-hours)

```bash
# Download and clean data
python scripts/wmt26_upgrade/download_wmt17_enzh.py
python scripts/wmt26_upgrade/clean_data_enzh.py \
    --src data_enzh/train.en --tgt data_enzh/train.zh \
    --src-out data_enzh/train.clean.en --tgt-out data_enzh/train.clean.zh

# Train tokenizer
python scripts/train_tokenizer.py \
    --src data_enzh/train.clean.en --tgt data_enzh/train.clean.zh \
    --model-prefix data_enzh/spm_enzh --vocab-size 32000 \
    --character-coverage 0.9995

# Train Base + Big (3 seeds each)
python train.py --config configs/base_en_zh.yaml
python train.py --config configs/base_en_zh.yaml --seed 1 --suffix _s1
python train.py --config configs/base_en_zh.yaml --seed 2 --suffix _s2
python train.py --config configs/base_en_zh.yaml --seed 3 --suffix _s3

python train.py --config configs/big_en_zh.yaml
python train.py --config configs/big_en_zh.yaml --seed 1 --suffix _s1
python train.py --config configs/big_en_zh.yaml --seed 2 --suffix _s2
python train.py --config configs/big_en_zh.yaml --seed 3 --suffix _s3
```

### Phase 4: Domain-conditional FT (~8 hours)

```bash
# From Machine-Translation-SFT/ root

# Step 1: Train domain classifier
python ../Machine_translation/scripts/wmt26_upgrade/domain_filter.py train-classifier \
    --data-dir ../Machine_translation/data/ \
    --model-out models/domain_classifier.bin

# Step 2: Filter scored TSV for news domain
python ../Machine_translation/scripts/wmt26_upgrade/domain_filter.py filter \
    --scored-tsv data/v2_scored.tsv \
    --classifier models/domain_classifier.bin \
    --target-domain news \
    --top-k 1000000 \
    --src-out data/sft_news_train.en \
    --tgt-out data/sft_news_train.fr

# Step 3: Fine-tune Base and Big on news-filtered data
python ../Machine_translation/train.py \
    --config configs/sft_base_enfr_news.yaml \
    --resume checkpoints/base_enfr_v1_redo/averaged.pt \
    --reset-optimizer

python ../Machine_translation/train.py \
    --config configs/sft_big_enfr_news.yaml \
    --resume checkpoints/big_enfr_v1_redo/averaged.pt \
    --reset-optimizer
```

### Phase 5: LLM-as-judge evaluation (~$5-10 API cost)

```bash
python scripts/wmt26_upgrade/llm_judge.py \
    --src data_enfr_v1/test.en --ref data_enfr_v1/test.fr \
    --hyp-a translations/base_v1.txt \
    --hyp-b translations/big_v1.txt \
    --label-a "Base v1.1 (60M)" --label-b "Big v1.1 (209M)" \
    --num-samples 200 \
    --out results/judge_base_vs_big_v1.json
```

### Phase 6: Aggregate results

```bash
python scripts/wmt26_upgrade/aggregate_results.py \
    --results-dir results/ \
    --out results/summary.json
```

## Time Estimates

| Phase | GPU-hours | Wall time (1× 5090) |
|-------|-----------|---------------------|
| Multi-seed (en-fr + en-de) | ~84h | ~3.5 days |
| COMET eval | ~3h | 3h |
| en-zh (Base + Big × 4 seeds) | ~50h | ~2 days |
| Domain-conditional FT | ~8h | 8h |
| LLM judge | 0 (API) | ~30 min |
| **Total** | **~145h** | **~7 days sequential** |
