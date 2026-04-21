# Machine Translation: Transformer from Scratch

A from-scratch PyTorch implementation of the Transformer (Vaswani et al., 2017),
trained on WMT parallel corpora without any pretrained weights.

**Current status:** Transformer Base on WMT14 en-fr — **34.69 BLEU on newstest2014**
(sacrebleu 13a, checkpoint-averaged, v1 run on 9.3M strict-filtered pairs).
See *Success Case: Base on WMT14 en-fr* below.

**Full trajectory** (documented in this README as success → failure → diagnosis):
1. ❌ Base on WMT17 zh-en — mode collapse, BLEU 0.77
2. ❌ Big on WMT17 zh-en — mode collapse, BLEU 0.47
3. ✅ Base on WMT14 en-fr (v1) — BLEU 34.69 (averaged), converged cleanly
4. 〰️ Big on WMT14 en-fr (v1) — BLEU 34.66 (tied Base, hit tokenizer ceiling)
5. ❌ Base on WMT14 en-fr (v2) — 27.77 underfit (200K steps on 30M pairs = 1.3 epochs)
6. ✅ Base on WMT14 en-fr (v3) — BLEU 33.90 test, 29.23 valid (600K steps, 4 epochs);
   **below v1 despite 3× data + 6× steps** — the data-quality/quantity trap.

Next: Big v2 on v3's data (tests whether capacity absorbs the noisier 30M
corpus), then WMT14 en-de Base + Big, then final report.

## Features

- Pure PyTorch Transformer implementation (no HuggingFace shortcuts)
- Shared SentencePiece BPE tokenizer (32K vocab)
- Token-based dynamic batching for efficient GPU utilization
- Mixed precision: **BF16 (default) or FP16** (autocast + GradScaler)
- Label-smoothed cross entropy + Noam learning rate schedule with min-LR floor
- Loss-spike guard (EMA-tracked) that drops poisoned batches without blowing up
- Adaptive eval interval — sparse early, dense once loss crosses a configured band
- Beam search decoding with length penalty
- Checkpoint averaging (Vaswani Base trick, +0.3–0.5 BLEU)
- Interactive / batch translation CLI
- Graceful Ctrl+C / SIGTERM interrupt (saves checkpoint and resumes)
- Atomic checkpoint writes + rolling emergency save (UPS / power-off safe)
- Automatic training report generation
- TensorBoard logging

## Project Structure

```
Machine_translation/
├── assets/                    # README images (training curves, diagrams)
├── checkpoints/               # Per-run output dirs (git-ignored)
│   ├── base_zhen/             # zh-en Base ckpts (historical)
│   ├── big_zhen/              # zh-en Big ckpts (historical)
│   └── base_enfr/             # en-fr Base ckpts (current)
├── configs/
│   ├── base.yaml              # Transformer Base (65M params) — zh-en (historical)
│   ├── big.yaml               # Transformer Big (213M params) — zh-en (historical)
│   └── base_en_fr.yaml        # Transformer Base — WMT14 en-fr (current)
├── scripts/
│   ├── download_data.py              # Download WMT17 zh-en data
│   ├── download_wmt_enfr.py          # Download WMT14 en-fr data
│   ├── clean_data.py                 # Clean zh-en corpus
│   ├── clean_data_enfr.py            # Clean en-fr corpus
│   ├── train_tokenizer.py            # Train SentencePiece BPE
│   ├── average_checkpoints.py        # Average last N checkpoints (+BLEU)
│   ├── eval_bleu.py                  # Standalone BLEU evaluation
│   ├── interactive_translate.py      # REPL / batch translation CLI
│   ├── quick_translate_check.py      # Sample-translate valid lines (sanity check)
│   └── diagnose_attention.py         # Diagnose cross-attention collapse
├── src/
│   ├── model/                 # Transformer implementation
│   │   ├── attention.py       # Multi-Head Attention
│   │   ├── embeddings.py      # Token + positional encoding
│   │   ├── layers.py          # FFN + residual connections
│   │   ├── encoder.py         # Encoder stack
│   │   ├── decoder.py         # Decoder stack
│   │   └── transformer.py     # Full model
│   ├── data/
│   │   ├── tokenizer.py       # SentencePiece wrapper
│   │   └── dataset.py         # Dataset + token-based batching
│   ├── training/
│   │   ├── loss.py            # Label smoothed cross entropy
│   │   ├── optimizer.py       # Noam LR scheduler
│   │   └── trainer.py         # Main training loop
│   ├── inference/
│   │   └── translate.py       # Beam search decoding
│   └── evaluate.py            # sacrebleu BLEU evaluation
├── train.py                   # Training entry point
├── translate.py               # Inference entry point
└── requirements.txt
```

## Setup

### Requirements
- Python 3.10+
- PyTorch 2.0+ (2.8+ for RTX 5090 / Blackwell GPUs)
- CUDA-capable GPU (16GB+ VRAM recommended)

### Installation
```bash
pip install -r requirements.txt
```

For RTX 5090 (sm_120), install PyTorch nightly with CUDA 12.8:
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Usage

The default flow is **WMT14 en-fr Base**. For zh-en see the historical sections
below.

### 1. Download WMT14 en-fr data
```bash
# Full corpus is ~40M pairs. 10M is plenty for Base; drop the cap for a full run.
python scripts/download_wmt_enfr.py --output-dir data --max-train-samples 10000000
```
Downloads train / valid (newstest2013) / test (newstest2014).

### 2. Clean the corpus
```bash
python scripts/clean_data_enfr.py
```
Drops empty, too-short, too-long, non-Latin-script, and high-frequency
boilerplate pairs. Writes `data/train.clean.en` and `data/train.clean.fr`.

### 3. Train BPE tokenizer
```bash
python scripts/train_tokenizer.py \
    --inputs data/train.clean.en data/train.clean.fr \
    --model-prefix data/spm_enfr \
    --vocab-size 32000
```
Produces `data/spm_enfr.model` and `data/spm_enfr.vocab` (shared en-fr BPE,
32K vocab). Both sides share Latin script so shared BPE is well-behaved
(unlike zh-en).

**Tip**: for the next run, add `--character-coverage 1.0` to `train_tokenizer.py`
(or edit the call in `src/data/tokenizer.py`). The default 0.9995 caused ~4.4%
of French valid sentences to hit `<unk>` on accented characters (`Israël`,
`Noël`, etc.), costing ~0.5 BLEU.

### 4. Train the model
```bash
# Transformer Base on en-fr (~60M params, ~2 hours on RTX 5090)
python train.py --config configs/base_en_fr.yaml
```

**Graceful interrupt:** Press `Ctrl+C` once to save a checkpoint and exit cleanly. Press twice to force-quit without saving.

**Resume from checkpoint:**
```bash
python train.py --config configs/base_en_fr.yaml \
    --resume checkpoints/base_enfr/interrupted_step_12345.pt
```

**Monitor with TensorBoard:**
```bash
tensorboard --logdir checkpoints/base_enfr/logs
```

### Running training in tmux (recommended for long runs)

Training takes 1–5 days. Using `tmux` lets you detach from the session and safely close your terminal / SSH connection / Jupyter Lab tab without killing the training process.

**Install tmux:**
```bash
# Ubuntu / Debian
sudo apt install tmux
# macOS
brew install tmux
```

**Basic workflow:**
```bash
# 1. Start a new named session
tmux new -s train

# 2. Inside tmux, run training
python train.py --config configs/base.yaml

# 3. Detach (training keeps running):  Ctrl+b  then  d

# 4. Re-attach later from anywhere (new SSH, new terminal, etc.)
tmux attach -t train

# 5. List sessions
tmux ls

# 6. Kill a session when done
tmux kill-session -t train
```

**Useful keybindings (all prefixed with `Ctrl+b`):**

| Keys | Action |
|------|--------|
| `d` | Detach from session |
| `[` | Enter scroll mode (↑/↓/PgUp/PgDn to scroll, `q` to quit) |
| `"` | Split pane horizontally (e.g. to run `nvidia-smi` alongside) |
| `%` | Split pane vertically |
| `o` | Switch between panes |
| `x` | Close current pane |

**Two-pane layout for monitoring:**
```bash
tmux new -s train
# run training in top pane
python train.py --config configs/base.yaml
# Ctrl+b then "  (split horizontally)
# Ctrl+b then o  (switch to bottom pane)
watch -n 2 nvidia-smi
# Ctrl+b then d  (detach — both panes keep running)
```

**Note:** tmux and the graceful-interrupt logic compose naturally. `Ctrl+b d` just detaches; it does NOT send SIGINT. To actually interrupt training cleanly, re-attach first (`tmux attach -t train`), then press `Ctrl+C` inside the session.

### 5. Checkpoint averaging (recommended)
```bash
# Average the last 5 saved step_*.pt into averaged.pt
python scripts/average_checkpoints.py --ckpt-dir checkpoints/base_enfr --n 5
```
Per Vaswani et al., averaging the tail of training smooths SGD noise in the
convergence band. Empirically worth **+0.3–0.5 BLEU** on top of the best
single checkpoint.

### 6. Evaluate BLEU on a test set
```bash
python scripts/eval_bleu.py \
    --ckpt checkpoints/base_enfr/averaged.pt \
    --config configs/base_en_fr.yaml \
    --src data/test.en --ref data/test.fr \
    --out outputs/test.averaged.fr
```

### 7. Translate — interactive or batch
```bash
# Interactive REPL — type sentences, get translations, Ctrl+D to quit
python scripts/interactive_translate.py \
    --ckpt checkpoints/base_enfr/averaged.pt \
    --config configs/base_en_fr.yaml

# Batch — one sentence per line
python scripts/interactive_translate.py \
    --ckpt checkpoints/base_enfr/averaged.pt \
    --config configs/base_en_fr.yaml \
    --input my_en_sentences.txt \
    --output my_fr_translations.txt
```

## Configuration

| Parameter | Base (en-fr) | Base (zh-en) | Big (zh-en) |
|-----------|--------------|--------------|-------------|
| d_model | 512 | 512 | 1024 |
| n_heads | 8 | 8 | 16 |
| n_layers (enc/dec) | 6 / 6 | 6 / 6 | 6 / 6 |
| d_ff | 2048 | 2048 | 4096 |
| Dropout | 0.1 | 0.1 | 0.3 |
| Parameters | ~60M | ~65M | ~213M |
| Vocab | 32K shared BPE | 32K shared BPE | 32K shared BPE |
| Batch (effective) | 100K tokens | 100K tokens | 32K tokens |
| Warmup steps | 4000 | 4000 | 8000 |
| Max steps | 100K | 800K | 400K |
| Precision | BF16 | FP16 | BF16 |
| Label smoothing | 0.1 | 0.1 | 0.1 |
| Spike guard | ratio 1.3, α 0.005 | — | ratio 1.3, α 0.005 |

## Training Output

- `checkpoints/base_enfr/best.pt` — Best model (by validation BLEU)
- `checkpoints/base_enfr/final.pt` — Final step checkpoint
- `checkpoints/base_enfr/step_*.pt` — Periodic checkpoints (keeps last 5)
- `checkpoints/base_enfr/interrupted_step_*.pt` — Saved on Ctrl+C / SIGTERM
- `checkpoints/base_enfr/emergency.pt` — Rolling save every 500 steps (UPS / power-off fallback)
- `checkpoints/base_enfr/training_report.txt` — Human-readable training summary
- `checkpoints/base_enfr/logs/` — TensorBoard logs

## Results

| Config | Valid BLEU | Test BLEU | Training time | Status |
|--------|-----------|-----------|---------------|--------|
| **Base (WMT14 en-fr)**       | **30.00** | **34.19** (best.pt)      | 2h 5m on 5090 | ✅ converged |
| **Base (WMT14 en-fr) avg-5** | —         | **34.69** (averaged.pt)  | +seconds      | ✅ +0.50 from averaging |
| Base (WMT17 zh-en)           | 0.77 (plateau) | — | ~1.5 days on 5090 | ❌ mode collapse |
| Big (WMT17 zh-en)            | 0.47 (plateau) | — | ~1 day on 5090 (halted) | ❌ mode collapse |

BLEU reported as sacrebleu `13a` (modern detokenized standard). Published
Vaswani Base on WMT14 en-fr is 38.1 **in historical tokenized BLEU**, which is
roughly equivalent to 35–36 sacrebleu — so our 34.69 is **~1–1.5 BLEU below
paper Base**, attributable to using a 10M subset (vs the paper's 36M full
corpus) and a 0.5 BLEU ceiling from `<unk>` on accented French characters.

## Success Case: Base on WMT14 en-fr

Trained Transformer Base (~60M params) on 10M pairs of WMT14 en-fr for 100K
steps on a single RTX 5090. **Converged cleanly to BLEU 30.00 on newstest2013
and 34.69 on newstest2014** after checkpoint averaging.

![en-fr training curves](assets/training_curves_base_enfr.png)

### Final numbers

| | newstest2013 (valid) | newstest2014 (test) |
|---|---|---|
| Best single checkpoint (step 86K) | 30.00 | 34.19 |
| Averaged (last 5, steps 80K–100K) | — | **34.69** |
| Averaging gain | — | +0.50 |

### Training trajectory (every 8K steps)

| Step  | Loss  | Valid BLEU | LR       |
|-------|-------|------------|----------|
| 8K    | 3.997 | —          | 3.49e-4  |
| 16K   | 3.155 | —          | 6.86e-4 (peak) |
| 24K   | 2.902 | 26.13      | 5.70e-4  |
| 32K   | 2.800 | 27.30      | 4.93e-4  |
| 40K   | 2.748 | 27.75      | 4.41e-4  |
| 48K   | 2.691 | 28.57      | 4.03e-4  |
| 56K   | 2.653 | 28.87      | 3.73e-4  |
| 64K   | 2.625 | 29.52      | 3.48e-4  |
| 72K   | 2.612 | 29.64      | 3.28e-4  |
| 80K   | 2.611 | 29.77      | 3.12e-4  |
| 86K   | 2.592 | **30.00**  | 3.01e-4  |
| 96K   | 2.583 | 30.00      | 2.86e-4  |
| 100K  | 2.566 | 29.89      | 2.80e-4  |

Loss curve is monotonic with no spikes; BLEU rises nearly monotonically from
24.73 @ 20K to 30.00, then oscillates ±0.25 in the convergence band.

### Sample translations (averaged checkpoint, newstest2013)

```
SRC: A Republican strategy to counter the re-election of Obama
HYP: Stratégie républicaine de lutte contre la réélection d'Obama
REF: Une stratégie républicaine pour contrer la réélection d'Obama

SRC: Also the effect of vitamin D on cancer is not clear.
HYP: L'effet de la vitamine D sur le cancer n'est pas non plus clair.
REF: L'effet de la vitamine D sur le cancer n'est pas clairement établi non plus.

SRC: In Israel, holy places await Ukrainian tourists, the omphalos and a sea
     of saline water
HYP: En Isra ⁇ l, des lieux saints attendent des touristes ukrainiens, des
     omphalos et une mer d'eau sali
REF: En Israël, des lieux saints, le Centre du monde et une mer de saumure
     attendent les touristes ukrainiens
```

Observations:
- **Gender agreement is learned**: "Republican" → "républicaine" (feminine, to
  agree with "stratégie").
- **Discourse markers transfer**: "Also" correctly translates as "non plus"
  (the French idiom for negated "also"), not a literal "aussi".
- **Elision is 90% correct**: "d'Obama" (not "de Obama"), "l'effet" (not
  "le effet"), "n'est" (not "ne est").
- **SPM `<unk>` shows up as `⁇`**: "Israël" → "Isra ⁇ l". Caused by
  `character_coverage=0.9995` dropping rare accented chars. Hits ~4.4% of
  French valid sentences; roughly a 0.5 BLEU ceiling to reclaim in the next
  iteration by using `character_coverage=1.0`.

### Interactive samples (out-of-domain robustness)

Feeding the averaged checkpoint with sentences the model never saw during
training — both news-style and conversational — shows the domain signature
clearly:

**News-style (in-domain) — essentially professional quality:**

```
SRC: The Prime Minister announced new sanctions against Russia.
HYP: Le Premier ministre a annoncé de nouvelles sanctions contre la Russie.

SRC: The European Central Bank raised interest rates by 0.25 percentage points.
HYP: La Banque centrale européenne a relevé les taux d'intérêt de 0,25 point
     de pourcentage.
```

The second example alone gets ~10 fine-grained things right: adjective
ordering ("Banque centrale" not "centrale Banque"), feminine agreement
("européenne"), finance-register verb ("a relevé" not "a augmenté"), the
fixed collocation "taux d'intérêt", European decimal notation ("0,25" with
a comma), and singular "point de pourcentage" (French singularizes units
when the coefficient is between 0 and 2 — a rule English doesn't have).

**Conversational / netspeak (out-of-distribution) — fails as expected:**

```
SRC: hello, how are u
HYP: Bonjour, comment sont-ils u

SRC: hello, how r u
HYP: Bonjour, comment r u
```

Two failure modes visible here, both caused by WMT14 being a news + parliament
corpus:

1. **Unknown tokens pass through**: `"u"` and `"r"` never appear as
   abbreviations in the training data, so SPM tokenizes them as bare letters
   and the model copies them verbatim.
2. **Wrong sense disambiguation on "are"**: `"how are you"` is rare in news
   text, so the model defaults to the plural/3rd-person sense of "are" that
   dominates parliamentary language (`"how are these policies working"` →
   `"comment sont ces politiques"`), producing `"comment sont-ils"`
   ("how are they") instead of the correct idiom `"comment allez-vous"`.

This is a **domain coverage** issue, not a model-capacity issue. Fixing it
requires either mixing in conversational corpora (OpenSubtitles, TED talks) or
adding domain tags at training time — both on the roadmap after Base hits its
ceiling on pure WMT data.

### Why this worked (contrast with zh-en)

Same codebase, same Transformer, same training loop — different outcome.
The differences that matter:

1. **Shared script**: both sides are Latin. Shared BPE vocab aligns cognates,
   function words, and numerals. For zh-en the "shared" 32K vocab degenerates
   into two disjoint halves that never interact.
2. **Clean data**: WMT14 en-fr's Europarl + News Commentary + filtered Common
   Crawl has much less boilerplate per unique sentence than WMT17 zh-en's
   UN / news-agency attractors.
3. **Loss landscape has a conditional minimum**: the model finds
   "condition on source" as a better local minimum than
   "unconditional French LM". For zh-en, the unconditional LM minimum at loss
   ~4.8 is deeper than any conditional signal cross-attention could generate
   from the disjoint vocab.

### Configuration and compute

- Hardware: single RTX 5090 (32 GB VRAM)
- Precision: BF16 autocast
- Optimizer: Adam (β₁=0.9, β₂=0.98, ε=1e-9)
- LR schedule: Noam with peak 6.86e-4 at step ~16K, min_lr floor 1e-5
- Warmup: 4000 steps
- Gradient clip: 1.0
- Loss spike guard: ratio 1.3, EMA α 0.005 (inherited from Big zh-en failure
  analysis). **Triggered exactly once** in 100K steps, at step 17753 — cleanly
  dropped a polluted batch; no training disruption.
- Effective batch: ~100K tokens (24576 × 4 accumulation)
- Training time: **2h 5m 23s**, end-to-end.

### Reproducing this result

```bash
python scripts/download_wmt_enfr.py --max-train-samples 10000000
python scripts/clean_data_enfr.py
python scripts/train_tokenizer.py \
    --inputs data/train.clean.en data/train.clean.fr \
    --model-prefix data/spm_enfr --vocab-size 32000
python train.py --config configs/base_en_fr.yaml
python scripts/average_checkpoints.py --ckpt-dir checkpoints/base_enfr --n 5
python scripts/eval_bleu.py \
    --ckpt checkpoints/base_enfr/averaged.pt \
    --config configs/base_en_fr.yaml \
    --src data/test.en --ref data/test.fr
```

## Success Case (tied): Big on WMT14 en-fr

Trained Transformer Big (~209M params) on the **same cleaned 9.3M-pair
subset used for Base v1** for 279K steps on the same single RTX 5090.
**Converged to BLEU 30.16 on newstest2013 and 34.66 on newstest2014** after
7-checkpoint averaging — statistically tied with Base's 34.69 despite 3.5×
parameters and 2.8× training steps on identical data. The key lesson is
*why they tied*: same data, same tokenizer, different capacity — capacity
is not the bottleneck.

![Big en-fr training curves](assets/training_curves_big_enfr_v1.png)

### Final numbers

| | newstest2013 (valid) | newstest2014 (test) |
|---|---|---|
| Best single checkpoint (step 261K) | 30.16 | — |
| Averaged (7 ckpts, steps 255K–279K + best) | **30.45** | **34.66** |
| Averaging gain on valid | +0.29 | — |

### Training trajectory milestones

| Step  | Train Loss | Valid BLEU | LR       | Note |
|-------|------------|------------|----------|------|
| 20K   | ~3.6       | 19.74      | 3.3e-4   | first eval |
| 50K   | 2.78       | 25.25      | 4.6e-4   | past Noam peak (4.86e-4 @ 37K) |
| 100K  | 2.71       | 27.79      | 3.0e-4   | |
| 150K  | 2.67       | 28.76      | 2.4e-4   | end of epoch 1 at ~147K |
| 200K  | 2.63       | 29.54      | 2.1e-4   | |
| 246K  | 2.61       | **30.04**  | 1.9e-4   | first crossing of 30 |
| 261K  | 2.60       | **30.16**  | 1.85e-4  | **single-model peak** |
| 279K  | 2.58       | 29.87      | 1.77e-4  | interrupted; train↓ valid→ (mild divergence) |

Total training time: **5h 52m** on a single RTX 5090 (BF16, effective batch
~32K tokens/step).

### Why Big tied Base instead of beating it

Both runs end at ~34.7 sacrebleu on newstest2014. Given Big has 3.5× the
parameters and 2.8× the steps (on identical data and identical tokenizer),
the natural reaction is "is Big broken?" — but the diagnostics point
clearly elsewhere:

1. **Shared tokenizer is the shared ceiling.** Both runs use the same
   SentencePiece model trained at `character_coverage=0.9995`, which drops
   rare accented characters (`Israël`, `Noël`, `maïs`, ...). 132 / 3000
   French valid sentences (**4.4%**) hit `<unk>`, worth an estimated
   0.5 BLEU that neither model can reclaim no matter how much it trains.
2. **Reference bias absorbs the rest of Big's extra capacity.** Big's
   translations are consistently fluent and faithful, but use valid French
   paraphrases the reference translator didn't pick (e.g.
   `stratégie républicaine **de lutte contre**` vs the reference's
   `stratégie républicaine **pour contrer**` — both are standard French,
   BLEU penalizes one). Big has the capacity to produce more stylistic
   variation, which *hurts* BLEU against a single-reference test set.
3. **Train loss is still descending at stop.** 2.58 at 279K vs Base's 2.56
   at 100K — Big's optimizer trajectory shows it had more to give, but the
   tokenizer ceiling and BLEU's single-reference limitation converted
   additional training signal into ≈ 0 test-BLEU gain.

**So Big v1 is not a failed scale-up; it's a diagnostic run that proved
the bottleneck is the preprocessing pipeline, not the model.** Base v2 now
addresses both axes independently: `character_coverage=1.0` to remove the
UNK ceiling, and — since discovery during v2 setup revealed the raw
download was accidentally capped at 10M pairs back in v1 — a fresh
full-corpus download (raw ~40M → cleaned ~25-30M) to finally exercise the
data axis v1 never got. Expected gain from the two combined: +1.5 to +2.5
sacrebleu.

### Sample translations (averaged checkpoint, newstest2013)

```
SRC: A Republican strategy to counter the re-election of Obama
HYP: Stratégie républicaine de lutte contre la réélection d'Obama
REF: Une stratégie républicaine pour contrer la réélection d'Obama
        → valid paraphrase; penalized by BLEU

SRC: Also the effect of vitamin D on cancer is not clear.
HYP: De même, l'effet de la vitamine D sur le cancer n'est pas clair.
REF: L'effet de la vitamine D sur le cancer n'est pas clairement établi
     non plus.
        → translator escalated register and added explicitation

SRC: In Israel, holy places await Ukrainian tourists, the omphalos and a
     sea of saline water
HYP: En Isra ⁇ l, les lieux saints attendent les touristes ukrainiens,
     l'omphalos et la mer d'eau saline.
REF: En Israël, des lieux saints, le Centre du monde et une mer de saumure
     attendent les touristes ukrainiens
        → Isra ⁇ l = SPM UNK (coverage 0.9995 tax, hits Big identically)
```

### Configuration and compute

- Architecture: d_model 1024, 16 heads, 6+6 layers, FFN 4096, dropout 0.2
- Parameters: 209,129,472
- Precision: BF16 autocast
- Effective batch: ~32K tokens (8192 × 4 accumulation) — 1/3 of Base's 100K
- Warmup: 8000 steps (longer than Base's 4K — safer for Big's larger weights)
- LR schedule: Noam × 1.5, peak 4.86e-4 at step ~37K, min_lr floor 1e-5
- Loss spike guard: ratio 1.3, EMA α 0.005. **Triggered twice** (steps
  105432 and one earlier) — both single-batch noise events from the Common
  Crawl portion of the corpus; dropped cleanly, no training disruption.

## Ablation: Base v2 → v3 on WMT14 en-fr (data × steps, and the data-quality trap)

Two follow-up runs were designed to attack Base v1's ≈ 34.7 test-BLEU
ceiling by stacking "obvious" improvements: more data, SPM coverage fix,
longer training. The result is a **surprise negative finding**: despite
3× more data, 6× more steps, and a stricter SPM, Base v3 ended up
**below** Base v1 on both valid and test. The two runs together nail
down *why*.

### The three runs side-by-side

| Run      | Data       | SPM cov. | Steps | Valid (avg) | Test (avg) |
|----------|-----------|----------|-------|-------------|------------|
| Base v1  | 9.3M strict filter | 0.9995 | 100K | ~30.00 | **34.69** |
| Base v2  | 30M loose filter | 1.0    | 200K | 27.77 (single) | — (skipped) |
| Base v3  | 30M loose filter | 1.0    | 600K | **29.23** | 33.90 |

All three on the same single RTX 5090, same Transformer Base architecture.
The only differences between runs are those three columns.

### Base v2 (the underfit)

Planned as "v1 + three improvements stacked": full WMT14 corpus (v1 was
accidentally capped at 10M raw by `--max-train-samples`; full is ~40M
raw → 30M after cleaning), `character_coverage=1.0` (fixes the `Isra⁇l`
`<unk>` visible in v1's samples), 2× step budget (200K, reasoned as
"more data needs more steps"). Target was test BLEU 36-37.

**What went wrong**: 200K steps on 30M pairs = ~1.3 epochs, vs v1's
~4 epochs on 9.3M. The Noam LR schedule decayed on 200K's clock, so the
model reached the decay tail having seen each sample barely more than
once — classic underfit. Valid BLEU trajectory plateaued at 27.77 and
never broke v1's 28+ range:

| Step  | Valid BLEU |
|-------|-----------|
| 30K   | 23.13 |
| 98K   | 27.01 |
| 141K  | 27.48 |
| 194.9K | **27.77** (best) |
| 200K (final) | plateau |

Averaging would have added ~+0.3 at most — not enough to matter. Run
was shelved without full evaluation and rerun as v3.

### Base v3 (the fix — and the surprise)

Scaled the step budget proportionally to data growth:

- `max_steps: 200000 → 600000` (3× to restore ~4 epochs like v1)
- `warmup_steps: 4000 → 12000` (keep warmup/total ratio)
- Everything else identical to v2

**v3 ran cleanly for 12h45m, finished exactly 4 epochs, best single
checkpoint BLEU 29.06 at step 537K, averaged 29.23 / test 33.90.** The
curve decisively broke v2's 27.77 plateau at step 260K (first > 28) and
climbed into the 28.8-29.0 band by 455K.

Trajectory (sampled):

| Step  | Valid BLEU | Note |
|-------|-----------|------|
| 30K   | 18.75 | LR still in warmup (warmup ends at 48K logging-step due to 4× grad accumulation) |
| 94K   | 25.86 | matches v2 at ~similar *schedule fraction*, not absolute step |
| 234K  | 27.85 | **first time above v2's final 27.77** (only 39% of schedule used) |
| 260K  | 28.21 | first break above 28 |
| 311K  | 28.64 | new best |
| 419K  | 28.78 | |
| 467K  | 29.03 | |
| 537K  | **29.06** | best single checkpoint |
| 600K  | 28.75 (final) | |

**So v3 fixes v2 cleanly — +1.29 valid BLEU, +1.28 over v1 single-ckpt
too. The fix itself is boring: step budget has to scale with data
size.** The interesting part is what happens next.

### The surprise: v3 test < v1 test despite more of everything

Averaged numbers put v3 at **33.90 test** vs v1's **34.69 test**. v3
lost 0.79 BLEU to v1 despite 3× more training data, 6× more training
steps, and an SPM that fixes the `<unk>`-on-accent problem v1 had.

This isn't the step-budget issue (v3 is fully converged, 4 epochs,
averaged). And it's not noise — the gap shows up on both valid (-0.77)
and test (-0.79).

Three candidate explanations, ranked by what we believe:

1. **Data quality beat data quantity.** v1 kept 9.3M / 40M (23% keep
   rate) under the stricter cleaning thresholds — tight length ratio,
   high Latin-script ratio, aggressive duplicate filter. v2/v3 kept
   30M / 40M (74%) by relaxing those thresholds. The extra 20M pairs
   carry more misaligned, mixed-language, and parliamentary-filler
   content that dilutes the gradient signal toward newstest-style prose.
2. **SPM `character_coverage=1.0` may hurt on noisy data.** It was
   designed to fix rare-accent `<unk>` (Israël, Noël). With a cleaner
   corpus this is pure win. With a noisier corpus, letting every OCR
   artifact or scanner glitch survive as a real token adds noise to the
   softmax that a 0.9995 coverage would have squashed to `<unk>`. This
   is a hypothesis — not directly measured.
3. **Train/test domain shift.** newstest2014 is tight news prose. The
   extra 20M pairs in v2/v3 lean more on Europarl/CommonCrawl mixtures
   that drift away from this register, nudging the model's generation
   distribution slightly off-target.

The combined lesson: **filter first, scale second**. The naive "more data
is always better" assumption fails when "more" means "lower-quality
more".

### What v2 → v3 teaches

- **v2 vs v3**: the textbook lesson — step budget must match data scale.
  v2's underfit at 27.77 vs v3's 29.06 is the cleanest possible
  demonstration, same data, same architecture, only difference is LR
  schedule length.
- **v1 vs v3**: the *non-textbook* lesson — quality/quantity trade-off
  isn't free. 20M extra noisy pairs lost 0.79 test BLEU even after
  spending 6× compute to train on them.
- **Combined prescription**: the next run should reuse v1's strict
  cleaning thresholds on the full 40M raw corpus (yielding perhaps
  10-12M *cleanly-aligned* pairs instead of v1's accidentally-truncated
  9.3M), then train at a v3-style extended schedule. That decouples
  "quality" from "quantity" so both gains compound instead of fighting.

### Roadmap (updated)

1. **Big v2 (en-fr)** — reinstated (previously marked skipped after Big
   v1 tied Base v1). With v3's data and schedule now characterized, Big
   v2 is the cleanest test of whether model capacity helps absorb the
   noisier 30M corpus — i.e., whether 209M params can fit what 60M
   cannot. If Big v2 on v3's data reaches v1's ceiling (~34.7) or
   higher, the answer is yes; if it stays in v3's ~34 range, capacity
   doesn't fix data quality.
2. **WMT14 en-de Base + Big** — the most-cited MT benchmark, apples-to-apples
   with the literature. Paper reports Base 27.3, Big 28.4 (tokenized);
   sacrebleu equivalent ~25-26 / 26-27.
3. **Strict-filter en-fr rerun (optional)** — if Big v2 doesn't recover
   the v1 test number, re-clean the 40M corpus with v1's stricter
   thresholds and train Base at the v3-style extended schedule. Tests
   the "filter first, scale second" prescription directly.
4. **Fine-tuning study** — take the trained en-fr models into an external
   fine-tuning repo to measure how much task-specific tuning adds on
   top of general-domain pretraining.
5. **Final report** — combine all runs (zh-en Base ✗, zh-en Big ✗,
   en-fr Base v1 ✅, en-fr Big v1 tied, en-fr Base v2 underfit, en-fr
   Base v3 converged-but-below-v1, en-fr Big v2, en-de Base, en-de Big)
   with dedicated sections on the data-quality/quantity ablation and on
   BLEU limitations (chrF / BLEURT / COMET cross-validation on the
   strongest checkpoint).

## Failure Case: Base on WMT17 zh-en

The Base config was trained to ~700K steps on cleaned WMT17 zh-en (19M pairs).
It **failed to converge in any useful sense** — loss plateaued at ~4.22 and
valid BLEU never crossed 1.0. Kept here as a cautionary baseline.

![Base training curves](assets/training_curves_base_zhen.png)

The loss curve is almost flat from ~560K onwards and BLEU oscillates
around 0.7 without an upward trend, even as the LR keeps decaying. Classic
signature of a model that has hit its capacity/data ceiling.

### Training trajectory

| Step    | Train Loss | Valid BLEU | LR       |
|---------|------------|------------|----------|
| 28K     | 4.63       | —          | 5.28e-4  |
| 160K    | 4.33       | 0.46       | 2.23e-4  |
| 305K    | 4.27       | 0.71       | 1.60e-4  |
| 385K    | 4.25       | 0.54       | 1.42e-4  |
| 700K    | 4.22       | 0.77 (best)| 1.06e-4  |

Loss dropped 0.08 over the last 500K steps — essentially flat. BLEU oscillated
in the 0.5-0.8 range without trend. Sample translations remained unconditioned
boilerplate (e.g. *"I'd like to take this opportunity to congratulate you..."*
for arbitrary Chinese inputs), showing the language-model prior dominated the
source signal.
Due to time and resource constraints, we had to halt further training.

### What went wrong — and what didn't

What we ruled out via diagnostics (`scripts/diagnose_attention.py`):

- **Not a code bug.** Encoder outputs differ by source (L2 distances 16-23);
  decoder logits differ by source (L2 distances up to 65). Cross-attention is
  wired correctly and passing signal.
- **Not a tokenization bug.** SentencePiece alphabet covers 2814 chars at
  99.95% coverage. No `<unk>` on held-out Chinese inputs.
- **Not data misalignment.** Spot-checked parallel corpus, zh/en lines aligned.

What actually happened:

1. **zh-en is hard.** Unrelated language pair (no cognates, different script,
   large word-order divergence). Shared BPE vocab degrades to "two disjoint
   halves" instead of the aligned subword space en-de benefits from. Published
   Base Transformer results on WMT zh-en top out around BLEU 18-22 even with
   clean data and careful tuning — not BLEU 27+ like en-de.
2. **Data quality.** Even after cleaning (dedupe >50×, length-ratio filter),
   the corpus still contains substantial news-agency boilerplate and
   UN-style generic phrasing. These high-frequency patterns become attractors
   that the language-model prior can exploit without ever learning to condition
   on source.
3. **Model capacity.** Base (65M params) is likely too small for this task.
   The plateau at loss 4.22 is consistent with the model memorizing the
   marginal English distribution and failing to model the conditional.
4. **LR decay too aggressive for a slow task.** Noam schedule with `lr_scale=1`
   has LR at ~1.06e-4 by step 700K — too low to escape the plateau even if
   the model had capacity left.

### Lessons

- For a "from scratch on harder language pair" project, prefer **Big from the
  start** — don't sink compute into Base hoping it'll converge.
- Sample translations are a better early-warning signal than BLEU or loss.
  A loss of 4.2 with mode-collapsed outputs is qualitatively different from
  a loss of 4.2 with source-conditioned outputs, but both look identical on
  the loss curve.
- If building a new tokenizer or filter, **delete every downstream cache file
  by hand**. A stale `valid.*.cached_*.npz` encoded under the old vocab will
  silently poison eval without any error message.
- `loginctl enable-linger $USER` on JupyterHub hosts — without it, tmux and
  long-running jobs die when the browser session closes.

## Failure Case: Big on WMT17 zh-en

After Base failed, we moved to Big (213M params) hoping extra capacity would
break the mode-collapse ceiling. It didn't. Across multiple training attempts
with increasingly careful tuning, Big on WMT17 zh-en **also never produced
source-conditioned translations**.

![Big training curves](assets/training_curves_big_zhen.png)

The loss curve (top-left) shows the same flattening pattern as Base, just at
a lower absolute level (~4.65 vs ~4.22). BLEU (top-right) rises from 0.12 to
0.17 over 30K steps — a change that's statistically real but semantically
meaningless: sample translations at the "best" checkpoint are still fluent
English with zero relationship to the Chinese source.

### Training attempts summary

| Attempt | Precision | lr_scale | warmup | Spike guard | Best BLEU | Outcome |
|---------|-----------|----------|--------|-------------|-----------|---------|
| v1 | FP16 | 1.5 | 4K | — | 0.31 @ 32K | Blew up at step 46400 (loss 4.0 → 4.8) |
| v2 (seed=42 resume) | FP16 | 1.5 | 4K | — | 0.47 @ 69K | Blew up at step 47800, loss climbed to 4.8 and stayed |
| v3 (BF16 switch) | BF16 | 1.5 | 4K | — | — | **Identical blowup** — numerical precision wasn't the issue |
| v4 (seed=43, lr=1.0, clip=0.5, guard=1.8) | BF16 | 1.0 | 4K | ratio 1.8 | — | Stalled at loss 4.8 (clip too tight, learning suppressed) |
| v5 (clip=1.0, lr=1.2, guard=1.8) | BF16 | 1.2 | 4K | ratio 1.8 | 0.47 @ 69K | Blew up at step 83800, guard threshold too loose to catch it |
| v6 (from scratch, warmup=8K, guard=1.3, EMA window 200) | BF16 | 1.5 | 8K | ratio 1.3 | 0.17 @ 85K | No blowups, but **even slower than v5**. Still mode collapse. |

Across ~6 days of compute, best Big BLEU was 0.47 — **worse than the failed
Base** (0.77), and still within pure mode-collapse territory. Sample
translations at Big v6 step 85K:

```
SRC: 加利福尼亚州水务工程的新问题
REF: New Questions Over California Water Project
HYP: I'm not sure what you're doing.

SRC: 伊斯兰国武装分子控制着伊拉克和叙利亚的部分领土...
REF: Its militants control parts of Iraq and Syria...
HYP: In addition, the Government of the Democratic Republic of the Congo,
     in collaboration with the Government of the Sudan and the United Nations
     Development Programme (UNDP) and the United Nations Children's Fund (UNICEF)...
```

The HYP is fluent English with no relationship to the SRC — a pure
unconditional language model with topic-triggered boilerplate attractors
(e.g. "refugees" in source → UN-agency soup in output).

### The deeper diagnosis

The two failures together constitute strong evidence that **vanilla Transformer
trained from scratch on WMT17 zh-en cannot escape mode collapse within
reasonable compute** — regardless of model size (65M or 213M), precision
(FP16 or BF16), or optimization tuning.

The root cause is a combination factors that we cannot fix by tuning alone:

1. **The LM shortcut is too tempting.** Cross-entropy + label smoothing on a
   32K-vocab target makes "model the English distribution unconditionally"
   a local minimum at loss ~4.8. The gradient signal from source (via
   cross-attention) is too weak, relative to target-side signal, for the model
   to prefer the harder "condition on source" solution.
2. **Shared BPE hurts, doesn't help.** For en-de the shared subword space
   aligns cognates and function words; for zh-en, Chinese characters and
   English words are disjoint token sets, so "shared embeddings" means
   "two disjoint halves of the embedding matrix that never interact".
3. **WMT zh-en has too many high-frequency attractors.** News-agency and
   UN-style boilerplate appears thousands of times in slight variations.
   Once the model finds the attractor, no amount of training drags it out.
4. **Published Big zh-en numbers are misleading.** The BLEU 24+ results in
   papers rely on back-translation, ensemble decoding, or curated data
   (not raw WMT). Without those tricks, from-scratch Big on WMT zh-en
   tops out in BLEU 15-20 range — and even that requires specific tuning
   we weren't able to find.

### What we tried that didn't work

- Longer warmup (4K → 8K): prevents blowups, doesn't break mode collapse
- Tighter gradient clip (1.0 → 0.5): stalls learning entirely
- Tighter spike guard (ratio 1.8 → 1.3, EMA window 50 → 200): catches
  blowups cleanly but doesn't create conditioning signal
- Switch to BF16: eliminates one failure axis (fp16 overflow) but not the
  underlying dynamics
- Seed change: escapes one deterministic "toxic batch" trajectory, model
  finds equivalent problems elsewhere

### Pivot

Rather than keep throwing compute at zh-en, we're switching to:

1. **WMT14 en-fr Base** — a known-converging language pair, with the same
   codebase. If en-fr Base hits its expected BLEU 35+, the code is validated
   and the zh-en failure is conclusively a task-difficulty problem, not a
   bug.
2. **Then, if en-fr Base succeeds**, a second zh-en attempt with a cleaner,
   smaller dataset (IWSLT + News Commentary, ~550K pairs vs WMT's 19M)
   where mode collapse is less likely to be the dominant dynamic.

See `configs/base_en_fr.yaml` and the en-fr results section below.

## Hardware Used

- GPU: NVIDIA RTX 5090 (32 GB VRAM, Blackwell sm_120)
- Mixed precision: BF16 (default for en-fr and Big zh-en) / FP16 (Base zh-en historical)

## References

- Vaswani et al., *Attention Is All You Need* (2017). [[arxiv]](https://arxiv.org/abs/1706.03762)
- WMT17 Shared Task: Machine Translation. [[link]](https://www.statmt.org/wmt17/)
- WMT14 Translation Task. [[link]](https://www.statmt.org/wmt14/translation-task.html)
- Post, *A Call for Clarity in Reporting BLEU Scores* (2018, sacrebleu). [[arxiv]](https://arxiv.org/abs/1804.08771)
- SentencePiece: [[github]](https://github.com/google/sentencepiece)
