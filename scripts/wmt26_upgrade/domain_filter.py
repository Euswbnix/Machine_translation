"""Domain-conditional QE filtering for WMT26 upgrade.

Two-stage pipeline:
  1. Train a fastText domain classifier on labeled WMT14 subcorpora
     (Europarl, News Commentary, Common Crawl, UN, Giga → 5 classes)
  2. Apply classifier to the CometKiwi-scored v2 TSV, keep only
     news-domain pairs, then take top-K by QE score.

This addresses the negative FT result from the paper: naive top-K-by-QE
skews toward UN/legislative; domain-conditional filtering keeps
news-domain pairs that are also high-quality.

Usage:
    # Step 1: build domain classifier
    python scripts/wmt26_upgrade/domain_filter.py train-classifier \
        --data-dir data/ \
        --model-out models/domain_classifier.bin

    # Step 2: filter scored TSV
    python scripts/wmt26_upgrade/domain_filter.py filter \
        --scored-tsv data/v2_scored.tsv \
        --classifier models/domain_classifier.bin \
        --target-domain news \
        --top-k 1000000 \
        --src-out data/sft_news_train.en \
        --tgt-out data/sft_news_train.fr
"""

import argparse
import sys
import tempfile
from pathlib import Path


def train_classifier(args):
    """Train a fastText domain classifier on WMT14 subcorpora."""
    import fasttext

    print("Building training data for domain classifier...", file=sys.stderr)

    train_lines = []
    label_counts = {}

    domain_files = {
        "news": ["commoncrawl", "news-commentary", "news_commentary"],
        "europarl": ["europarl"],
        "un": ["un", "undoc", "giga-fren"],
    }

    data_dir = Path(args.data_dir)
    for label, patterns in domain_files.items():
        count = 0
        for src_file in data_dir.glob("*.en"):
            fname = src_file.stem.lower()
            if any(p in fname for p in patterns):
                with open(src_file, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if len(line) > 20:
                            train_lines.append(f"__label__{label} {line}")
                            count += 1
                            if count >= 500000:
                                break
                if count >= 500000:
                    break
        label_counts[label] = count
        print(f"  {label}: {count:,} lines", file=sys.stderr)

    if not train_lines:
        print("No domain-labeled data found. Falling back to heuristic labeling.",
              file=sys.stderr)
        print("Place subcorpus files in data/ with names containing "
              "'europarl', 'news', 'un', or 'giga'.", file=sys.stderr)
        sys.exit(1)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write("\n".join(train_lines))
        train_path = tmp.name

    print(f"Training fastText classifier on {len(train_lines):,} lines...",
          file=sys.stderr)
    model = fasttext.train_supervised(
        train_path,
        epoch=10,
        lr=0.5,
        wordNgrams=2,
        dim=100,
        loss="softmax",
    )

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model_out)
    print(f"Saved classifier to {args.model_out}", file=sys.stderr)

    result = model.test(train_path)
    print(f"Train accuracy: {result[1]:.3f} (P@1), {len(train_lines):,} samples",
          file=sys.stderr)

    Path(train_path).unlink()


def filter_scored(args):
    """Filter scored TSV: keep target-domain pairs, take top-K by QE."""
    import fasttext

    print(f"Loading classifier from {args.classifier}...", file=sys.stderr)
    model = fasttext.load_model(args.classifier)

    target_label = f"__label__{args.target_domain}"

    print(f"Reading {args.scored_tsv} and filtering for {target_label}...",
          file=sys.stderr)

    domain_hits = []
    total = 0
    with open(args.scored_tsv, encoding="utf-8") as f:
        for line in f:
            total += 1
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            try:
                score = float(parts[0])
            except ValueError:
                continue
            src, tgt = parts[1], parts[2]

            pred = model.predict(src, k=1)
            pred_label = pred[0][0]
            pred_conf = pred[1][0]

            if pred_label == target_label and pred_conf >= args.min_confidence:
                domain_hits.append((score, src, tgt))

            if total % 1000000 == 0:
                print(f"  processed {total:,}, domain hits: {len(domain_hits):,}",
                      file=sys.stderr)

    print(f"\nTotal: {total:,}, domain hits: {len(domain_hits):,} "
          f"({100*len(domain_hits)/max(total,1):.1f}%)", file=sys.stderr)

    domain_hits.sort(key=lambda r: r[0], reverse=True)
    domain_hits = domain_hits[:args.top_k]

    seen = set()
    deduped = []
    for score, src, tgt in domain_hits:
        key = (src, tgt)
        if key not in seen:
            seen.add(key)
            deduped.append((score, src, tgt))
    domain_hits = deduped

    print(f"After top-{args.top_k} + dedup: {len(domain_hits):,} pairs "
          f"(score range {domain_hits[-1][0]:.4f} - {domain_hits[0][0]:.4f})",
          file=sys.stderr)

    with open(args.src_out, "w", encoding="utf-8") as fs, \
         open(args.tgt_out, "w", encoding="utf-8") as ft:
        for _, s, t in domain_hits:
            fs.write(s + "\n")
            ft.write(t + "\n")

    print(f"Wrote {args.src_out} and {args.tgt_out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    tc = sub.add_parser("train-classifier")
    tc.add_argument("--data-dir", required=True)
    tc.add_argument("--model-out", default="models/domain_classifier.bin")

    fl = sub.add_parser("filter")
    fl.add_argument("--scored-tsv", required=True)
    fl.add_argument("--classifier", required=True)
    fl.add_argument("--target-domain", default="news")
    fl.add_argument("--min-confidence", type=float, default=0.5)
    fl.add_argument("--top-k", type=int, default=1000000)
    fl.add_argument("--src-out", required=True)
    fl.add_argument("--tgt-out", required=True)

    args = ap.parse_args()
    if args.command == "train-classifier":
        train_classifier(args)
    else:
        filter_scored(args)


if __name__ == "__main__":
    main()
