"""Download and prepare WMT17 en-zh parallel corpus.

WMT17 en-zh is the standard non-European MT benchmark:
  - ~25M raw pairs (UN, News Commentary, CWMT)
  - newsdev2017 (valid) / newstest2017 (test)
  - Morphologically very different from en-fr/en-de: no shared script,
    word segmentation needed, different optimal BPE settings.

This provides the third language pair for the WMT26 upgrade, testing
whether the data-quality/capacity sign-flip generalizes beyond European
languages.

Usage:
    python scripts/wmt26_upgrade/download_wmt17_enzh.py
    python scripts/wmt26_upgrade/download_wmt17_enzh.py --max-train-samples 10000000
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def download_wmt(output_dir: str, max_train_samples: int | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading WMT17 zh-en dataset...")
    dataset = load_dataset("wmt17", "zh-en")

    # Train
    print("Processing training data...")
    train_data = dataset["train"]
    n = min(len(train_data), max_train_samples) if max_train_samples else len(train_data)

    with open(output_dir / "train.en", "w", encoding="utf-8") as f_en, \
         open(output_dir / "train.zh", "w", encoding="utf-8") as f_zh:
        for i in tqdm(range(n), desc="train"):
            pair = train_data[i]["translation"]
            f_en.write(pair["en"].strip() + "\n")
            f_zh.write(pair["zh"].strip() + "\n")
    print(f"  Wrote {n:,} training pairs")

    # Valid (newsdev2017)
    print("Processing validation data...")
    valid_data = dataset["validation"]
    with open(output_dir / "valid.en", "w", encoding="utf-8") as f_en, \
         open(output_dir / "valid.zh", "w", encoding="utf-8") as f_zh:
        for item in tqdm(valid_data, desc="valid"):
            pair = item["translation"]
            f_en.write(pair["en"].strip() + "\n")
            f_zh.write(pair["zh"].strip() + "\n")
    print(f"  Wrote {len(valid_data):,} validation pairs")

    # Test (newstest2017)
    print("Processing test data...")
    test_data = dataset["test"]
    with open(output_dir / "test.en", "w", encoding="utf-8") as f_en, \
         open(output_dir / "test.zh", "w", encoding="utf-8") as f_zh:
        for item in tqdm(test_data, desc="test"):
            pair = item["translation"]
            f_en.write(pair["en"].strip() + "\n")
            f_zh.write(pair["zh"].strip() + "\n")
    print(f"  Wrote {len(test_data):,} test pairs")

    print(f"\nAll data saved to {output_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data_enzh", help="Output directory")
    ap.add_argument("--max-train-samples", type=int, default=None)
    args = ap.parse_args()
    download_wmt(args.output_dir, args.max_train_samples)


if __name__ == "__main__":
    main()
