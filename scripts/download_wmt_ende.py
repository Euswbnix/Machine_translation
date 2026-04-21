"""Download and prepare WMT14 en-de parallel corpus for training.

Same shape as download_wmt_enfr.py, adapted for en-de. WMT14 en-de is the
most-cited MT benchmark — Vaswani et al. report Base 27.3 / Big 28.4
(tokenized BLEU), sacrebleu equivalent ~25-26 / 26-27.

Standard valid/test splits: newstest2013 / newstest2014.

WMT14 de-en raw corpus is ~4.5M pairs (much smaller than en-fr's ~40M) —
mostly Europarl + News Commentary + Common Crawl. No cap needed by default.

Usage:
    python scripts/download_wmt_ende.py
    python scripts/download_wmt_ende.py --max-train-samples 4000000
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def download_wmt(output_dir: str, max_train_samples: int | None = None):
    """Download WMT14 de-en data and save as parallel text files.

    Note: HuggingFace's wmt14 config is "de-en". Source and target are just
    two keys in the translation dict, direction is our choice. We train
    en→de (en is source, de is target).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading WMT14 de-en dataset...")
    dataset = load_dataset("wmt14", "de-en")

    # Train
    print("Processing training data...")
    train_data = dataset["train"]
    if max_train_samples:
        train_data = train_data.select(range(min(max_train_samples, len(train_data))))
    _save_split(train_data, output_dir / "train.en", output_dir / "train.de")

    # Valid = newstest2013
    print("Processing validation data (newstest2013)...")
    _save_split(dataset["validation"], output_dir / "valid.en", output_dir / "valid.de")

    # Test = newstest2014
    print("Processing test data (newstest2014)...")
    _save_split(dataset["test"], output_dir / "test.en", output_dir / "test.de")

    print(f"\nData saved to {output_dir}")
    for split in ["train", "valid", "test"]:
        n = sum(1 for _ in open(output_dir / f"{split}.en"))
        print(f"  {split}: {n:,} sentence pairs")


def _save_split(data, src_path: Path, tgt_path: Path):
    """Save a dataset split to parallel text files (en source, de target)."""
    with open(src_path, "w", encoding="utf-8") as f_src, \
         open(tgt_path, "w", encoding="utf-8") as f_tgt:
        for example in tqdm(data, desc=f"Writing {src_path.stem}"):
            translation = example["translation"]
            en = translation["en"].strip().replace("\n", " ")
            de = translation["de"].strip().replace("\n", " ")
            if en and de:
                f_src.write(en + "\n")
                f_tgt.write(de + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download WMT14 en-de data")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Max training samples (default: all ~4.5M).",
    )
    args = parser.parse_args()
    download_wmt(args.output_dir, args.max_train_samples)


if __name__ == "__main__":
    main()
