"""Download and prepare WMT14 en-fr parallel corpus for training.

Same shape as download_data.py (zh-en), adapted for en-fr. We go with WMT14
because:
  - Standard newstest2013 (valid) / newstest2014 (test) benchmarks exist
  - en-fr shares a script → BPE is well-behaved (unlike zh-en)
  - Clean comparable benchmarks in the literature (Transformer Base ≈ 27 BLEU)

WMT14 fr-en on HuggingFace includes Europarl + News Commentary + Common Crawl +
Giga + UN. The raw corpus is ~40M pairs — most of it is noisy Common Crawl and
Giga. We filter to a cleaner subset by default (length + length-ratio filters
applied downstream by prepare_data.py, but we can also cap here).

Usage:
    python scripts/download_wmt_enfr.py
    python scripts/download_wmt_enfr.py --max-train-samples 5000000
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def download_wmt(output_dir: str, max_train_samples: int | None = None):
    """Download WMT14 fr-en data and save as parallel text files.

    Note: HuggingFace's wmt14 config is "fr-en" (not "en-fr"). Source and target
    are just two keys in the translation dict, direction is our choice. We train
    en→fr (en is source, fr is target)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading WMT14 fr-en dataset...")
    dataset = load_dataset("wmt14", "fr-en")

    # Train
    print("Processing training data...")
    train_data = dataset["train"]
    if max_train_samples:
        train_data = train_data.select(range(min(max_train_samples, len(train_data))))
    _save_split(train_data, output_dir / "train.en", output_dir / "train.fr")

    # Valid = newstest2013
    print("Processing validation data (newstest2013)...")
    _save_split(dataset["validation"], output_dir / "valid.en", output_dir / "valid.fr")

    # Test = newstest2014
    print("Processing test data (newstest2014)...")
    _save_split(dataset["test"], output_dir / "test.en", output_dir / "test.fr")

    print(f"\nData saved to {output_dir}")
    for split in ["train", "valid", "test"]:
        n = sum(1 for _ in open(output_dir / f"{split}.en"))
        print(f"  {split}: {n:,} sentence pairs")


def _save_split(data, src_path: Path, tgt_path: Path):
    """Save a dataset split to parallel text files (en source, fr target)."""
    with open(src_path, "w", encoding="utf-8") as f_src, \
         open(tgt_path, "w", encoding="utf-8") as f_tgt:
        for example in tqdm(data, desc=f"Writing {src_path.stem}"):
            translation = example["translation"]
            en = translation["en"].strip().replace("\n", " ")
            fr = translation["fr"].strip().replace("\n", " ")
            if en and fr:
                f_src.write(en + "\n")
                f_tgt.write(fr + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download WMT14 en-fr data")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Max training samples (default: all ~40M). "
             "Recommend 5M-10M after dedup/length filter for a Base run.",
    )
    args = parser.parse_args()
    download_wmt(args.output_dir, args.max_train_samples)


if __name__ == "__main__":
    main()
