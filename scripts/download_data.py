"""Download and prepare WMT zh-en parallel corpus for training.

Uses the HuggingFace datasets library to download WMT data,
then saves to plain text files for tokenizer training and model training.
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def download_wmt(output_dir: str, max_train_samples: int | None = None):
    """Download WMT17 zh-en data and save as parallel text files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading WMT17 zh-en dataset...")
    dataset = load_dataset("wmt17", "zh-en")

    # Process train split
    print("Processing training data...")
    train_data = dataset["train"]
    if max_train_samples:
        train_data = train_data.select(range(min(max_train_samples, len(train_data))))

    _save_split(train_data, output_dir / "train.zh", output_dir / "train.en")

    # Process validation split
    print("Processing validation data...")
    _save_split(dataset["validation"], output_dir / "valid.zh", output_dir / "valid.en")

    # Process test split
    print("Processing test data...")
    _save_split(dataset["test"], output_dir / "test.zh", output_dir / "test.en")

    print(f"Data saved to {output_dir}")
    for split in ["train", "valid", "test"]:
        n = sum(1 for _ in open(output_dir / f"{split}.zh"))
        print(f"  {split}: {n:,} sentence pairs")


def _save_split(data, src_path: Path, tgt_path: Path):
    """Save a dataset split to parallel text files."""
    with open(src_path, "w") as f_src, open(tgt_path, "w") as f_tgt:
        for example in tqdm(data, desc=f"Writing {src_path.stem}"):
            translation = example["translation"]
            zh = translation["zh"].strip().replace("\n", " ")
            en = translation["en"].strip().replace("\n", " ")
            if zh and en:
                f_src.write(zh + "\n")
                f_tgt.write(en + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download WMT zh-en data")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Max training samples (for debugging)",
    )
    args = parser.parse_args()
    download_wmt(args.output_dir, args.max_train_samples)


if __name__ == "__main__":
    main()
