"""Train a SentencePiece BPE tokenizer on the parallel corpus."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.tokenizer import train_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input text files")
    parser.add_argument("--model-prefix", type=str, default="data/spm",
                        help="Output model prefix")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--character-coverage", type=float, default=1.0,
                        help="Character coverage. 1.0 = no <unk> from rare chars "
                             "(recommended). 0.9995 was the old default and caused "
                             "~4.4%% of French valid sentences to hit <unk> on "
                             "accented characters.")
    parser.add_argument("--input-sentence-size", type=int, default=10_000_000,
                        help="Max sentences sampled for BPE training.")
    parser.add_argument("--num-threads", type=int, default=None,
                        help="SPM BPE training threads. Default: all CPU cores "
                             "(os.cpu_count()). BPE has serial merges so scaling "
                             "isn't linear — expect 1.5–2× going 8 → 32.")
    args = parser.parse_args()

    num_threads = args.num_threads or (os.cpu_count() or 8)
    print(f"Using {num_threads} threads for SPM training.")

    train_tokenizer(
        args.inputs,
        args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        input_sentence_size=args.input_sentence_size,
        num_threads=num_threads,
    )


if __name__ == "__main__":
    main()
