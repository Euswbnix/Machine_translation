"""Train a SentencePiece BPE tokenizer on the parallel corpus."""

import argparse

from src.data.tokenizer import train_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input text files")
    parser.add_argument("--model-prefix", type=str, default="data/spm", help="Output model prefix")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    args = parser.parse_args()

    train_tokenizer(args.inputs, args.model_prefix, args.vocab_size)


if __name__ == "__main__":
    main()
