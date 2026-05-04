"""Clean WMT17 en-zh parallel data with v1-style strict filters.

Filters:
  - Remove empty / whitespace-only lines
  - Remove duplicates
  - Length range: [3, 200] tokens (whitespace-split for en, char-split for zh)
  - Length ratio: src/tgt in [0.3, 3.0] (wider than en-fr due to script diff)
  - Chinese character ratio >= 0.3 on target side

Usage:
    python scripts/wmt26_upgrade/clean_data_enzh.py \
        --src data_enzh/train.en --tgt data_enzh/train.zh \
        --src-out data_enzh/train.clean.en --tgt-out data_enzh/train.clean.zh
"""

import argparse
import re
import sys
import unicodedata


def is_cjk(ch):
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF) or
        (0x3400 <= cp <= 0x4DBF) or
        (0x20000 <= cp <= 0x2A6DF) or
        (0x2A700 <= cp <= 0x2B73F) or
        (0x2B740 <= cp <= 0x2B81F) or
        (0x2B820 <= cp <= 0x2CEAF) or
        (0xF900 <= cp <= 0xFAFF) or
        (0x2F800 <= cp <= 0x2FA1F)
    )


def zh_char_ratio(text):
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if is_cjk(ch))
    return cjk / len(text)


def en_token_count(text):
    return len(text.split())


def zh_token_count(text):
    return len(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--src-out", required=True)
    ap.add_argument("--tgt-out", required=True)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--min-ratio", type=float, default=0.3)
    ap.add_argument("--max-ratio", type=float, default=3.0)
    ap.add_argument("--min-zh-ratio", type=float, default=0.3)
    args = ap.parse_args()

    seen = set()
    kept = 0
    total = 0
    reasons = {"empty": 0, "dup": 0, "len": 0, "ratio": 0, "script": 0}

    with open(args.src, encoding="utf-8") as fs, \
         open(args.tgt, encoding="utf-8") as ft, \
         open(args.src_out, "w", encoding="utf-8") as fso, \
         open(args.tgt_out, "w", encoding="utf-8") as fto:

        for src, tgt in zip(fs, ft):
            total += 1
            src = src.strip()
            tgt = tgt.strip()

            if not src or not tgt:
                reasons["empty"] += 1
                continue

            pair_key = (src, tgt)
            if pair_key in seen:
                reasons["dup"] += 1
                continue
            seen.add(pair_key)

            src_len = en_token_count(src)
            tgt_len = zh_token_count(tgt)

            if not (args.min_len <= src_len <= args.max_len):
                reasons["len"] += 1
                continue
            if not (args.min_len <= tgt_len <= args.max_len):
                reasons["len"] += 1
                continue

            if tgt_len > 0 and src_len > 0:
                ratio = src_len / tgt_len
                if not (args.min_ratio <= ratio <= args.max_ratio):
                    reasons["ratio"] += 1
                    continue

            if zh_char_ratio(tgt) < args.min_zh_ratio:
                reasons["script"] += 1
                continue

            fso.write(src + "\n")
            fto.write(tgt + "\n")
            kept += 1

            if total % 5000000 == 0:
                print(f"  {total:,} processed, {kept:,} kept", file=sys.stderr)

    keep_rate = 100 * kept / max(total, 1)
    print(f"\nTotal: {total:,}  Kept: {kept:,}  ({keep_rate:.1f}%)", file=sys.stderr)
    print(f"Filtered out:", file=sys.stderr)
    for reason, count in sorted(reasons.items()):
        print(f"  {reason}: {count:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
