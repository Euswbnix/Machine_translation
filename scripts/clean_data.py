"""Clean the parallel training corpus.

Drops:
  - empty lines
  - too-short or too-long pairs
  - pairs with extreme zh-char / en-token length ratios
  - English lines that appear more than DUP_THRESHOLD times (UN boilerplate etc.)
"""

import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/train.zh")
    parser.add_argument("--tgt", default="data/train.en")
    parser.add_argument("--out-src", default="data/train.clean.zh")
    parser.add_argument("--out-tgt", default="data/train.clean.en")
    parser.add_argument("--dup-threshold", type=int, default=50,
                        help="Drop tgt lines appearing more than this many times")
    parser.add_argument("--min-src-chars", type=int, default=3)
    parser.add_argument("--min-tgt-tokens", type=int, default=3)
    parser.add_argument("--max-src-chars", type=int, default=400)
    parser.add_argument("--max-tgt-tokens", type=int, default=200)
    parser.add_argument("--min-ratio", type=float, default=0.5)
    parser.add_argument("--max-ratio", type=float, default=5.0)
    args = parser.parse_args()

    print("Pass 1: counting tgt-line frequencies...")
    cnt: Counter = Counter()
    with open(args.tgt) as f:
        for line in f:
            cnt[line] += 1
    bad = {l for l, c in cnt.items() if c > args.dup_threshold}
    bad_total = sum(cnt[l] for l in bad)
    print(f"  blocking {len(bad):,} unique lines "
          f"({bad_total:,} total occurrences)")
    del cnt

    print("Pass 2: filtering...")
    kept = dropped = 0
    reasons: Counter = Counter()
    with open(args.src) as fs, open(args.tgt) as ft, \
            open(args.out_src, "w") as os_, open(args.out_tgt, "w") as ot:
        for z, e in zip(fs, ft):
            zl = len(z.strip())
            el = len(e.split())
            if zl == 0 or el == 0:
                reasons["empty"] += 1
                dropped += 1
                continue
            if zl < args.min_src_chars or el < args.min_tgt_tokens:
                reasons["too_short"] += 1
                dropped += 1
                continue
            if zl > args.max_src_chars or el > args.max_tgt_tokens:
                reasons["too_long"] += 1
                dropped += 1
                continue
            r = zl / el
            if r < args.min_ratio or r > args.max_ratio:
                reasons["bad_ratio"] += 1
                dropped += 1
                continue
            if e in bad:
                reasons["duplicate"] += 1
                dropped += 1
                continue
            os_.write(z)
            ot.write(e)
            kept += 1

    total = kept + dropped
    print(f"\nkept:    {kept:>12,} ({100*kept/total:.1f}%)")
    print(f"dropped: {dropped:>12,} ({100*dropped/total:.1f}%)")
    print("reasons:")
    for k, v in reasons.most_common():
        print(f"  {k:<12} {v:>12,}")
    print(f"\nwrote:")
    print(f"  {args.out_src}")
    print(f"  {args.out_tgt}")


if __name__ == "__main__":
    main()
