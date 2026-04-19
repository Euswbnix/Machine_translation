"""Clean the en-fr parallel training corpus.

Same idea as clean_data.py but both sides are word-tokenized (split on
whitespace) since both languages use Latin script. Drops:
  - empty lines
  - too-short or too-long pairs (token count)
  - pairs with extreme src/tgt length ratio
  - tgt lines appearing more than DUP_THRESHOLD times (Common Crawl / UN boilerplate)
  - pairs where one side is suspiciously non-Latin (likely encoding garbage)
"""

import argparse
from collections import Counter


def _latin_ratio(s: str) -> float:
    if not s:
        return 0.0
    # crude: fraction of chars that are ASCII letters/punct/digits/space or common Latin extended
    good = sum(1 for c in s if c.isascii() or 0x00C0 <= ord(c) <= 0x017F)
    return good / len(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/train.en")
    parser.add_argument("--tgt", default="data/train.fr")
    parser.add_argument("--out-src", default="data/train.clean.en")
    parser.add_argument("--out-tgt", default="data/train.clean.fr")
    parser.add_argument("--dup-threshold", type=int, default=50)
    parser.add_argument("--min-tokens", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--min-ratio", type=float, default=0.5)
    parser.add_argument("--max-ratio", type=float, default=2.0)
    parser.add_argument("--min-latin-ratio", type=float, default=0.9)
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
        for s, t in zip(fs, ft):
            s_stripped = s.strip()
            t_stripped = t.strip()
            sl = len(s_stripped.split())
            tl = len(t_stripped.split())
            if sl == 0 or tl == 0:
                reasons["empty"] += 1
                dropped += 1
                continue
            if sl < args.min_tokens or tl < args.min_tokens:
                reasons["too_short"] += 1
                dropped += 1
                continue
            if sl > args.max_tokens or tl > args.max_tokens:
                reasons["too_long"] += 1
                dropped += 1
                continue
            r = sl / tl
            if r < args.min_ratio or r > args.max_ratio:
                reasons["bad_ratio"] += 1
                dropped += 1
                continue
            if (_latin_ratio(s_stripped) < args.min_latin_ratio or
                    _latin_ratio(t_stripped) < args.min_latin_ratio):
                reasons["non_latin"] += 1
                dropped += 1
                continue
            if t in bad:
                reasons["duplicate"] += 1
                dropped += 1
                continue
            os_.write(s)
            ot.write(t)
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
