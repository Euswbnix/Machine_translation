"""LLM-as-judge pairwise preference evaluation.

Compares translations from two models (A vs B) using Claude as a judge.
Outputs win/loss/tie counts and percentages.

Usage:
    python scripts/wmt26_upgrade/llm_judge.py \
        --src data_enfr_v1/test.en \
        --ref data_enfr_v1/test.fr \
        --hyp-a translations/base_v1.txt \
        --hyp-b translations/big_v1.txt \
        --label-a "Base (60M)" \
        --label-b "Big (209M)" \
        --num-samples 200 \
        --out results/judge_base_vs_big_v1.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("pip install anthropic", file=sys.stderr)
    sys.exit(1)


JUDGE_PROMPT = """You are an expert machine translation evaluator for {src_lang}→{tgt_lang}.

Compare two translations of the same source sentence. Judge which translation is better based on:
1. Accuracy (faithful to the source meaning)
2. Fluency (natural in the target language)
3. Adequacy (no omissions or additions)

Source ({src_lang}): {source}
Reference ({tgt_lang}): {reference}

Translation A: {hyp_a}
Translation B: {hyp_b}

Which translation is better? Reply with exactly one of:
- "A" if Translation A is clearly better
- "B" if Translation B is clearly better
- "TIE" if they are roughly equal in quality

Reply with ONLY the letter or "TIE", nothing else."""


def judge_pair(client, src, ref, hyp_a, hyp_b, src_lang, tgt_lang, swap):
    if swap:
        hyp_a, hyp_b = hyp_b, hyp_a

    prompt = JUDGE_PROMPT.format(
        src_lang=src_lang, tgt_lang=tgt_lang,
        source=src, reference=ref,
        hyp_a=hyp_a, hyp_b=hyp_b,
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )

    verdict = response.content[0].text.strip().upper()

    if swap:
        if verdict == "A":
            verdict = "B"
        elif verdict == "B":
            verdict = "A"

    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp-a", required=True)
    ap.add_argument("--hyp-b", required=True)
    ap.add_argument("--label-a", default="Model A")
    ap.add_argument("--label-b", default="Model B")
    ap.add_argument("--src-lang", default="en")
    ap.add_argument("--tgt-lang", default="fr")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--out", default="results/judge.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.src) as f:
        srcs = [l.strip() for l in f]
    with open(args.ref) as f:
        refs = [l.strip() for l in f]
    with open(args.hyp_a) as f:
        hyps_a = [l.strip() for l in f]
    with open(args.hyp_b) as f:
        hyps_b = [l.strip() for l in f]

    assert len(srcs) == len(refs) == len(hyps_a) == len(hyps_b)

    indices = list(range(len(srcs)))
    if args.num_samples < len(indices):
        indices = random.sample(indices, args.num_samples)

    client = anthropic.Anthropic()

    wins_a, wins_b, ties, errors = 0, 0, 0, 0
    verdicts = []

    for i, idx in enumerate(indices):
        swap = random.random() < 0.5
        try:
            v = judge_pair(
                client, srcs[idx], refs[idx], hyps_a[idx], hyps_b[idx],
                args.src_lang, args.tgt_lang, swap,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(indices)}] Error: {e}", file=sys.stderr)
            errors += 1
            v = "ERROR"
            time.sleep(2)

        if v == "A":
            wins_a += 1
        elif v == "B":
            wins_b += 1
        elif v == "TIE":
            ties += 1

        verdicts.append({"idx": idx, "verdict": v, "swapped": swap})

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(indices)}] "
                  f"A={wins_a} B={wins_b} tie={ties} err={errors}",
                  file=sys.stderr)

    n = wins_a + wins_b + ties
    result = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "num_samples": len(indices),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "errors": errors,
        "win_rate_a": round(wins_a / max(n, 1) * 100, 1),
        "win_rate_b": round(wins_b / max(n, 1) * 100, 1),
        "tie_rate": round(ties / max(n, 1) * 100, 1),
        "verdicts": verdicts,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"  {args.label_a} wins: {wins_a} ({result['win_rate_a']}%)",
          file=sys.stderr)
    print(f"  {args.label_b} wins: {wins_b} ({result['win_rate_b']}%)",
          file=sys.stderr)
    print(f"  Ties: {ties} ({result['tie_rate']}%)", file=sys.stderr)
    print(f"  Errors: {errors}", file=sys.stderr)
    print(f"  Saved to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
