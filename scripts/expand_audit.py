"""Re-test the trained model on a much larger held-out set, with honest intervals.

The audit split carried 47 ``hard_negative_large`` turns, and the model got all
47 right. That is a good sign and a weak measurement: by the rule of three, zero
failures in 47 trials is consistent with a true failure rate anywhere up to
about 6.4%. A model card that says "zero regret" on that basis is claiming more
than the evidence supports.

Pool C's clause combinatorics allow thousands of unique turns, so the fix costs
seconds: generate a far larger held-out set from the same pool, assert it does
not overlap training, and re-measure. Nothing is tuned against it and the model
is not retrained — this is the same evaluation on more data, which can only make
the estimate better or reveal that it was never real.

Wilson intervals are reported rather than point estimates, because the whole
reason for doing this is that a point estimate from 47 samples was misleading.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from littleleo.baselines import keyword  # noqa: E402
from littleleo.metrics import risk_report, route_report  # noqa: E402
from littleleo.model import MAX_LENGTH, LittleLeo  # noqa: E402
from littleleo.schema import SITE_ACT, SITE_PRE, Risk, Route  # noqa: E402
from littleleo.synth import _act_examples, _pre_examples, read_jsonl  # noqa: E402


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """A binomial confidence interval that behaves at the extremes.

    The normal approximation gives a zero-width interval when the observed rate
    is 0 or 1, which is exactly the case this script exists to report on.
    Wilson does not.
    """
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
              / denominator)
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def build(target_per_category: int, existing: set[str], seed: int) -> list:
    """Generate a large pool-C set, skipping anything already in the corpus."""
    rng = random.Random(seed)
    kept: list = []
    counts: dict[str, int] = {}
    wanted_total = target_per_category * 5
    for _ in range(400):
        for candidate in _pre_examples(rng, 1200, "audit"):
            if counts.get(candidate.source, 0) >= target_per_category:
                continue
            key = candidate.encoded()
            if key in existing:
                continue
            existing.add(key)
            kept.append(candidate)
            counts[candidate.source] = counts.get(candidate.source, 0) + 1
        if len(kept) >= wanted_total:
            break
    for _ in range(200):
        for candidate in _act_examples(rng, 600, "audit"):
            if counts.get(candidate.source, 0) >= target_per_category:
                continue
            key = candidate.encoded()
            if key in existing:
                continue
            existing.add(key)
            kept.append(candidate)
            counts[candidate.source] = counts.get(candidate.source, 0) + 1
        if all(counts.get(s, 0) >= target_per_category
               for s in ("act_p0", "act_p1", "act_p2")):
            break
    return kept


@torch.no_grad()
def infer(model, tokenizer, rows):
    encoded = tokenizer([r.encoded() for r in rows], truncation=True,
                        max_length=MAX_LENGTH, padding="longest",
                        return_tensors="pt")
    routes, risks = [], []
    for start in range(0, len(rows), 128):
        chunk = {k: v[start:start + 128] for k, v in encoded.items()}
        out = model(chunk["input_ids"], chunk["attention_mask"])
        routes.extend(Route(int(v)) for v in out["route_logits"].argmax(-1))
        risks.extend(Risk(int(v)) for v in out["risk_logits"].argmax(-1))
    return routes, risks


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path,
                        default=ROOT / "artifacts" / "ll-22m")
    parser.add_argument("--per-category", type=int, default=600)
    parser.add_argument("--seed", type=int, default=77001)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data" / "expanded-audit.json")
    args = parser.parse_args(argv[1:])

    corpus = read_jsonl(ROOT / "data" / "corpus.jsonl")
    train_keys = {r.encoded() for r in corpus if r.split == "train"}
    all_keys = {r.encoded() for r in corpus}

    rows = build(args.per_category, set(all_keys), args.seed)
    overlap = sum(1 for r in rows if r.encoded() in train_keys)
    assert overlap == 0, f"{overlap} generated turns appear in train"
    print(f"generated {len(rows)} fresh pool-C turns, 0 overlap with train")

    blob = torch.load(args.artifact / "model.pt", map_location="cpu",
                      weights_only=False)
    model = LittleLeo(blob["trunk"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.artifact)

    routes, risks = infer(model, tokenizer, rows)

    pre = [(r, p) for r, p in zip(rows, routes) if r.site == SITE_PRE]
    act = [(r, p) for r, p in zip(rows, risks) if r.site == SITE_ACT]
    route_stats = route_report([r.route for r, _ in pre], [p for _, p in pre])
    risk_stats = risk_report([r.risk for r, _ in act], [p for _, p in act])

    print(f"\nROUTE  n={route_stats['n']}  acc {route_stats['accuracy']:.3f}  "
          f"downgraded {route_stats['downgraded']:.3f}  "
          f"regret {route_stats['regret']:.4f}  "
          f"mean cost {route_stats['mean_cost']:.3f}")

    buckets: dict[str, dict] = {}
    for row, predicted in pre:
        b = buckets.setdefault(row.source, {"n": 0, "ok": 0, "regret": 0})
        b["n"] += 1
        b["ok"] += int(predicted == row.route)
        b["regret"] += int(predicted < row.route)

    print(f"\n{'category':<24}{'n':>6}{'acc':>8}{'regret':>9}"
          f"{'regret 95% CI':>22}")
    print("-" * 69)
    report: dict = {"route": route_stats, "risk": risk_stats, "categories": {}}
    for name, b in sorted(buckets.items()):
        low, high = wilson(b["regret"], b["n"])
        print(f"{name:<24}{b['n']:>6}{b['ok']/b['n']:>8.3f}"
              f"{b['regret']/b['n']:>9.4f}"
              f"{f'[{low:.4f}, {high:.4f}]':>22}")
        report["categories"][name] = {
            "n": b["n"], "accuracy": round(b["ok"] / b["n"], 4),
            "regret": round(b["regret"] / b["n"], 4),
            "regret_ci95": [round(low, 4), round(high, 4)]}

    p0_low, p0_high = wilson(
        round(risk_stats["p0_recall"] * risk_stats["p0_support"]),
        risk_stats["p0_support"])
    print(f"\nRISK  P0 recall {risk_stats['p0_recall']:.4f} "
          f"(n={risk_stats['p0_support']})  95% CI "
          f"[{p0_low:.4f}, {p0_high:.4f}]  missed {risk_stats['p0_missed']}")

    rules = [keyword(r)[1] for r, _ in act]
    combined = [Risk(max(int(a), int(b))) for a, b in zip(rules, [p for _, p in act])]
    combined_stats = risk_report([r.risk for r, _ in act], combined)
    print(f"      regex->encoder P0 recall {combined_stats['p0_recall']:.4f}  "
          f"missed {combined_stats['p0_missed']}")
    report["risk_combined"] = combined_stats

    args.out.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
