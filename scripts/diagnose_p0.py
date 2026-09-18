"""Why the encoder missed eleven P0 actions on test and none on audit.

Two questions, answered from the data rather than from a theory:

1. What do the misses actually say? Printed verbatim, with the tail phrasing
   that produced them, so the gap can be named instead of guessed at.
2. Does the deterministic gate close it? The architecture says the regex runs
   first and the encoder only escalates, so the combined recall is what a host
   would actually experience. That number is the one that matters; the
   encoder's solo recall is diagnostic.

The combination rule is the safety invariant in one line:

    final = max(regex_risk, encoder_risk)

The encoder may raise a tier. It may never lower one. A host that follows this
cannot be made less careful by a model that is wrong, which is the whole reason
the model is allowed near a destructive operation at all.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from littleleo.baselines import keyword  # noqa: E402
from littleleo.metrics import risk_report  # noqa: E402
from littleleo.model import MAX_LENGTH, LittleLeo  # noqa: E402
from littleleo.schema import SITE_ACT, Risk  # noqa: E402
from littleleo.synth import ACT_TAILS, read_jsonl  # noqa: E402


def load(artifact: Path):
    blob = torch.load(artifact / "model.pt", map_location="cpu",
                      weights_only=False)
    model = LittleLeo(blob["trunk"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, AutoTokenizer.from_pretrained(artifact)


@torch.no_grad()
def encoder_risk(model, tokenizer, rows) -> list[Risk]:
    encoded = tokenizer([r.encoded() for r in rows], truncation=True,
                        max_length=MAX_LENGTH, padding="longest",
                        return_tensors="pt")
    out: list[Risk] = []
    for start in range(0, len(rows), 64):
        chunk = {k: v[start:start + 64] for k, v in encoded.items()}
        logits = model(chunk["input_ids"], chunk["attention_mask"])["risk_logits"]
        out.extend(Risk(int(v)) for v in logits.argmax(-1))
    return out


def which_tail(action: str) -> str:
    """Which generator tail produced this action, for naming the gap."""
    for pool, by_risk in ACT_TAILS.items():
        for tails in by_risk.values():
            for tail in tails:
                stem = tail.split("{system}")[0].strip()
                if stem and stem in action:
                    return f"{pool}: {tail}"
    return "unmatched"


def main() -> int:
    artifact = ROOT / "artifacts" / "ll-22m"
    model, tokenizer = load(artifact)
    rows = read_jsonl(ROOT / "data" / "corpus.jsonl")

    summary: dict = {}
    for split in ("test", "audit"):
        act = [r for r in rows if r.split == split and r.site == SITE_ACT]
        predicted = encoder_risk(model, tokenizer, act)
        rules = [keyword(r)[1] for r in act]
        combined = [Risk(max(int(a), int(b))) for a, b in zip(rules, predicted)]

        truth = [r.risk for r in act]
        enc = risk_report(truth, predicted)
        rul = risk_report(truth, rules)
        com = risk_report(truth, combined)

        misses = [(r, p) for r, p in zip(act, predicted)
                  if r.risk == Risk.P0_DESTRUCTIVE and p != Risk.P0_DESTRUCTIVE]

        print(f"\n===== {split.upper()} =====")
        print(f"  encoder alone : P0 recall {enc['p0_recall']:.3f}  "
              f"missed {enc['p0_missed']}")
        print(f"  regex alone   : P0 recall {rul['p0_recall']:.3f}  "
              f"missed {rul['p0_missed']}")
        print(f"  regex->encoder: P0 recall {com['p0_recall']:.3f}  "
              f"missed {com['p0_missed']}  precision {com['p0_precision']:.3f}")

        if misses:
            tails = Counter(which_tail(r.action) for r, _ in misses)
            print(f"\n  the {len(misses)} encoder misses, by generator tail:")
            for tail, count in tails.most_common():
                print(f"    {count:>3}x  {tail}")
            print("\n  verbatim (first 5):")
            for row, predicted_risk in misses[:5]:
                print(f"    [{predicted_risk.name}] {row.action}")

        summary[split] = {"encoder": enc, "regex": rul, "combined": com,
                          "missed_tails": dict(Counter(
                              which_tail(r.action) for r, _ in misses))}

    # Which P0 tails exist per pool, and which carry an overt danger token.
    print("\n===== P0 TAILS BY POOL =====")
    overt = ("irreversibl", "drop", "remove", "purge", "overwrite", "replace",
             "clear", "shell", "execute")
    for pool, by_risk in ACT_TAILS.items():
        for tail in by_risk[Risk.P0_DESTRUCTIVE]:
            flags = [w for w in overt if w in tail.lower()]
            has_path = "{system}" in tail
            print(f"  {pool}: path={'yes' if has_path else 'NO '}  "
                  f"tokens={flags or 'none'}")
            print(f"       {tail}")

    (ROOT / "data" / "p0-diagnosis.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten: {ROOT / 'data' / 'p0-diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
