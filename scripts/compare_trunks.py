"""Put two trunks side by side on the same splits, and say which one earns its size.

A bigger trunk is only worth shipping if it moves the number the cost matrix
cares about. Accuracy alone does not decide it: a model can be more accurate and
still more expensive, because a single downgrade costs twenty times what a single
over-service costs. So the column that settles the argument is ``mean_cost``, and
the column that vetoes it regardless is ``regret``.

    python scripts/compare_trunks.py
    python scripts/compare_trunks.py --a artifacts/ll-22m --b artifacts/ll-150m

Reports raw training-time evaluation, which is what both reports contain. The
shipped pipeline adds a fast path and a confidence floor on top, so these are
trunk-versus-trunk figures and not the published headline numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ROUTE_FIELDS = [
    ("accuracy", "accuracy", "higher"),
    ("regret", "regret", "lower"),
    ("downgraded", "downgraded", "-"),
    ("wasteful", "wasteful", "lower"),
    ("mean_cost", "mean cost", "lower"),
    ("spend_vs_always_large", "spend vs always-large", "lower"),
]
RISK_FIELDS = [
    ("accuracy", "accuracy", "higher"),
    ("p0_recall", "P0 recall", "higher"),
    ("p0_missed", "P0 missed", "lower"),
    ("p0_precision", "P0 precision", "higher"),
    ("mean_cost", "mean cost", "lower"),
]


def load(path: Path) -> dict:
    report = path / "report.json"
    if not report.exists():
        raise SystemExit(f"no report at {report}")
    return json.loads(report.read_text(encoding="utf-8"))


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def verdict(a, b, direction: str) -> str:
    """Which side is better on this field, or blank when it is not a contest."""
    if direction == "-" or not isinstance(a, (int, float)) \
            or not isinstance(b, (int, float)):
        return ""
    if a == b:
        return "tie"
    better_b = b > a if direction == "higher" else b < a
    return "B" if better_b else "A"


def table(name: str, fields, a: dict, b: dict, label_a: str, label_b: str) -> None:
    print(f"\n  {name}")
    print(f"    {'field':<24} {label_a:>12} {label_b:>12}   better")
    for key, label, direction in fields:
        if key not in a and key not in b:
            continue
        va, vb = a.get(key), b.get(key)
        print(f"    {label:<24} {fmt(va):>12} {fmt(vb):>12}   "
              f"{verdict(va, vb, direction)}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, default=ROOT / "artifacts" / "ll-22m")
    parser.add_argument("--b", type=Path, default=ROOT / "artifacts" / "ll-150m")
    args = parser.parse_args(argv[1:])

    ra, rb = load(args.a), load(args.b)
    label_a, label_b = args.a.name, args.b.name

    print("=" * 72)
    for tag, report in ((label_a, ra), (label_b, rb)):
        print(f"{tag:<12} {report['parameters']/1e6:>7.1f}M params   "
              f"trained {report['train_seconds']/60:.1f} min   "
              f"trunk {report['trunk']}")
    print("=" * 72)

    for split in ("test", "audit"):
        sa, sb = ra["reports"].get(split), rb["reports"].get(split)
        if not sa or not sb:
            continue
        print(f"\n[{split}]")
        if "route" in sa and "route" in sb:
            table("route", ROUTE_FIELDS, sa["route"], sb["route"],
                  label_a, label_b)
        if "risk" in sa and "risk" in sb:
            table("risk", RISK_FIELDS, sa["risk"], sb["risk"], label_a, label_b)

    # Per-category audit accuracy is where the argument actually lives: the
    # aggregate hides the two hard-negative cells, which are a twentieth of the
    # corpus and the entire reason an encoder is used instead of regex.
    ca = ra["reports"].get("audit", {}).get("by_category", {})
    cb = rb["reports"].get("audit", {}).get("by_category", {})
    if ca and cb:
        print("\n[audit, per category]")
        print(f"    {'category':<28} {'n':>5} {label_a:>10} {label_b:>10}   delta")
        for cat in sorted(set(ca) | set(cb)):
            da, db = ca.get(cat, {}), cb.get(cat, {})
            aa, ab = da.get("accuracy"), db.get("accuracy")
            delta = ""
            if isinstance(aa, float) and isinstance(ab, float):
                delta = f"{ab - aa:+.3f}"
            print(f"    {cat:<28} {da.get('n', db.get('n', '')):>5} "
                  f"{fmt(aa):>10} {fmt(ab):>10}   {delta}")

    # The decision, stated rather than left to the reader.
    aa = ra["reports"].get("audit", {}).get("route", {})
    ab = rb["reports"].get("audit", {}).get("route", {})
    if aa and ab:
        print("\n" + "=" * 72)
        cost_a, cost_b = aa.get("mean_cost"), ab.get("mean_cost")
        reg_a, reg_b = aa.get("regret"), ab.get("regret")
        print(f"audit mean cost: {label_a} {fmt(cost_a)}  ->  "
              f"{label_b} {fmt(cost_b)}")
        print(f"audit regret   : {label_a} {fmt(reg_a)}  ->  "
              f"{label_b} {fmt(reg_b)}")
        if isinstance(cost_a, float) and isinstance(cost_b, float):
            change = 100 * (cost_b - cost_a) / cost_a if cost_a else 0.0
            print(f"\n{label_b} is {abs(change):.1f}% "
                  f"{'cheaper' if change < 0 else 'more expensive'} than "
                  f"{label_a} on the audit split.")
            if isinstance(reg_b, float) and reg_b > (reg_a or 0):
                print("REGRET INCREASED -- a cheaper mean cost does not "
                      "compensate for this. Downgrades cost 20x.")
        print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
