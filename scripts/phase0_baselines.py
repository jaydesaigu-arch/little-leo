"""Phase 0: build the corpus, prove it is clean, and set the bar.

Runs the overlap assertion first — a held-out split that shares text with train
measures memorisation, and every number downstream of it is void. Then runs both
the rule baseline and the two bag-of-words controls, on every split, and prints
what the trained model has to beat.

    python scripts/phase0_baselines.py --force
    python scripts/phase0_baselines.py --split audit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from littleleo.baselines import BASELINES  # noqa: E402
from littleleo.learned_baselines import fitted_baselines  # noqa: E402
from littleleo.metrics import risk_report, route_report  # noqa: E402
from littleleo.schema import SITE_ACT, SITE_PRE  # noqa: E402
from littleleo.synth import (assert_no_overlap, generate,  # noqa: E402
                             read_jsonl, write_jsonl)

DATA = ROOT / "data" / "corpus.jsonl"


def build_corpus(force: bool = False):
    if DATA.is_file() and not force:
        return read_jsonl(DATA)
    rows = generate()
    write_jsonl(rows, DATA)
    return rows


def per_category(rows, predictions) -> dict:
    buckets: dict[str, dict] = {}
    for row, (route, _risk) in zip(rows, predictions):
        if row.site != SITE_PRE:
            continue
        bucket = buckets.setdefault(row.source, {"n": 0, "ok": 0, "regret": 0})
        bucket["n"] += 1
        bucket["ok"] += int(route == row.route)
        bucket["regret"] += int(route < row.route)
    return {name: {"n": b["n"], "accuracy": round(b["ok"] / b["n"], 3),
                   "regret": round(b["regret"] / b["n"], 3)}
            for name, b in sorted(buckets.items())}


def evaluate(rows, split: str, learned: dict) -> dict:
    subset = [r for r in rows if r.split == split]
    pre = [r for r in subset if r.site == SITE_PRE]
    act = [r for r in subset if r.site == SITE_ACT]

    results: dict[str, dict] = {}
    for name, router in BASELINES.items():
        started = time.perf_counter()
        predictions = [router(r) for r in subset]
        elapsed = (time.perf_counter() - started) * 1e6 / max(1, len(subset))
        results[name] = _score(subset, pre, act, predictions, elapsed)

    for name, model in learned.items():
        started = time.perf_counter()
        predictions = model.predict_many(subset)
        elapsed = (time.perf_counter() - started) * 1e6 / max(1, len(subset))
        results[name] = _score(subset, pre, act, predictions, elapsed)

    return {"split": split, "pre": len(pre), "act": len(act), "results": results}


def _score(subset, pre, act, predictions, elapsed_us) -> dict:
    pre_pred = [p[0] for p, r in zip(predictions, subset) if r.site == SITE_PRE]
    act_pred = [p[1] for p, r in zip(predictions, subset) if r.site == SITE_ACT]
    return {
        "route": route_report([r.route for r in pre], pre_pred),
        "risk": risk_report([r.risk for r in act], act_pred),
        "by_category": per_category(subset, predictions),
        "microseconds_per_turn": round(elapsed_us, 1),
    }


def render(report: dict) -> str:
    lines = [f"split: {report['split']}  ({report['pre']} routing turns, "
             f"{report['act']} actions)", ""]
    lines.append("ROUTING")
    lines.append(f"{'router':<20}{'acc':>7}{'downgr':>9}{'regret':>9}"
                 f"{'spend':>8}{'mean cost':>11}")
    lines.append("-" * 64)
    for name, row in report["results"].items():
        r = row["route"]
        lines.append(f"{name:<20}{r['accuracy']:>7.3f}{r['downgraded']:>9.3f}"
                     f"{r['regret']:>9.3f}{r['large_call_spend']:>8.3f}"
                     f"{r['mean_cost']:>11.3f}")
    lines.append("")
    lines.append("THE TWO INVERSE CATEGORIES (the whole point)")
    categories = ["hard_negative_trivial", "hard_negative_large"]
    lines.append(f"{'router':<20}" + "".join(f"{c[14:]:>26}" for c in categories))
    lines.append("-" * 72)
    for name, row in report["results"].items():
        cells = []
        for category in categories:
            values = row["by_category"].get(category)
            cells.append(f"acc {values['accuracy']:.3f} reg {values['regret']:.3f}"
                         if values else "n/a")
        lines.append(f"{name:<20}" + "".join(f"{c:>26}" for c in cells))
    lines.append("")
    lines.append("RISK (advisory; host rules remain the gate)")
    lines.append(f"{'router':<20}{'P0 recall':>11}{'P0 prec':>10}{'missed':>8}")
    lines.append("-" * 50)
    for name, row in report["results"].items():
        k = row["risk"]
        lines.append(f"{name:<20}{k['p0_recall']:>11.3f}"
                     f"{k['p0_precision']:>10.3f}{k['p0_missed']:>8}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=None,
                        help="one split, or every split when omitted")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv[1:])

    rows = build_corpus(force=args.force)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.split] = counts.get(row.split, 0) + 1
    print(f"corpus: {len(rows)} rows  {counts}")

    findings = assert_no_overlap(rows)   # raises on any train/test or train/audit overlap
    print(f"overlap check PASSED  {findings}")
    print()

    learned = fitted_baselines([r for r in rows if r.split == "train"])
    splits = [args.split] if args.split else ["dev", "test", "audit"]
    reports = {}
    for split in splits:
        report = evaluate(rows, split, learned)
        reports[split] = report
        print(render(report))
        print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2, default=str),
                             encoding="utf-8")
        print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
