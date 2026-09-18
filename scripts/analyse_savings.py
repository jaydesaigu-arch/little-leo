"""Turn the savings run into the numbers a README may honestly print.

Three separate figures, because one number would hide the thing that matters.

**Gross saving** counts a ``NO_MODEL`` decision as costing nothing, which is
arithmetically true and practically misleading: if the task needed an answer and
none was produced, that is a failure whose dollar cost happens to be zero.

**Net saving** excludes prompts where routing declined to call anything but the
task required an answer. This is the figure to quote.

**No-answer rate** is reported on its own line, because it is the failure a
reader most needs to see and the only one that never shows up as cost.

Quality is never reported without its interval. A materially-worse rate drawn
from a dozen judgements can sit anywhere between "excellent" and "unshippable",
and a point estimate from that sample would be a claim the data cannot support.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
              / denominator)
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def main() -> int:
    path = ROOT / "data" / "savings.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    n = len(rows)

    baseline = sum(r["baseline_cost_usd"] for r in rows)
    routed = sum(r["routed_cost_usd"] for r in rows)

    # A NO_MODEL decision on a task that needed an answer: no call, no answer.
    no_answer = [r for r in rows
                 if r["ll_route"] == "NO_MODEL" and r["asserted_tier"] != "NO_MODEL"]
    served = [r for r in rows if r not in no_answer]
    net_baseline = sum(r["baseline_cost_usd"] for r in served)
    net_routed = sum(r["routed_cost_usd"] for r in served)

    print(f"PROMPTS: {n}")
    print()
    print("COST")
    print(f"  baseline, every prompt to frontier   ${baseline:.5f}")
    print(f"  LL-22M routed                        ${routed:.5f}")
    print(f"  GROSS SAVING                         {100 * (1 - routed / baseline):.1f}%")
    print(f"  net of unanswered prompts            {100 * (1 - net_routed / net_baseline):.1f}%"
          f"   (n={len(served)})")
    print()

    print("TOKENS (completion + prompt, as billed)")
    tok_base = sum((r.get("frontier_in") or 0) + (r.get("frontier_out") or 0) for r in rows)
    tok_routed = 0
    for r in rows:
        if r["ll_route"] == "LARGE":
            tok_routed += (r.get("frontier_in") or 0) + (r.get("frontier_out") or 0)
        elif r["ll_route"] == "SMALL":
            tok_routed += (r.get("cheap_in") or 0) + (r.get("cheap_out") or 0)
    print(f"  baseline  {tok_base:>8,}")
    print(f"  routed    {tok_routed:>8,}")
    print(f"  saving    {100 * (1 - tok_routed / tok_base):>7.1f}%")
    print()

    print("ROUTING (asserted tier -> LL decision)")
    grid: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        grid[(r["asserted_tier"], r["ll_route"])] += 1
    for (asserted, chosen), count in sorted(grid.items()):
        flag = ""
        if chosen == "NO_MODEL" and asserted != "NO_MODEL":
            flag = "   <-- no answer delivered"
        elif asserted == "LARGE" and chosen != "LARGE":
            flag = "   <-- downgraded a hard task"
        print(f"  {asserted:<9} -> {chosen:<9} {count:>4}{flag}")
    low, high = wilson(len(no_answer), n)
    print(f"\n  NO-ANSWER RATE  {len(no_answer)}/{n} = {100 * len(no_answer) / n:.1f}%"
          f"   95% CI [{100 * low:.1f}%, {100 * high:.1f}%]")
    print()

    judged = [r for r in rows if r.get("judged")]
    worse = [r for r in judged if r.get("cheap_worse")]
    low, high = wilson(len(worse), len(judged))
    print("QUALITY, where routing sent work to the cheap model")
    print(f"  judged downgrades      {len(judged)}")
    print(f"  materially worse       {len(worse)} = "
          f"{100 * len(worse) / max(1, len(judged)):.1f}%")
    print(f"  95% CI                 [{100 * low:.1f}%, {100 * high:.1f}%]")
    verdicts: dict[str, int] = defaultdict(int)
    for r in judged:
        verdicts[r.get("verdict", "?")] += 1
    print(f"  verdict spread         {dict(verdicts)}")
    if worse:
        print("  cases:")
        for r in worse:
            print(f"    {r['kind']:<15} {r.get('reason','')[:72]}")
    print()

    print("PER-TIER, where the saving comes from")
    per: dict[str, dict] = defaultdict(lambda: {"n": 0, "base": 0.0, "routed": 0.0})
    for r in rows:
        bucket = per[r["ll_route"]]
        bucket["n"] += 1
        bucket["base"] += r["baseline_cost_usd"]
        bucket["routed"] += r["routed_cost_usd"]
    for tier in ("NO_MODEL", "SMALL", "LARGE"):
        b = per.get(tier)
        if not b:
            continue
        print(f"  {tier:<9} n={b['n']:<4} baseline ${b['base']:.5f} -> "
              f"routed ${b['routed']:.5f}   saves ${b['base'] - b['routed']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
