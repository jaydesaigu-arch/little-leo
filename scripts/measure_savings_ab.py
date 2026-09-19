"""Score two routers against one set of live calls, so the comparison is paired.

Running ``measure_savings.py`` twice and differencing the totals would measure
two things at once. Every prompt is sent to both models on every run, and the
judge is asked again, so a second run redraws both the provider's sampling and
the judge's verdict on borderline pairs. That verdict is not stable: five reruns
of the *identical* comparison returned materially-worse rates of 17%, 11.1%,
3.3%, 15% and 10.0%. A difference between two runs of ten points could be
entirely judge noise and nothing to do with the routers.

So this script makes the calls **once** and scores both routers against the same
responses and the same verdicts. Each prompt produces:

* one cheap answer and one frontier answer, billed once;
* at most one judge verdict, reused by both routers;
* two routed costs, one per router, computed from those same numbers.

What remains between the two columns is the routing decisions, which is the only
thing under test. Where both routers agree, they are charged identically by
construction and contribute nothing to the difference -- which is the point.

    python scripts/measure_savings_ab.py --a artifacts/ll-22m/onnx \\
                                         --b artifacts/ll-150m/onnx
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from littleleo.grounding_fixture import ITEMS, TIER_NAMES  # noqa: E402
from littleleo.pipeline import LittleLeoRouter  # noqa: E402
from littleleo.providers import openrouter_key, openrouter_pricing  # noqa: E402

# Reuse the call machinery and prompts verbatim rather than restating them, so
# this experiment cannot quietly diverge from the one that produced the
# published figures.
from measure_savings import (ANSWER_SYSTEM, ANSWER_TOKENS, CHEAP,  # noqa: E402
                             FRONTIER, JUDGE, JUDGE_SYSTEM, JUDGE_TOKENS,
                             Budget, call)


def routed_cost(route: str, cheap_cost: float, frontier_cost: float):
    """What this router would have paid for this prompt, and who served it."""
    if route == "NO_MODEL":
        return 0.0, "none"
    if route == "SMALL":
        return cheap_cost, CHEAP
    return frontier_cost, FRONTIER


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path,
                        default=ROOT / "artifacts" / "ll-22m" / "onnx")
    parser.add_argument("--b", type=Path,
                        default=ROOT / "artifacts" / "ll-150m" / "onnx")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ceiling", type=float, default=3.00)
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data" / "savings-ab.jsonl")
    args = parser.parse_args(argv[1:])

    key = openrouter_key(ROOT)
    if not key:
        print("no OpenRouter key", file=sys.stderr)
        return 2
    pricing = openrouter_pricing(key, [CHEAP, FRONTIER, JUDGE])

    label_a, label_b = args.a.parent.name, args.b.parent.name
    router_a = LittleLeoRouter(args.a, quantised=False)
    router_b = LittleLeoRouter(args.b, quantised=False)
    print(f"A = {label_a}   B = {label_b}")

    budget = Budget(args.ceiling, pricing)
    rng = random.Random(args.seed)
    prompts = [i for i in ITEMS for _ in range(args.repeats)]
    rng.shuffle(prompts)
    print(f"{len(prompts)} prompts, ceiling ${args.ceiling:.2f}\n")

    totals = {"baseline": 0.0, label_a: 0.0, label_b: 0.0}
    tokens = {"baseline": 0, label_a: 0, label_b: 0}
    splits: dict[str, dict[str, int]] = {label_a: defaultdict(int),
                                         label_b: defaultdict(int)}
    downgrades = {label_a: 0, label_b: 0}
    worse = {label_a: 0, label_b: 0}
    unanswered = {label_a: 0, label_b: 0}
    disagreements = 0

    handle = args.out.open("w", encoding="utf-8")
    aborted = ""
    for index, item in enumerate(prompts, 1):
        if budget.would_breach(CHEAP) or budget.would_breach(FRONTIER):
            aborted = f"ceiling reached after {index - 1} prompts"
            break

        da = router_a.route(item.request)
        db = router_b.route(item.request)
        ra, rb = da["route"], db["route"]
        splits[label_a][ra] += 1
        splits[label_b][rb] += 1
        if ra != rb:
            disagreements += 1

        text = item.prompt()
        cheap = call(key, CHEAP, ANSWER_SYSTEM, text, ANSWER_TOKENS[CHEAP])
        cheap_cost = budget.record(CHEAP, cheap["usage"])
        frontier = call(key, FRONTIER, ANSWER_SYSTEM, text,
                        ANSWER_TOKENS[FRONTIER])
        frontier_cost = budget.record(FRONTIER, frontier["usage"])

        cheap_tok = int((cheap["usage"] or {}).get("prompt_tokens") or 0) + \
            int((cheap["usage"] or {}).get("completion_tokens") or 0)
        front_tok = int((frontier["usage"] or {}).get("prompt_tokens") or 0) + \
            int((frontier["usage"] or {}).get("completion_tokens") or 0)

        cost_a, served_a = routed_cost(ra, cheap_cost, frontier_cost)
        cost_b, served_b = routed_cost(rb, cheap_cost, frontier_cost)
        totals["baseline"] += frontier_cost
        totals[label_a] += cost_a
        totals[label_b] += cost_b
        tokens["baseline"] += front_tok
        tokens[label_a] += 0 if ra == "NO_MODEL" else (
            cheap_tok if ra == "SMALL" else front_tok)
        tokens[label_b] += 0 if rb == "NO_MODEL" else (
            cheap_tok if rb == "SMALL" else front_tok)

        # A prompt that needed an answer and got no call is a failure, and it is
        # counted per router rather than pooled.
        for label, route in ((label_a, ra), (label_b, rb)):
            if route == "NO_MODEL" and TIER_NAMES[item.tier] != "NO_MODEL":
                unanswered[label] += 1
            if route in ("NO_MODEL", "SMALL"):
                downgrades[label] += 1

        # One verdict, reused by whichever routers downgraded. Asking twice
        # would reintroduce exactly the variance this design removes.
        record = {"kind": item.kind, "asserted_tier": TIER_NAMES[item.tier],
                  f"route_{label_a}": ra, f"route_{label_b}": rb,
                  "agree": ra == rb,
                  "baseline_cost_usd": round(frontier_cost, 8),
                  f"cost_{label_a}": round(cost_a, 8),
                  f"cost_{label_b}": round(cost_b, 8),
                  "served_a": served_a, "served_b": served_b}

        needs_judge = ("SMALL" in (ra, rb)) and cheap["text"] and frontier["text"]
        if needs_judge and not budget.would_breach(JUDGE):
            cheap_is_a = rng.random() < 0.5
            x, y = (cheap, frontier) if cheap_is_a else (frontier, cheap)
            verdict = call(key, JUDGE, JUDGE_SYSTEM,
                           "REQUEST:" + chr(10) + item.request + chr(10) * 2
                           + "SOURCE MATERIAL:" + chr(10) + item.payload
                           + chr(10) * 2 + "ANSWER A:" + chr(10) + x["text"]
                           + chr(10) * 2 + "ANSWER B:" + chr(10) + y["text"],
                           JUDGE_TOKENS)
            budget.record(JUDGE, verdict["usage"])
            parsed = {}
            try:
                body = verdict["text"]
                parsed = json.loads(body[body.index("{"):body.rindex("}") + 1])
            except Exception:
                pass
            raw = str(parsed.get("verdict", "UNPARSED")).upper()
            cheap_worse = ((raw == "A_WORSE") == cheap_is_a
                           if raw in ("A_WORSE", "B_WORSE") else False)
            record.update({"judged": True, "verdict": raw,
                           "cheap_worse": cheap_worse,
                           "reason": str(parsed.get("reason", ""))[:120]})
            for label, route in ((label_a, ra), (label_b, rb)):
                if route == "SMALL" and cheap_worse:
                    worse[label] += 1
        else:
            record["judged"] = False

        handle.write(json.dumps(record, ensure_ascii=False) + chr(10))
        handle.flush()
        if index % 20 == 0:
            print(f"  {index}/{len(prompts)}  ${budget.spent:.3f}")
    handle.close()

    n = index - 1 if aborted else len(prompts)
    print(f"\nspent ${budget.spent:.4f} over {budget.calls} calls")
    if aborted:
        print(f"ABORTED: {aborted}")

    print(f"\n{'':<22}{'baseline':>12}{label_a:>14}{label_b:>14}")
    print(f"{'cost usd':<22}{totals['baseline']:>12.5f}"
          f"{totals[label_a]:>14.5f}{totals[label_b]:>14.5f}")
    base = totals["baseline"] or 1.0
    print(f"{'cost saving':<22}{'--':>12}"
          f"{100*(1-totals[label_a]/base):>13.1f}%"
          f"{100*(1-totals[label_b]/base):>13.1f}%")
    tbase = tokens["baseline"] or 1
    print(f"{'tokens':<22}{tokens['baseline']:>12}"
          f"{tokens[label_a]:>14}{tokens[label_b]:>14}")
    print(f"{'token saving':<22}{'--':>12}"
          f"{100*(1-tokens[label_a]/tbase):>13.1f}%"
          f"{100*(1-tokens[label_b]/tbase):>13.1f}%")
    print(f"{'unanswered':<22}{'--':>12}"
          f"{str(unanswered[label_a]) + '/' + str(n):>14}"
          f"{str(unanswered[label_b]) + '/' + str(n):>14}")
    print(f"{'downgrades':<22}{'--':>12}"
          f"{downgrades[label_a]:>14}{downgrades[label_b]:>14}")
    print(f"{'judged worse':<22}{'--':>12}"
          f"{worse[label_a]:>14}{worse[label_b]:>14}")
    print(f"\nrouting disagreements: {disagreements}/{n}")
    for label in (label_a, label_b):
        print(f"  {label:<10} " + "  ".join(
            f"{k} {v}" for k, v in sorted(splits[label].items())))
    if disagreements == 0:
        print("\nThe routers made identical decisions on every prompt. Any cost "
              "difference between them would be measurement error, and there "
              "is none by construction.")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
