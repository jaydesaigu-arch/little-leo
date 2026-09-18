"""Measure what Little Leo actually saves, in tokens and dollars, against quality.

This is the only experiment in the project that answers the question a reader of
the README will ask: *does routing save anything worth having?* Everything else
measures whether the model reproduces labels. This measures money.

Design
------

Every prompt is sent to **both** models, so the counterfactual is observed rather
than assumed. Then:

* **Baseline** — every prompt served by the frontier model. What an unrouted
  agent spends.
* **Routed** — the tier LL-22M *actually chose*, not the tier the corpus asserts.
  ``NO_MODEL`` costs nothing because no call is made; ``SMALL`` is served by the
  cheap model; ``LARGE`` by the frontier model.

Routing with the model's own decisions is the whole point. Using the corpus
labels would measure the generator that produced them, and the generator is not
what anybody is being asked to install.

Quality is judged only where routing made a difference — where LL sent a prompt
somewhere cheaper than the baseline — because that is the only place a saving
could have cost something. A third-vendor judge sees both answers blind with the
order randomised.

Two figures come out, and neither is meaningful alone: **percent saved** and
**percent of downgrades that were materially worse**. A router that saves 90%
by ruining a third of the answers is not a saving, and one that saves 2% safely
is not worth installing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from littleleo.grounding_fixture import ITEMS, TIER_NAMES  # noqa: E402
from littleleo.pipeline import LittleLeoRouter  # noqa: E402
from littleleo.providers import openrouter_key, openrouter_pricing  # noqa: E402

CHEAP = "google/gemini-2.5-flash"
FRONTIER = "openai/gpt-5.4"
JUDGE = "anthropic/claude-opus-4.5"   # neutral to both contestants

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

#: Per model, because a model that reasons internally needs room for the
#: reasoning *and* the answer. A shared cap silently deletes the answers of
#: whichever model thinks hardest, which biases the result toward it.
ANSWER_TOKENS = {CHEAP: 1600, FRONTIER: 1200}
JUDGE_TOKENS = 250

ANSWER_SYSTEM = (
    "You are an assistant embedded in a developer's tooling. Answer the user's "
    "request directly and completely, using the source material provided. Do "
    "not ask clarifying questions; if something is ambiguous, state your "
    "assumption in one line and proceed."
)

JUDGE_SYSTEM = (
    "You are evaluating two answers to the same request. The source material is "
    "included. Decide whether one is MATERIALLY worse for the person who asked."
    "\n\n'Materially worse' means it would change what they do next: it is "
    "factually wrong about the source material, misses the point of the "
    "request, ignores a stated constraint, or fails to deliver what was asked "
    "for.\n\nIgnore entirely: style, tone, length, formatting, and whether an "
    "answer is cut off at the end. Both were generated under token limits and "
    "truncation is an artefact, not a quality difference. Judge only the "
    "content present.\n\nIf both would serve the person equally well, say "
    "EQUIVALENT.\n\nReply with one line of JSON and nothing else:\n"
    '{"verdict": "A_WORSE" | "B_WORSE" | "EQUIVALENT", "reason": "<15 words max>"}'
)


class Budget:
    def __init__(self, ceiling: float, pricing: dict):
        self.ceiling, self.pricing = ceiling, pricing
        self.spent, self.calls = 0.0, 0
        self.by_model: dict[str, float] = defaultdict(float)
        self.tokens: dict[str, dict] = defaultdict(lambda: {"in": 0, "out": 0})

    def cost_of(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        rates = self.pricing.get(model)
        if not rates:
            return 0.0
        return (prompt_tokens * rates["prompt_per_mtok_usd"] / 1e6
                + completion_tokens * rates["completion_per_mtok_usd"] / 1e6)

    def would_breach(self, model: str) -> bool:
        worst = self.cost_of(model, 2500, ANSWER_TOKENS.get(model, 1200))
        return (self.spent + worst) > self.ceiling

    def record(self, model: str, usage: dict) -> float:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost = self.cost_of(model, prompt_tokens, completion_tokens)
        self.spent += cost
        self.calls += 1
        self.by_model[model] += cost
        self.tokens[model]["in"] += prompt_tokens
        self.tokens[model]["out"] += completion_tokens
        return cost


def call(key: str, model: str, system: str, user: str, max_tokens: int,
         retries: int = 3) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    last = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.load(response)
            choice = body["choices"][0]
            return {"text": choice["message"].get("content") or "",
                    "usage": body.get("usage") or {},
                    "finish": choice.get("finish_reason", "")}
        except urllib.error.HTTPError as error:
            last = f"HTTP {error.code}"
            if error.code in (429, 500, 502, 503, 529):
                time.sleep(2 ** attempt)
                continue
            break
        except Exception as error:  # noqa: BLE001
            last = type(error).__name__
            time.sleep(2 ** attempt)
    return {"text": "", "usage": {}, "finish": f"error:{last}"}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ceiling", type=float, default=3.50)
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--onnx", type=Path,
                        default=ROOT / "artifacts" / "ll-22m" / "onnx")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data" / "savings.jsonl")
    args = parser.parse_args(argv[1:])

    key = openrouter_key(ROOT)
    if not key:
        print("no OpenRouter key", file=sys.stderr)
        return 2
    pricing = openrouter_pricing(key, [CHEAP, FRONTIER, JUDGE])
    for model in (CHEAP, FRONTIER, JUDGE):
        rates = pricing[model]
        print(f"  {model:<30} in ${rates['prompt_per_mtok_usd']:.4f}  "
              f"out ${rates['completion_per_mtok_usd']:.4f}")

    router = LittleLeoRouter(args.onnx, quantised=False)  # FP32: the gated artifact
    budget = Budget(args.ceiling, pricing)
    rng = random.Random(args.seed)

    prompts = [i for i in ITEMS for _ in range(args.repeats)]
    rng.shuffle(prompts)
    print(f"\n{len(prompts)} prompts, ceiling ${args.ceiling:.2f}\n")

    handle = args.out.open("w", encoding="utf-8")
    aborted = ""
    for index, item in enumerate(prompts, 1):
        if budget.would_breach(CHEAP) or budget.would_breach(FRONTIER):
            aborted = f"ceiling reached after {index - 1} prompts"
            break

        # LL-22M's own decision, on the request as a user would type it.
        decision = router.route(item.request)
        chosen = decision["route"]

        text = item.prompt()
        cheap = call(key, CHEAP, ANSWER_SYSTEM, text, ANSWER_TOKENS[CHEAP])
        cheap_cost = budget.record(CHEAP, cheap["usage"])
        frontier = call(key, FRONTIER, ANSWER_SYSTEM, text, ANSWER_TOKENS[FRONTIER])
        frontier_cost = budget.record(FRONTIER, frontier["usage"])

        # What each policy would have paid for this prompt.
        baseline_cost = frontier_cost
        if chosen == "NO_MODEL":
            routed_cost, served_by = 0.0, "none"
        elif chosen == "SMALL":
            routed_cost, served_by = cheap_cost, CHEAP
        else:
            routed_cost, served_by = frontier_cost, FRONTIER

        record = {
            "kind": item.kind, "asserted_tier": TIER_NAMES[item.tier],
            "ll_route": chosen, "ll_confidence": decision["confidence"],
            "ll_source": decision.get("source"),
            "served_by": served_by,
            "baseline_cost_usd": round(baseline_cost, 8),
            "routed_cost_usd": round(routed_cost, 8),
            "cheap_in": cheap["usage"].get("prompt_tokens"),
            "cheap_out": cheap["usage"].get("completion_tokens"),
            "frontier_in": frontier["usage"].get("prompt_tokens"),
            "frontier_out": frontier["usage"].get("completion_tokens"),
            "cheap_finish": cheap["finish"], "frontier_finish": frontier["finish"],
        }

        # Judge only where routing actually changed what the user would receive.
        if chosen == "LARGE":
            record.update({"judged": False, "downgraded": False})
        elif not cheap["text"] or not frontier["text"]:
            record.update({"judged": False, "downgraded": True,
                           "cheap_worse": bool(not cheap["text"] and frontier["text"]),
                           "reason": "empty answer"})
        elif chosen == "NO_MODEL":
            # Nothing was served at all, so the comparison is against silence:
            # the frontier answer is what the user would have had.
            record.update({"judged": False, "downgraded": True,
                           "cheap_worse": None,
                           "reason": "no call made; not comparable"})
        else:
            if budget.would_breach(JUDGE):
                record.update({"judged": False, "downgraded": True,
                               "reason": "judge skipped, ceiling"})
            else:
                cheap_is_a = rng.random() < 0.5
                a, b = (cheap, frontier) if cheap_is_a else (frontier, cheap)
                verdict = call(key, JUDGE, JUDGE_SYSTEM,
                               "REQUEST:" + chr(10) + item.request + chr(10) * 2
                               + "SOURCE MATERIAL:" + chr(10) + item.payload + chr(10) * 2
                               + "ANSWER A:" + chr(10) + a["text"] + chr(10) * 2
                               + "ANSWER B:" + chr(10) + b["text"], JUDGE_TOKENS)
                budget.record(JUDGE, verdict["usage"])
                parsed = {}
                try:
                    body = verdict["text"]
                    parsed = json.loads(body[body.index("{"):body.rindex("}") + 1])
                except Exception:
                    pass
                raw = str(parsed.get("verdict", "UNPARSED")).upper()
                worse = ((raw == "A_WORSE") == cheap_is_a
                         if raw in ("A_WORSE", "B_WORSE") else False)
                record.update({"judged": True, "downgraded": True,
                               "verdict": raw, "cheap_worse": worse,
                               "reason": str(parsed.get("reason", ""))[:120]})
        handle.write(json.dumps(record, ensure_ascii=False) + chr(10))
        handle.flush()
        if index % 20 == 0:
            print(f"  {index}/{len(prompts)}  ${budget.spent:.3f}")
    handle.close()

    print(f"\nspent ${budget.spent:.4f} over {budget.calls} calls")
    for model, cost in sorted(budget.by_model.items()):
        t = budget.tokens[model]
        print(f"  {model:<30} ${cost:.4f}   {t['in']:>7} in  {t['out']:>7} out")
    if aborted:
        print(f"ABORTED: {aborted}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
