"""Phase 0: decide whether the teacher is good enough to distil from. ~$1.

Distillation copies the teacher's *distribution*, which means it copies the
teacher's mistakes and, worse, the teacher's confidence in them. A teacher that
is 0.99-confident and 70% right does not produce a 70%-right student; it produces
a student that is wrong with conviction, and every confidence-threshold defence
downstream silently stops working. That is not hypothetical -- it is precisely how
LL-149M failed: its wrong NO_MODEL calls sat at 0.998 and 0.989, sailed over a
0.90 floor, and six users got silence.

So before any traces are collected or any GPU-minute is spent, this script asks
one question on a hundred hand-labelled examples: *is this teacher worth copying?*
It gates on three things, and all three must pass:

1. **Agreement >= 0.85.** The teacher must broadly agree with a human.
2. **Confidence is informative.** Accuracy must rise with confidence. A teacher
   whose confidence carries no signal cannot transfer calibration, and calibration
   is the only reason to prefer soft labels over hard ones.
3. **Every class recalls >= 0.70.** A teacher that scores 0.90 overall by getting
   every SUCCESS right and every FATAL_HALT wrong is useless for the one decision
   that matters.

Fail any one and **do not distil**. There is no partial credit here.

Method
------
The teacher is asked for exactly one token -- ``A``, ``B`` or ``C`` -- with
``max_tokens=1``, ``logprobs=true`` and ``top_logprobs=20``. The single token is
not the point; the *distribution over the top 20* is. Renormalised onto the three
class letters it gives ``q``, which is the soft label the training objective in
``SPEC.md`` marginalises over. One token also means the call costs essentially the
prompt alone, which is what keeps the whole exercise near a dollar.

Labelling discipline
--------------------
Hand-label before running this. If the human column is filled in after seeing the
teacher's answers, this script measures anchoring, not agreement, and it has no
way to detect that -- so it is stated here rather than enforced.

The sample must also be hard. A hundred clean successes will score 0.95 agreement
and teach nothing, so a skewed sample is refused unless ``--allow-skewed`` is
passed deliberately.

    python sentinel/scripts/phase0_teacher_check.py --list-teachers
    python sentinel/scripts/phase0_teacher_check.py --input sentinel/data/phase0.jsonl --dry-run
    python sentinel/scripts/phase0_teacher_check.py --input sentinel/data/phase0.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from littleleo.providers import (describe, openrouter_key,  # noqa: E402
                                 openrouter_models, openrouter_pricing,
                                 verify_openrouter)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
# the contract, mirrored from SPEC.md section 3
#
# Ordered SUCCESS < TRANSIENT_ERROR < FATAL_HALT. The ordering is what makes the
# escalate-only invariant expressible: a model may move a verdict up this list
# and never down. These live here for Phase 0 only; they move to
# sentinel/schema.py when there is a model to write a contract for.
# ---------------------------------------------------------------------------

CLASSES = ("SUCCESS", "TRANSIENT_ERROR", "FATAL_HALT")

#: One letter per class. Single letters because the answer must fit in one
#: token for every tokeniser a Qwen deployment might use -- a class *name* would
#: split, and the first token of "TRANSIENT_ERROR" is shared with nothing useful.
LETTERS = {"A": "SUCCESS", "B": "TRANSIENT_ERROR", "C": "FATAL_HALT"}

# --- the gates. Declared before any teacher has been measured, on purpose. ---

#: Gate 1. Below this the teacher and the human are describing different tasks.
MIN_AGREEMENT = 0.85

#: Gate 2. Accuracy on the confident half must exceed accuracy on the unconfident
#: half by at least this much. Zero would pass a teacher whose confidence is
#: noise; this asks for a real, if modest, ordering signal.
MIN_CONFIDENCE_LIFT = 0.05

#: Gate 3. No class may fall below this. Macro-recall would let FATAL_HALT hide.
MIN_CLASS_RECALL = 0.70

#: Not a gate, a validity check. If this much of the top-20 mass lands outside
#: A/B/C the prompt is not landing and the numbers below describe nothing.
MAX_OFF_CLASS_MASS = 0.25

#: Also a validity check. Rows that yield no distribution are dropped, and a run
#: that silently drops a third of its sample is reporting on whatever is left --
#: which is not a random third. Scoring 90 of 100 is tolerable; 60 is a broken
#: measurement wearing a plausible number.
MIN_SCORED_SHARE = 0.90

#: Not a gate either. Refuses a sample that cannot discriminate: if one class is
#: this much of the file, agreement is measuring the prior, not the teacher.
MAX_CLASS_SHARE = 0.60

#: The student's window, from SPEC.md. Reported against the teacher's input so a
#: teacher that only succeeds by reading far past what the student will ever see
#: is visible now rather than in Phase 3.
STUDENT_WINDOW_TOKENS = 256
CHARS_PER_TOKEN = 4.0          # rough, and labelled as rough wherever printed

SYSTEM = (
    "You judge whether an autonomous agent should trust the result of a tool "
    "call it just made. Answer with exactly one letter and nothing else.\n"
    "A = SUCCESS. The call did what it said. The agent should continue.\n"
    "B = TRANSIENT_ERROR. It failed but a retry could plausibly succeed.\n"
    "C = FATAL_HALT. It failed permanently, OR it reported success while "
    "producing a corrupt or incomplete result. The agent must stop.\n"
    "A zero exit code with a traceback, a truncated write, or an empty result "
    "that should not be empty are all C."
)

USER = ("tool: {tool}\n"
        "arguments: {args}\n"
        "--- output ---\n{output}\n--- end output ---\n"
        "One letter: A, B or C.")


# ---------------------------------------------------------------------------
# the call
# ---------------------------------------------------------------------------

def ask(key: str, model: str, tool: str, args: str, output: str,
        provider: str = "", retries: int = 3) -> dict:
    """One token from the teacher, and the distribution behind it.

    Returns ``{"probs": {}, "off_class": float, "reason": str, "served_by": str,
    "usage": {}, "error": ""}``. ``probs`` is empty when nothing parseable came
    back, and ``reason`` says which kind of nothing -- the caller counts those
    separately, because a missing distribution and a distribution spent on prose
    are different faults with different fixes, and averaging them together hides
    both.

    ``provider.require_parameters`` is not optional. OpenRouter load-balances one
    model id across several upstream providers, and they do not all implement
    ``top_logprobs``; one that does not returns a perfectly good answer with an
    empty ``logprobs`` block. Without this flag roughly half the rows in a trial
    run came back with no distribution at all -- silently, since the *letter*
    was right. A soft-label pipeline that quietly collects hard labels for half
    its corpus is the exact failure this phase exists to catch, so it is pinned
    here rather than left to routing luck.
    """
    routing: dict = {"require_parameters": True}
    if provider:
        # Pinned, and fallbacks off: see --provider on why comparability across
        # rows matters more here than availability.
        routing["order"] = [provider]
        routing["allow_fallbacks"] = False
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": USER.format(
                         tool=tool, args=args, output=output)}],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
        "provider": routing,
    }).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})

    last = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.load(response)
            choice = body["choices"][0]
            probs, off_class, reason = _distribution(choice)
            return {"probs": probs, "off_class": off_class, "reason": reason,
                    "served_by": str(body.get("provider") or "?"),
                    "usage": body.get("usage") or {}, "error": ""}
        except urllib.error.HTTPError as error:
            # The body can echo the request, including the header. Status only.
            last = f"HTTP {error.code}"
            if error.code in (429, 500, 502, 503, 529):
                time.sleep(2 ** attempt)
                continue
            break
        except Exception as error:                              # noqa: BLE001
            last = type(error).__name__
            time.sleep(2 ** attempt)
    return {"probs": {}, "off_class": 0.0, "reason": "call_failed",
            "served_by": "?", "usage": {}, "error": last}


def _distribution(choice: dict) -> tuple[dict, float, str]:
    """Renormalise the top-20 onto the three classes.

    Mass landing outside A/B/C is measured, not discarded quietly: it is the
    single best signal that the prompt is not landing. A teacher that spends 40%
    of its probability on prose has not understood the question, and every
    agreement figure computed from the remaining 60% is describing a coin flip
    with extra steps.
    """
    slots = (choice.get("logprobs") or {}).get("content") or []
    if not slots:
        # An answer with no distribution behind it. Distinct from prose, and
        # almost always a provider that ignored top_logprobs -- see ask().
        return {}, 0.0, "no_logprobs"
    entries = slots[0].get("top_logprobs") or []
    by_class: dict[str, float] = defaultdict(float)
    total = 0.0
    for entry in entries:
        token = str(entry.get("token", "")).strip().upper()
        probability = math.exp(float(entry.get("logprob", -99.0)))
        total += probability
        # " A", "A", "A." all mean A. Anything longer is prose, not an answer.
        if token[:1] in LETTERS and len(token) <= 2:
            by_class[LETTERS[token[:1]]] += probability

    on_class = sum(by_class.values())
    if on_class <= 0.0:
        return {}, 1.0, "off_class"
    off_class = max(0.0, 1.0 - on_class / total) if total > 0 else 0.0
    return ({name: by_class.get(name, 0.0) / on_class for name in CLASSES},
            off_class, "")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def expected_calibration_error(scored: list[dict], bins: int = 10) -> float:
    """Standard binned ECE over the teacher's top-class probability.

    Reported here so Phase 3's calibration gate has a teacher-side number to be
    compared against. A student cannot be better calibrated than what it was
    taught, so this is the ceiling on that gate.
    """
    buckets: list[list[dict]] = [[] for _ in range(bins)]
    for row in scored:
        index = min(bins - 1, int(row["confidence"] * bins))
        buckets[index].append(row)
    total = len(scored)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        accuracy = sum(r["correct"] for r in bucket) / len(bucket)
        confidence = sum(r["confidence"] for r in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(accuracy - confidence)
    return error


def risk_coverage(scored: list[dict]) -> list[tuple[float, float]]:
    """Accuracy when the least-confident answers are abandoned.

    This is the curve a host actually operates on: it does not ask "how good is
    the model", it asks "if I only act on the top 70%, what am I acting on".
    """
    ordered = sorted(scored, key=lambda r: -r["confidence"])
    points = []
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        kept = ordered[:max(1, int(len(ordered) * coverage))]
        points.append((coverage, sum(r["correct"] for r in kept) / len(kept)))
    return points


def confidence_lift(scored: list[dict]) -> tuple[float, float, float]:
    """Accuracy on the confident half minus accuracy on the unconfident half.

    A median split rather than a correlation because the gate is a decision, and
    a decision wants a number in the same units as the thing being decided. It
    also degrades honestly: a teacher that is uniformly confident produces two
    statistically identical halves and a lift near zero, which is the correct
    verdict for a teacher whose confidence means nothing.
    """
    ordered = sorted(scored, key=lambda r: r["confidence"])
    half = len(ordered) // 2
    low, high = ordered[:half], ordered[half:]
    accuracy_low = sum(r["correct"] for r in low) / max(1, len(low))
    accuracy_high = sum(r["correct"] for r in high) / max(1, len(high))
    return accuracy_high - accuracy_low, accuracy_low, accuracy_high


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    """Read and validate the hand-labelled file, refusing anything ambiguous."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{number}: not JSON ({error.msg})")
        label = str(row.get("human", "")).strip().upper()
        if label not in CLASSES:
            raise SystemExit(
                f"{path}:{number}: 'human' is {label or '(missing)'}, expected "
                f"one of {', '.join(CLASSES)}. Every row needs a hand label -- "
                f"an unlabelled row cannot be scored, and dropping it silently "
                f"would bias the sample toward the easy cases.")
        row["human"] = label
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def check_sample(rows: list[dict], allow_skewed: bool) -> None:
    """Refuse a sample that would score well while proving nothing."""
    counts = defaultdict(int)
    for row in rows:
        counts[row["human"]] += 1
    print("hand labels:")
    for name in CLASSES:
        share = counts[name] / len(rows)
        print(f"  {name:<16} {counts[name]:>4}  ({share:.0%})")

    missing = [n for n in CLASSES if counts[n] == 0]
    if missing:
        raise SystemExit(
            f"\nno examples of {', '.join(missing)}. Class recall cannot be "
            f"measured for a class that is absent, so gate 3 would pass by "
            f"vacancy.")

    worst = max(counts.values()) / len(rows)
    if worst > MAX_CLASS_SHARE and not allow_skewed:
        dominant = max(counts, key=counts.get)
        raise SystemExit(
            f"\n{dominant} is {worst:.0%} of the sample, over the "
            f"{MAX_CLASS_SHARE:.0%} ceiling. A teacher can score "
            f"{worst:.2f} agreement on this file by answering {dominant} to "
            f"everything, so the agreement gate would measure the prior rather "
            f"than the teacher. Over-sample the hard cases -- exit-0-with-"
            f"traceback, empty-but-successful, partial writes -- or pass "
            f"--allow-skewed if the skew is deliberate and understood.")


# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "sentinel" / "data" / "phase0.jsonl",
                        help="hand-labelled JSONL; see the module docstring")
    # The non-thinking 2507 instruct build: Apache-2.0 open weights, $0.0875/Mtok
    # at the time of writing, and it answers in one token. A "-thinking" build
    # cannot be used at all here -- it spends its first token on reasoning, so
    # max_tokens=1 returns nothing classifiable.
    parser.add_argument("--model", default="qwen/qwen3-235b-a22b-2507",
                        help="teacher model id (Apache-2.0, non-thinking)")
    # OpenRouter spreads one model id across upstream providers that quantise
    # differently, so two rows can be scored by two different numeric builds of
    # "the same" model. For agreement that is noise; for the calibration gate it
    # is not, because the confidences being compared then come from different
    # models. Pin one provider when the numbers are going in a report.
    parser.add_argument("--provider", default="",
                        help="pin one upstream provider (e.g. Parasail) so "
                             "confidences are comparable across rows")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "sentinel" / "data" / "phase0-teacher.jsonl")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "sentinel" / "data" / "phase0-report.json")
    parser.add_argument("--ceiling", type=float, default=2.00,
                        help="hard USD ceiling; the run stops rather than crossing it")
    parser.add_argument("--max-output-chars", type=int, default=4000,
                        help="truncate tool output before sending, to bound cost")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N rows (0 = all)")
    parser.add_argument("--allow-skewed", action="store_true",
                        help="proceed despite a dominant class; see check_sample")
    parser.add_argument("--dry-run", action="store_true",
                        help="estimate cost and validate the file, spend nothing")
    parser.add_argument("--list-teachers", action="store_true",
                        help="list Qwen models currently offered, and stop")
    args = parser.parse_args(argv[1:])

    key = openrouter_key(ROOT)
    if not key:
        print("no OpenRouter key found (Openrouter.txt or OPENROUTER_API_KEY)",
              file=sys.stderr)
        return 2
    # describe() returns facts about the key. The key itself is never printed.
    print(f"key: {describe(key)}")

    if args.list_teachers:
        for model_id in openrouter_models(key, "qwen"):
            print(f"  {model_id}")
        print("\nPick an Apache-2.0 Qwen build with open weights: a teacher "
              "under restrictive terms\ncontaminates an Apache-2.0 release -- "
              "see SPEC.md section 4. Skip any '-thinking'\nbuild, which spends "
              "its first token reasoning and returns nothing classifiable\n"
              "under max_tokens=1.")
        return 0

    rows = load(args.input)
    if args.limit:
        rows = rows[:args.limit]
    print(f"\n{len(rows)} hand-labelled rows from {args.input}\n")
    check_sample(rows, args.allow_skewed)

    # A teacher that needs far more context than the student will ever get is a
    # teacher whose labels the student cannot learn to reproduce. Cheap to see
    # now; expensive to discover after 20k traces.
    long_rows = sum(1 for r in rows
                    if len(str(r.get("output", ""))) / CHARS_PER_TOKEN
                    > STUDENT_WINDOW_TOKENS)
    if long_rows:
        print(f"\nnote: {long_rows}/{len(rows)} outputs are longer than the "
              f"student's {STUDENT_WINDOW_TOKENS}-token window (rough estimate "
              f"at {CHARS_PER_TOKEN:.0f} chars/token). The teacher sees them "
              f"whole; the student will not. If agreement comes mostly from "
              f"these, the labels may not be learnable at 256 tokens.")

    pricing = openrouter_pricing(key, [args.model])
    rates = pricing.get(args.model)
    if not rates:
        print(f"\n{args.model} is not offered. Try --list-teachers.",
              file=sys.stderr)
        return 2
    prompt_chars = sum(min(len(str(r.get("output", ""))), args.max_output_chars)
                       + len(str(r.get("tool", "")))
                       + len(str(r.get("args", ""))) for r in rows)
    estimate = ((prompt_chars / CHARS_PER_TOKEN + len(SYSTEM) / CHARS_PER_TOKEN
                 * len(rows)) * rates["prompt_per_mtok_usd"] / 1e6)
    print(f"\nteacher: {args.model}")
    print(f"  prompt ${rates['prompt_per_mtok_usd']:.4f}/Mtok, "
          f"context {rates['context']}")
    print(f"  estimated spend: ${estimate:.4f} over {len(rows)} calls "
          f"(ceiling ${args.ceiling:.2f})")

    if args.dry_run:
        print("\ndry run: file validated, nothing sent.")
        return 0
    if estimate > args.ceiling:
        print(f"\nestimate exceeds the ceiling. Raise --ceiling deliberately "
              f"or cut the sample with --limit.", file=sys.stderr)
        return 2

    balance = verify_openrouter(key)
    if not balance.get("ok"):
        print(f"\nkey did not authenticate: {balance.get('reason')}",
              file=sys.stderr)
        return 2
    remaining = balance.get("remaining_usd")
    if remaining is not None:
        print(f"  balance remaining: ${float(remaining):.2f}")

    # --- score -----------------------------------------------------------
    scored: list[dict] = []
    unparsed: dict[str, list[int]] = defaultdict(list)
    served_by: dict[str, int] = defaultdict(int)
    off_class_total = 0.0
    spent = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    handle = args.out.open("w", encoding="utf-8")

    for index, row in enumerate(rows, 1):
        if spent > args.ceiling:
            print(f"\nceiling reached after {index - 1} rows; stopping.")
            break
        output = str(row.get("output", ""))[:args.max_output_chars]
        result = ask(key, args.model, str(row.get("tool", "")),
                     str(row.get("args", "")), output, args.provider)
        served_by[result["served_by"]] += 1

        usage = result["usage"]
        spent += (int(usage.get("prompt_tokens") or 0)
                  * rates["prompt_per_mtok_usd"] / 1e6
                  + int(usage.get("completion_tokens") or 0)
                  * rates["completion_per_mtok_usd"] / 1e6)

        record = {"index": index, "tool": row.get("tool", ""),
                  "human": row["human"], "note": row.get("note", ""),
                  "error": result["error"], "off_class": result["off_class"],
                  "served_by": result["served_by"]}
        if not result["probs"]:
            unparsed[result["reason"]].append(index)
            record["teacher"] = None
            record["reason"] = result["reason"]
        else:
            probs = result["probs"]
            teacher = max(probs, key=probs.get)
            off_class_total += result["off_class"]
            record.update({"teacher": teacher, "probs": probs,
                           "confidence": probs[teacher],
                           "correct": int(teacher == row["human"])})
            scored.append(record)
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        if index % 10 == 0:
            print(f"  {index}/{len(rows)}  ${spent:.4f}")
    handle.close()

    if not scored:
        reasons = ", ".join(f"{name}={len(rows)}"
                            for name, rows in unparsed.items()) or "unknown"
        print(f"\nnothing parseable came back ({reasons}). Nothing below could "
              f"be computed.", file=sys.stderr)
        return 1

    # --- gates -----------------------------------------------------------
    agreement = sum(r["correct"] for r in scored) / len(scored)
    lift, accuracy_low, accuracy_high = confidence_lift(scored)
    ece = expected_calibration_error(scored)

    recall = {}
    for name in CLASSES:
        actual = [r for r in scored if r["human"] == name]
        recall[name] = (sum(r["correct"] for r in actual) / len(actual)
                        if actual else 0.0)

    mean_off_class = off_class_total / len(scored)

    print(f"\n{'=' * 62}\nteacher: {args.model}")
    print(f"scored {len(scored)}/{len(rows)} rows, spent ${spent:.4f}")
    for reason, indices in sorted(unparsed.items()):
        print(f"  dropped ({reason}): {len(indices)} rows {indices[:8]}")
    if unparsed.get("no_logprobs"):
        print("  -> a provider ignored top_logprobs despite require_parameters;"
              "\n     pin a known-good one with --provider (see the list above "
              "each row).")
    if len(served_by) > 1:
        # Not fatal, but the calibration numbers below are then a blend.
        blend = ", ".join(f"{name} x{count}"
                          for name, count in sorted(served_by.items()))
        print(f"\nserved by {len(served_by)} providers: {blend}")
        print("  -> confidences come from differently-quantised builds. Fine "
              "for agreement,\n     not for the calibration figures. Re-run "
              "with --provider before quoting ECE.")

    print("\nconfusion (rows = human, columns = teacher):")
    print(f"  {'':<16}" + "".join(f"{n[:9]:>11}" for n in CLASSES))
    for truth in CLASSES:
        line = f"  {truth:<16}"
        for predicted in CLASSES:
            line += f"{sum(1 for r in scored if r['human'] == truth and r['teacher'] == predicted):>11}"
        print(line)

    print("\nrisk-coverage (accuracy after dropping the least confident):")
    for coverage, accuracy in risk_coverage(scored):
        print(f"  coverage {coverage:.0%}   accuracy {accuracy:.3f}")
    print(f"\nECE: {ece:.4f}   mean off-class mass: {mean_off_class:.3f}")

    gates = {
        "agreement": (agreement >= MIN_AGREEMENT,
                      f"agreement {agreement:.3f} >= {MIN_AGREEMENT}"),
        "confidence_informative": (
            lift >= MIN_CONFIDENCE_LIFT,
            f"confident half {accuracy_high:.3f} vs unconfident half "
            f"{accuracy_low:.3f}, lift {lift:+.3f} >= {MIN_CONFIDENCE_LIFT}"),
        "class_recall": (
            min(recall.values()) >= MIN_CLASS_RECALL,
            "  ".join(f"{n}={recall[n]:.3f}" for n in CLASSES)
            + f"  (floor {MIN_CLASS_RECALL})"),
        "prompt_landed": (
            mean_off_class <= MAX_OFF_CLASS_MASS,
            f"off-class mass {mean_off_class:.3f} <= {MAX_OFF_CLASS_MASS}"),
        "answers_usable": (
            len(scored) / len(rows) >= MIN_SCORED_SHARE,
            f"{len(scored)}/{len(rows)} rows yielded a distribution "
            f"({len(scored) / len(rows):.0%} >= {MIN_SCORED_SHARE:.0%})"),
    }

    print("\ngates:")
    for name, (passed, detail) in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<24} {detail}")

    verdict = all(passed for passed, _ in gates.values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "teacher": args.model, "n_scored": len(scored), "n_rows": len(rows),
        "dropped": {reason: len(i) for reason, i in unparsed.items()},
        "served_by": dict(served_by), "provider_pinned": args.provider or None,
        "spend_usd": round(spent, 4),
        "agreement": round(agreement, 4), "ece": round(ece, 4),
        "confidence_lift": round(lift, 4),
        "accuracy_low_half": round(accuracy_low, 4),
        "accuracy_high_half": round(accuracy_high, 4),
        "mean_off_class_mass": round(mean_off_class, 4),
        "recall": {n: round(recall[n], 4) for n in CLASSES},
        "risk_coverage": [[c, round(a, 4)] for c, a in risk_coverage(scored)],
        "gates": {n: bool(p) for n, (p, _) in gates.items()},
        "passes": verdict,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'=' * 62}")
    if verdict:
        print("PASS -- this teacher is worth distilling from.")
        print("Next: SPEC.md section 9, Phase 1. Establish how many *real* "
              "traces\nLeo can supply before collecting anything. That is the "
              "binding constraint,\nand distillation does nothing about it.")
    else:
        print("FAIL -- do not distil from this teacher.")
        print("A teacher that fails any gate above produces a student that is "
              "wrong\nwith conviction, which is the one failure this design "
              "exists to avoid.\nTry another Qwen build (--list-teachers), or "
              "sharpen the prompt if the\noff-class mass is high.")
    print(f"per-row detail: {args.out}\nreport: {args.report}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
