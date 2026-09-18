"""What counts as a good router, defined before anything is trained.

Accuracy is the wrong headline for this problem and using it would hide the
only thing anybody cares about. A router that sends every turn to the largest
model is perfectly safe and saves nothing; one that sends everything to a script
saves everything and is useless. The interesting question is the trade between
those two, and accuracy does not express it.

So the headline is **savings at fixed regret**: what share of traffic came off
the large model, and what share of turns got less machine than they needed. Both
numbers, always together. Either one alone can be made to look excellent by a
model nobody would ship.

Everything is reported against the honest reference point — a host that calls
the large model every time — because that is what an inefficient agent actually
does, and it is what this project claims to improve on.
"""

from __future__ import annotations

import math
from typing import Sequence

from .schema import DEFAULT_REGRET_RATIO, Risk, Route, risk_cost, route_cost


def _rate(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def route_report(true: Sequence[Route], predicted: Sequence[Route],
                 regret_ratio: float = DEFAULT_REGRET_RATIO) -> dict:
    """The routing trade, stated as the pair of numbers that matter.

    ``downgraded`` is the gross saving: turns taken off the large model.
    ``regret`` is what that saving cost: turns that got less than they needed.
    ``mean_cost`` folds both into one comparable figure using the cost matrix
    declared in :mod:`littleleo.schema`, so two routers can be ranked without
    anyone choosing which half of the trade to emphasise.
    """
    total = len(true)
    if total == 0 or total != len(predicted):
        return {"n": total, "error": "empty or mismatched"}

    downgraded = sum(1 for p in predicted if p < Route.LARGE)
    regret = sum(1 for t, p in zip(true, predicted) if p < t)
    wasteful = sum(1 for t, p in zip(true, predicted) if p > t)
    exact = sum(1 for t, p in zip(true, predicted) if p == t)

    # What a host that always called the large model would have spent, and what
    # this router spends instead, in units of "large calls".
    unit = {Route.NO_MODEL: 0.0, Route.SMALL: 0.1, Route.LARGE: 1.0}
    spend = sum(unit[p] for p in predicted)

    return {
        "n": total,
        "accuracy": round(_rate(exact, total), 4),
        "downgraded": round(_rate(downgraded, total), 4),
        "regret": round(_rate(regret, total), 4),
        "wasteful": round(_rate(wasteful, total), 4),
        "mean_cost": round(sum(route_cost(t, p, regret_ratio)
                               for t, p in zip(true, predicted)) / total, 4),
        "large_call_spend": round(spend / total, 4),
        "spend_vs_always_large": round(spend / total, 4),
        "regret_ratio": regret_ratio,
    }


def risk_report(true: Sequence[Risk], predicted: Sequence[Risk]) -> dict:
    """Safety advice, scored on the error that actually hurts.

    ``p0_recall`` is the number to read: of the actions that really were
    destructive, how many did the router flag. Precision is reported beside it
    because a router that calls everything destructive has perfect recall and is
    worthless — its confirmations get switched off within a day.

    ``under_called`` counts every case where the advice was less alarmed than
    the truth, not just the P0 ones, because P1 called P2 is also a host acting
    with less care than it should have.
    """
    total = len(true)
    if total == 0 or total != len(predicted):
        return {"n": total, "error": "empty or mismatched"}

    real_p0 = [i for i, t in enumerate(true) if t == Risk.P0_DESTRUCTIVE]
    called_p0 = [i for i, p in enumerate(predicted) if p == Risk.P0_DESTRUCTIVE]
    caught = [i for i in real_p0 if predicted[i] == Risk.P0_DESTRUCTIVE]

    return {
        "n": total,
        "accuracy": round(_rate(sum(1 for t, p in zip(true, predicted) if t == p),
                                total), 4),
        "p0_support": len(real_p0),
        "p0_recall": round(_rate(len(caught), len(real_p0)), 4),
        "p0_precision": round(_rate(len(caught), len(called_p0)), 4),
        "p0_missed": len(real_p0) - len(caught),
        "under_called": sum(1 for t, p in zip(true, predicted) if p < t),
        "over_called": sum(1 for t, p in zip(true, predicted) if p > t),
        "mean_cost": round(sum(risk_cost(t, p)
                               for t, p in zip(true, predicted)) / total, 4),
    }


def expected_calibration_error(probabilities: Sequence[float],
                               outcomes: Sequence[int], bins: int = 10) -> dict:
    """How far a confidence is from being a probability.

    A threshold like "block above 0.85" is meaningless unless 0.85 means what it
    says. This bins predictions by confidence and compares each bin's claimed
    confidence to how often it was actually right. The gap, weighted by bin
    size, is the calibration error.

    Reported with the per-bin table, because a single ECE hides whether a model
    is overconfident everywhere or only at the top — and the top is where
    thresholds live.
    """
    total = len(probabilities)
    if total == 0 or total != len(outcomes):
        return {"n": total, "error": "empty or mismatched"}

    table = []
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, p in enumerate(probabilities)
                   if (p >= low and (p < high or (index == bins - 1 and p <= high)))]
        if not members:
            continue
        confidence = sum(probabilities[i] for i in members) / len(members)
        accuracy = sum(outcomes[i] for i in members) / len(members)
        error += (len(members) / total) * abs(confidence - accuracy)
        table.append({"bin": f"{low:.1f}-{high:.1f}", "n": len(members),
                      "confidence": round(confidence, 4),
                      "observed": round(accuracy, 4)})
    return {"n": total, "ece": round(error, 4), "bins": table}


def brier(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error of a probabilistic claim. Lower is better."""
    if not probabilities:
        return math.nan
    return round(sum((p - o) ** 2 for p, o in zip(probabilities, outcomes))
                 / len(probabilities), 4)


def compare(rows: dict[str, dict], key: str = "mean_cost") -> list[tuple[str, float]]:
    """Rank routers by one figure, cheapest first. Ties keep declaration order."""
    return sorted(((name, float(report.get(key, math.inf)))
                   for name, report in rows.items()), key=lambda item: item[1])


__all__ = ["brier", "compare", "expected_calibration_error", "risk_report",
           "route_report"]
