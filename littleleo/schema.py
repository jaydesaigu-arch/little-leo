"""The contract: what Little Leo decides, and what it costs to decide wrongly.

This module is the specification. Everything else in the repository — the data
generator, the model, the evaluation, the exported runtime — is written against
what is declared here, and none of it may quietly widen the contract.

Two call sites, one encoder
---------------------------

A router cannot answer both of its questions at the same moment.

``PRE``
    Runs **before** any model is called, on the user's turn alone. Answers "how
    much machine does this need?" — which is the whole point, because by the
    time a large model has answered, routing it is moot.

``ACT``
    Runs **after** a model has proposed a tool call, on the turn plus that
    proposal. Answers "how dangerous is this?" — which cannot be asked earlier,
    because the proposal does not exist yet.

Both go through the same weights, distinguished by a marker token. One file to
ship, one thing to version.

The safety invariant
--------------------

**Little Leo may only ever make a host more cautious or more expensive. Never
less.**

This is not a guideline, it is what makes the component safe to adopt. A host
that follows it degrades, when the router is deleted, broken, or successfully
attacked, to exactly the behaviour it had before installing it: use the big
model, ask before doing anything destructive. Concretely:

* ``route`` is a *hint*. Absent, low-confidence or malformed, the host uses its
  own default, which should be the most capable engine it has.
* ``risk`` may raise a tier and may never lower one. The host's own deterministic
  rules decide whether something is destructive; this model only adds caution.
* ``abstain`` exists so the model can say "I do not recognise this", and a host
  that honours it is never routing on a guess.

A classifier is not a security boundary. Anything that treats this one as the
thing standing between a user and ``rm -rf`` has misread the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

# ---------------------------------------------------------------------------
# call sites
# ---------------------------------------------------------------------------

SITE_PRE = "PRE"
SITE_ACT = "ACT"
SITES = (SITE_PRE, SITE_ACT)

#: Prepended to the encoded text so one set of weights serves both questions.
SITE_MARKERS = {SITE_PRE: "[PRE]", SITE_ACT: "[ACT]"}


# ---------------------------------------------------------------------------
# head 1: route
# ---------------------------------------------------------------------------

class Route(IntEnum):
    """How much machine a turn needs.

    Deliberately abstract. A published router cannot know what engines a host
    has, so it does not name them: it reports a tier and the host maps it to
    whatever it actually runs. A host with only one model can still use the
    tier to decide whether to call it at all.
    """

    NO_MODEL = 0   # answerable by a script, a cache, a lookup or a constant
    SMALL = 1      # a cheap model is enough: rephrase, extract, format, classify
    LARGE = 2      # genuine reasoning, planning, ambiguity, or long dependencies


ROUTE_NAMES = {r: r.name for r in Route}


# ---------------------------------------------------------------------------
# head 2: risk
# ---------------------------------------------------------------------------

class Risk(IntEnum):
    """How much damage a proposed action could do. Advisory only."""

    P2_READONLY = 0    # observes; leaves nothing changed
    P1_MUTABLE = 1     # changes something recoverable
    P0_DESTRUCTIVE = 2  # irreversible, or reaches outside the machine


RISK_NAMES = {r: r.name for r in Risk}


# ---------------------------------------------------------------------------
# heads 3 and 4: gate and abstain
# ---------------------------------------------------------------------------

#: Head 3. Calibrated probability that a human should confirm before acting.
GATE = "gate"

#: Head 4. Calibrated probability that this input is unlike anything the model
#: was trained on. A router that cannot say "I do not know" will route on a
#: guess, and the host has no way to tell that apart from a confident answer.
ABSTAIN = "abstain"


# ---------------------------------------------------------------------------
# the cost matrix
#
# Written before any model is trained, on purpose. Thresholds derived from a
# cost matrix chosen afterwards are thresholds chosen to flatter the model.
# ---------------------------------------------------------------------------

#: How many wasted large-model calls one bad answer is worth.
#:
#: This single number decides the router's whole personality. At 1 the model
#: would happily downgrade half the traffic and hand back rubbish; at 1000 it
#: would never downgrade anything and save nothing. 20 says: sending twenty
#: easy turns to an expensive model is a fair price for not botching one hard
#: one. Hosts with different economics should re-derive their thresholds with a
#: different value, which is why it is a named constant and not a magic number
#: buried in the evaluation.
DEFAULT_REGRET_RATIO = 20.0

#: Token ceiling for both call sites. It **truncates**; it does not pad. Every
#: encoder in this repository tokenises with ``padding="longest"``, so a turn
#: costs what its own length costs and a turn shorter than the ceiling is
#: unaffected by where the ceiling sits. Raising it is free for short traffic
#: and is paid for only by turns that are genuinely long.
#:
#: It lives in the contract rather than beside the training code because it is a
#: property of the published artifact: a host tokenising differently from the
#: exporter gets different answers from the same weights.
#:
#: **Why 256 and not 64.** Under this corpus the decisive clause is always the
#: *last* element of a turn — ``_sentence(lead, obj, collapser)`` and every
#: sibling in ``synth.py`` put the collapser, qualifier or transform at the end.
#: Truncation therefore does not merely degrade the input, it removes precisely
#: the span that carries the label, leaving a bare lead and object that is
#: ambiguous across all three tiers by construction. For this architecture a
#: clipped turn is worse than a short one: it is a confidently mislabelled one.
#:
#: 64 was never measured as sufficient. It was inherited from a corpus whose
#: longest example is 47 tokens, and it silently clipped real traffic that the
#: corpus does not contain. Raising it removes that failure mode; it does not
#: teach the model about long turns, which no ceiling can do — see the model
#: card's limitation on input length.
MAX_LENGTH = 256


def route_cost(true: Route, predicted: Route,
               regret_ratio: float = DEFAULT_REGRET_RATIO) -> float:
    """Cost of predicting *predicted* when the truth was *true*.

    Downgrading is the expensive mistake: the user gets a worse answer and
    usually cannot tell why. Upgrading merely wastes tokens and latency. The
    two are priced in the same unit — "wasted large calls" — so they can be
    summed into one number that means something.
    """
    if predicted == true:
        return 0.0
    steps = abs(int(true) - int(predicted))
    if predicted < true:
        return regret_ratio * steps      # too little machine: quality regret
    return 1.0 * steps                    # too much machine: wasted spend


def risk_cost(true: Risk, predicted: Risk) -> float:
    """Cost of a risk misjudgement, given the escalate-only invariant.

    Under-calling risk is what matters, and it is priced steeply. Over-calling
    it costs a confirmation prompt somebody has to dismiss, which is a real
    cost — an assistant that asks permission for everything gets its
    permissions turned off — but not a comparable one.

    These numbers describe the model's *advice*. In a host that follows the
    invariant, a missed P0 is still caught by that host's own rules; the price
    here is of the advice being wrong, not of the damage being done.
    """
    if predicted == true:
        return 0.0
    if predicted < true:
        return 50.0 * (int(true) - int(predicted))   # under-called danger
    return 1.0 * (int(predicted) - int(true))        # nagging


# ---------------------------------------------------------------------------
# one labelled example
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Example:
    """One labelled turn. ``risk`` is only meaningful at the ACT site."""

    site: str
    text: str                 # the user's turn
    action: str = ""          # the proposed tool call, ACT only
    route: Route = Route.LARGE
    risk: Risk = Risk.P2_READONLY
    gate: int = 0             # 1 if a human should confirm
    source: str = ""          # which generator template produced it
    split: str = "train"

    def encoded(self) -> str:
        """Exactly what the encoder sees. One place, so nothing drifts."""
        marker = SITE_MARKERS[self.site]
        if self.site == SITE_ACT and self.action:
            return f"{marker} {self.text} [SEP] {self.action}"
        return f"{marker} {self.text}"

    def as_dict(self) -> dict:
        return {
            "site": self.site, "text": self.text, "action": self.action,
            "route": int(self.route), "risk": int(self.risk),
            "gate": int(self.gate), "source": self.source, "split": self.split,
        }

    @staticmethod
    def from_dict(row: dict) -> "Example":
        return Example(
            site=str(row["site"]), text=str(row["text"]),
            action=str(row.get("action") or ""),
            route=Route(int(row.get("route", Route.LARGE))),
            risk=Risk(int(row.get("risk", Risk.P2_READONLY))),
            gate=int(row.get("gate", 0)), source=str(row.get("source") or ""),
            split=str(row.get("split") or "train"),
        )


#: Splits. ``audit`` is carved out on day one and never trained on, so there is
#: always one set of numbers nobody has tuned against.
SPLITS = ("train", "dev", "test", "audit")


__all__ = [
    "ABSTAIN", "DEFAULT_REGRET_RATIO", "Example", "GATE", "MAX_LENGTH",
    "RISK_NAMES",
    "ROUTE_NAMES", "Risk", "Route", "SITES", "SITE_ACT", "SITE_MARKERS",
    "SITE_PRE", "SPLITS", "risk_cost", "route_cost",
]
