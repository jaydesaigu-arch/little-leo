"""The bar the model has to clear, written before the model exists.

Every router project should have to answer one question in public: *is this
better than twenty lines of string matching?* Surprisingly many never ask it,
and the ones that do sometimes discover the answer is no.

So the baselines are built first and reported beside the model for the life of
the project. Two of them are deliberately stupid, to fix the ends of the scale:

``always_large``
    What an inefficient agent already does. Zero savings, zero regret. Every
    claim this project makes is a claim about beating this.

``always_no_model``
    The opposite floor. Maximum savings, catastrophic regret. It exists so that
    "we saved 90% of calls" can never be printed without something to compare
    the quality cost against.

``keyword``
    The real bar. A short, readable rule set of the kind a competent engineer
    writes in an afternoon. If the trained model cannot beat this on mean cost,
    the honest conclusion is to ship the rules and delete the model.

The risk rules are a separate matter and are expected to do well — that is the
design working, not a problem. Destructive shell and SQL operations really are
mostly recognisable by pattern, which is exactly why the architecture makes
deterministic rules the gate and the model only an escalation signal. A model
that merely matches these rules on risk has still earned its place if it catches
the phrasings they miss.
"""

from __future__ import annotations

import re
from typing import Callable

from .schema import Example, Risk, Route

Router = Callable[[Example], tuple[Route, Risk]]


# ---------------------------------------------------------------------------
# the two ends of the scale
# ---------------------------------------------------------------------------

def always_large(example: Example) -> tuple[Route, Risk]:
    """The status quo for an agent with no routing at all."""
    return Route.LARGE, Risk.P0_DESTRUCTIVE if _looks_destructive(example) else Risk.P2_READONLY


def always_small(example: Example) -> tuple[Route, Risk]:
    return Route.SMALL, Risk.P2_READONLY


def always_no_model(example: Example) -> tuple[Route, Risk]:
    return Route.NO_MODEL, Risk.P2_READONLY


# ---------------------------------------------------------------------------
# the bar worth beating
# ---------------------------------------------------------------------------

#: An output bound: the answer is a word, a digit, a letter. Whatever the turn
#: sounded like, a script could produce this.
_COLLAPSER = re.compile(
    r"(only|just|nothing but|nothing beyond|bare|solitary|single (word|letter|"
    r"character|syllable|digit)|one (word|token|letter|syllable)|yes or no|"
    r"true or false|one-word|a one-word)", re.I)

#: A condition or dependency that has to be reasoned about.
_QUALIFIER = re.compile(
    r"(given|taking into account|bearing in mind|considering|accounting for|"
    r"allowing for|supposing|on the assumption|while (working|reasoning|"
    r"determining)|what breaks|which (assumption|invariant|of the two)|"
    r"under (active )?load|without downtime|cannot take downtime|at ten times|"
    r"mid-transaction|out of order|race for)", re.I)

#: Bounded, mechanical text work.
_TRANSFORM = re.compile(
    r"(rewrite|rewrit|bullet points|shorten|trim it|fix the typos|correct the "
    r"spelling|translate|reformat|convert it|render it|recast|tidy up|lay it out|"
    r"compress it|normalise|present tense|extract the|pull out the)", re.I)


def keyword(example: Example) -> tuple[Route, Risk]:
    """A rules-only router, rewritten for the compositional corpus.

    The rules look for the clause that decides the tier rather than for topic
    words, which is the same thing a competent engineer would do after reading
    twenty examples. It is a fair opponent: an unfair one makes the model look
    good and teaches nobody anything.

    Its ceiling is that it can only recognise clause wordings somebody thought
    to enumerate. That is exactly the limitation the audit split is built to
    expose.
    """
    return _keyword_route(example), _keyword_risk(example)


def _keyword_route(example: Example) -> Route:
    text = (example.text or "").strip()
    if not text:
        return Route.LARGE
    # Order matters: a collapser bounds the work however heavy the rest reads,
    # so it is tested before the qualifier.
    if _COLLAPSER.search(text):
        return Route.NO_MODEL
    if _QUALIFIER.search(text):
        return Route.LARGE
    if _TRANSFORM.search(text):
        return Route.SMALL
    return Route.LARGE


#: Irreversible, or reaching outside the file that was opened.
_DESTRUCTIVE = re.compile(
    r"(run it in a subshell|execute what it prints|pipe what it emits|"
    r"into the shell|evaluate its contents|write the output back over|"
    r"overwrite /|replace /|replace ~|remove the originals|drop the table|"
    r"purge every row|irreversibly clear|clear the store|/etc/|/var/log/|"
    r"authorized_keys|\.git/config)", re.I)

#: Changes something recoverable.
_MUTABLE = re.compile(
    r"(write a tidied copy|append a note|commit the result|save a corrected|"
    r"stage the change|store an amended|record the outcome|a new file)", re.I)


def _looks_destructive(example: Example) -> bool:
    return bool(_DESTRUCTIVE.search(example.action or ""))


def _keyword_risk(example: Example) -> Risk:
    action = example.action or ""
    if not action:
        return Risk.P2_READONLY
    if _DESTRUCTIVE.search(action):
        return Risk.P0_DESTRUCTIVE
    if _MUTABLE.search(action):
        return Risk.P1_MUTABLE
    return Risk.P2_READONLY


BASELINES: dict[str, Router] = {
    "always_large": always_large,
    "always_small": always_small,
    "always_no_model": always_no_model,
    "keyword": keyword,
}


__all__ = ["BASELINES", "Router", "always_large", "always_no_model",
           "always_small", "keyword"]
