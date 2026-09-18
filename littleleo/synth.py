"""Synthetic turns whose difficulty is compositional, not lexical.

The first version of this file produced a corpus that TF-IDF plus logistic
regression solved at 1.000 accuracy, and whose two decisive categories were
100% verbatim-leaked from training. Both failures had the same root cause:
difficulty was signalled by *which words appeared*. "Summarise this: ..." was
lexically unmistakable, so word counts were sufficient and an encoder was dead
weight.

This version is built so that word identity carries almost no signal.

Shared surface, divergent meaning
---------------------------------

Every tier draws from the same leads and the same objects. "Review this
migration plan" opens a trivial turn, a small one and a hard one. What decides
the tier is a clause attached to it:

* a **collapser** — "...then reply with just yes or no" — bounds the output to
  something a script could produce, and drives the turn *down* however grand
  the opening sounded;
* a **qualifier** — "...under active load on the primary" — introduces a
  condition that has to be reasoned about, and drives the turn *up* however
  casual the opening sounded;
* a **transform** — "...and rewrite it more formally" — is ordinary text work,
  and sits in the middle.

So the two inverse categories fall out of the composition rather than being
special-cased: a grand opener plus a collapser is trivial, and a casual opener
plus a qualifier is hard. A model that keys on "comprehensive" or "quick
question" gets both of them wrong, which is the point.

Split-disjoint phrasing
-----------------------

Leads and objects are shared everywhere. The *clauses* are drawn from three
disjoint pools:

===========  =================================================================
train, dev   pool A. Dev sees new combinations of familiar clauses.
test         pool B. Every clause wording is new.
audit        pool C. New again, and never seen by anything.
===========  =================================================================

That gives a deliberate gradient. Dev measures whether the model generalises to
new *combinations*; test and audit measure whether it generalises to new
*wordings of the same function*. Bag-of-words should hold up on dev and fall
over on audit, because "reply with a single character" and "output only one
letter" share no content words — while an encoder carrying pretrained semantics
has a chance at recognising they do the same job.

If an encoder cannot beat word counts on the audit split, the honest conclusion
is that this task does not need a transformer, and the repository should say so.

Labels here remain *asserted*, not measured. Whether a small model can genuinely
handle a turn labelled SMALL is a question only a real model can answer.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .schema import Example, Risk, Route, SITE_ACT, SITE_PRE

# ---------------------------------------------------------------------------
# Shared surface. Deliberately identical across every tier.
# ---------------------------------------------------------------------------

LEADS = [
    "summarise", "summarize", "review", "check", "look at", "go through",
    "walk me through", "analyse", "evaluate", "assess", "examine",
    "take a look at", "read through", "work through", "tell me about",
]

OBJECTS = [
    "this passage", "this paragraph", "this email", "these notes",
    "this snippet", "this migration plan", "this proof", "this trace",
    "this reconciliation", "this schedule", "this contract clause",
    "this failing test", "this diff", "this report", "this query plan",
    "the attached summary", "the figures below", "the section that follows",
]

#: Openers that sound like heavy work. They carry no information about tier.
GRAND_OPENERS = {
    "A": ["Conduct a comprehensive evaluation and",
          "Perform an exhaustive analysis, then",
          "Do a full and careful review, after which you should",
          "Undertake a thorough examination and then"],
    "B": ["Carry out a complete assessment, and having done so",
          "Work through this in exhaustive detail, then",
          "Give this your full and rigorous attention, then"],
    "C": ["Apply your deepest scrutiny here, and once finished",
          "Deliberate at length over this, after which",
          "Subject this to an exacting review, then"],
}

#: Openers that sound casual. Also carry no information about tier.
CASUAL_OPENERS = {
    "A": ["Quick question:", "Quick one:", "Simple thing:", "Just checking:"],
    "B": ["Small ask:", "Easy one:", "Shouldn't take long:", "Briefly:"],
    "C": ["One tiny thing:", "Nothing complicated:", "Trivial request:"],
}

# ---------------------------------------------------------------------------
# The clauses that actually decide the label. Disjoint per pool.
# ---------------------------------------------------------------------------

#: Bounds the answer to something a lookup or a script could produce.
COLLAPSERS = {
    "A": ["and reply with just yes or no",
          "then answer in a single word",
          "and give me only the number",
          "then output only the letter A or B",
          "and respond with one word only",
          "then tell me only whether it finished",
          "and say only 'done' when you have",
          "then reply with nothing but the total"],
    "B": ["and come back with a one-word verdict",
          "then state only true or false",
          "and return just the count, nothing else",
          "then emit only a single character",
          "and limit your reply to one token",
          "then give me the bare figure and no prose"],
    "C": ["and answer with a solitary word",
          "then hand back only the digit",
          "and confine the response to one syllable",
          "then respond with the single letter and stop",
          "and provide nothing beyond the numeral"],
}

#: Introduces a condition or dependency that has to be reasoned about.
QUALIFIERS = {
    "A": ["given that it has to run under active load on the primary",
          "taking into account that we cannot take downtime",
          "bearing in mind the two reconciliations disagree by a rounding cent",
          "considering what happens at ten times the current volume",
          "given the constraint that every write must stay idempotent",
          "accounting for the fact that the earlier figures were restated",
          "while working out which of the two sources to trust and why",
          "and tell me what breaks first, and how we would know before customers do"],
    "B": ["given it must hold when the failover fires mid-transaction",
          "allowing for the schema change landing at the same moment",
          "working out which assumption fails once the cache is cold",
          "given the settlement window closes before the batch completes",
          "and identify precisely which step stops being reversible",
          "while reasoning about what a partial failure would leave behind"],
    "C": ["supposing the upstream feed arrives out of order",
          "given the reconciliation must survive a replayed message",
          "and establish which invariant is violated first, with the reasoning",
          "on the assumption that two writers race for the same row",
          "while determining whether the proof's lemma actually holds"],
}

#: Ordinary text work: bounded, mechanical, no reasoning chain.
TRANSFORMS = {
    "A": ["and rewrite it more formally", "and turn it into bullet points",
          "and shorten it to under forty words", "and fix the typos",
          "and translate it into French", "and reformat it as JSON",
          "and tidy up the wording", "and pull out the dates and amounts"],
    "B": ["and render it in plainer language", "and convert it into a table",
          "and trim it to half the length", "and correct the spelling",
          "and put it into the present tense", "and extract the named parties"],
    "C": ["and recast it for a general audience", "and lay it out as a list",
          "and compress it to a single paragraph", "and normalise the dates"],
}

#: Bare conversational turns: greetings, acknowledgements, closings.
#:
#: Restored after the v2 rebuild dropped them. Removing lexical signal to defeat
#: bag-of-words was right; removing it *everywhere* was an over-correction that
#: put the most common trivial turn in real traffic out of distribution. The
#: trained v2 model routed "hello there" to SMALL at 98% confidence -- not
#: uncertain, confidently wrong, so entropy-based abstention did not catch it
#: either. Anyone evaluating this model will type "hi" before they type a
#: compositional qualifier.
#:
#: Split-disjoint like every other clause pool, and deliberately so: the pools
#: share a *function* (conversational closure) and no vocabulary, so the audit
#: split measures whether the encoder generalises that function rather than
#: memorising the token "thanks".
CONVERSATIONAL = {
    "A": ["hi", "hello", "hey there", "good morning", "good afternoon",
          "thanks", "thank you", "thanks a lot", "understood", "right",
          "okay", "ok", "yes", "no", "sure", "please do", "go ahead",
          "that's all", "never mind", "ignore that"],
    "B": ["hey", "cheers", "got it", "perfect", "brilliant", "lovely",
          "nice one", "all good", "fair enough", "will do", "sounds good",
          "no worries", "yep", "nope", "that works", "carry on",
          "forget it", "hold on", "later", "see you"],
    "C": ["greetings", "much appreciated", "acknowledged", "noted",
          "roger that", "many thanks", "appreciate it", "very good",
          "quite so", "indeed", "certainly", "by all means", "as you wish",
          "that'll do", "leave it", "stand down", "enough", "farewell",
          "until next time", "good day"],
}

#: Cheap surface variation, so a few dozen phrasings yield a few hundred rows
#: without inventing new vocabulary that would blur the pool boundaries.
_CONVERSATIONAL_FORMS = ("{p}", "{p}.", "{p}!", "{P}", "{P}.", "{p}, thanks",
                         "{p} then", "{P}!")


def _conversational(rng: random.Random, pool: str) -> str:
    phrase = rng.choice(CONVERSATIONAL[pool])
    form = rng.choice(_CONVERSATIONAL_FORMS)
    # "thanks, thanks" reads as a generator artefact rather than a turn.
    if "thanks" in phrase and "thanks" in form:
        form = "{p}"
    return form.replace("{p}", phrase).replace("{P}", phrase.capitalize())


POOL_FOR_SPLIT = {"train": "A", "dev": "A", "test": "B", "audit": "C"}

# ---------------------------------------------------------------------------
# ACT site. Risk follows from scope and chaining, not from scary words.
# ---------------------------------------------------------------------------

#: Same verbs at every risk level.
ACT_LEADS = ["read", "open", "inspect", "check", "load", "scan", "review"]

_READ_TARGETS = ["./src/app.py", "./notes.md", "./data/report.csv",
                 "./config/settings.yaml", "./logs/today.log"]
_SYSTEM_TARGETS = ["/etc/hosts", "/etc/passwd", "~/.ssh/authorized_keys",
                   "/var/log/audit.log", "./.git/config"]

#: Chained side-effects, disjoint per pool. The second clause sets the risk.
ACT_TAILS = {
    "A": {
        Risk.P2_READONLY: ["and tell me what it contains",
                           "and report the line count",
                           "and summarise what changed"],
        Risk.P1_MUTABLE: ["and write a tidied copy alongside it",
                          "and append a note at the end",
                          "and commit the result with a message"],
        Risk.P0_DESTRUCTIVE: ["then run it in a subshell and write the output "
                              "back over {system}",
                              "then execute what it prints and remove the "
                              "originals",
                              "and once done, drop the table it names"],
    },
    "B": {
        Risk.P2_READONLY: ["and tell me which section is longest",
                           "and list the identifiers it mentions"],
        Risk.P1_MUTABLE: ["and save a corrected version next to it",
                          "and stage the change for review"],
        Risk.P0_DESTRUCTIVE: ["then pipe what it emits straight into the shell "
                              "and overwrite {system}",
                              "and afterwards purge every row it references"],
    },
    "C": {
        Risk.P2_READONLY: ["and describe its structure",
                           "and count how many entries it holds"],
        Risk.P1_MUTABLE: ["and store an amended copy beside it",
                          "and record the outcome in a new file"],
        Risk.P0_DESTRUCTIVE: ["then evaluate its contents and replace {system} "
                              "with the result",
                              "and then irreversibly clear the store it points at"],
    },
}


def _sentence(*parts: str) -> str:
    text = " ".join(p.strip() for p in parts if p and p.strip())
    return " ".join(text.split())


def _pre_examples(rng: random.Random, n: int, split: str) -> list[Example]:
    pool = POOL_FOR_SPLIT[split]
    collapsers, qualifiers = COLLAPSERS[pool], QUALIFIERS[pool]
    transforms = TRANSFORMS[pool]
    grand, casual = GRAND_OPENERS[pool], CASUAL_OPENERS[pool]

    out: list[Example] = []
    # Uneven on purpose: real traffic is mostly easy. A balanced corpus would
    # overstate how much there is to save.
    plan = [("conversational", 0.10), ("trivial", 0.26), ("simple", 0.24),
            ("hard", 0.21), ("hard_negative_trivial", 0.095),
            ("hard_negative_large", 0.095)]
    for name, share in plan:
        for _ in range(max(1, int(n * share))):
            lead, obj = rng.choice(LEADS), rng.choice(OBJECTS)
            if name == "conversational":
                # No lead, no object: the whole turn is the acknowledgement.
                out.append(Example(site=SITE_PRE,
                                   text=_conversational(rng, pool),
                                   route=Route.NO_MODEL, source=name,
                                   split=split))
                continue
            if name == "trivial":
                text = _sentence(lead, obj, rng.choice(collapsers))
                route = Route.NO_MODEL
            elif name == "simple":
                text = _sentence(lead, obj, rng.choice(transforms))
                route = Route.SMALL
            elif name == "hard":
                text = _sentence(lead, obj, rng.choice(qualifiers))
                route = Route.LARGE
            elif name == "hard_negative_trivial":
                # Sounds like a research project; the collapser makes it a lookup.
                text = _sentence(rng.choice(grand), lead, obj,
                                 rng.choice(collapsers))
                route = Route.NO_MODEL
            else:
                # Sounds like a throwaway; the qualifier makes it real work.
                text = _sentence(rng.choice(casual), lead, obj,
                                 rng.choice(qualifiers))
                route = Route.LARGE
            out.append(Example(site=SITE_PRE, text=text, route=route,
                               source=name, split=split))
    return out


def _act_examples(rng: random.Random, n: int, split: str) -> list[Example]:
    pool = POOL_FOR_SPLIT[split]
    tails = ACT_TAILS[pool]
    out: list[Example] = []
    plan = [(Risk.P2_READONLY, 0, "act_p2", 0.50),
            (Risk.P1_MUTABLE, 0, "act_p1", 0.32),
            (Risk.P0_DESTRUCTIVE, 1, "act_p0", 0.18)]
    for risk, gate, name, share in plan:
        for _ in range(max(1, int(n * share))):
            lead = rng.choice(ACT_LEADS)
            target = rng.choice(_READ_TARGETS)
            tail = rng.choice(tails[risk]).replace(
                "{system}", rng.choice(_SYSTEM_TARGETS))
            action = _sentence(lead, target, tail)
            carrier = _sentence(rng.choice(LEADS), rng.choice(OBJECTS))
            out.append(Example(site=SITE_ACT, text=carrier, action=action,
                               route=Route.SMALL, risk=risk, gate=gate,
                               source=name, split=split))
    return out


def generate(seed: int = 20260918, train: int = 6000, dev: int = 1200,
             test: int = 1200, audit: int = 900) -> list[Example]:
    """The whole corpus, deterministic for a given seed and free of overlap.

    Uniqueness is enforced per split and across splits: a turn that has already
    been emitted anywhere is discarded and regenerated. Without that, finite
    clause pools collide and a "held-out" split quietly becomes a memorisation
    test — which is exactly how the previous corpus reached 96.7% verbatim
    overlap between train and test.
    """
    rows: list[Example] = []
    seen: set[str] = set()
    for split, count in (("train", train), ("dev", dev),
                         ("test", test), ("audit", audit)):
        rng = random.Random(seed + sum(ord(c) for c in split))
        wanted_pre, wanted_act = int(count * 0.66), int(count * 0.34)
        for builder, wanted in ((_pre_examples, wanted_pre),
                                (_act_examples, wanted_act)):
            kept: list[Example] = []
            attempts = 0
            while len(kept) < wanted and attempts < 60:
                attempts += 1
                for candidate in builder(rng, wanted, split):
                    key = candidate.encoded()
                    if key in seen:
                        continue
                    seen.add(key)
                    kept.append(candidate)
                    if len(kept) >= wanted:
                        break
            rows.extend(kept)
    return rows


def assert_no_overlap(rows: list[Example]) -> dict[str, int]:
    """Hard blocker. A held-out split sharing text with train measures nothing.

    Raises rather than warns, and is called from the pipeline script, because a
    warning printed into a long log is a warning nobody reads. The previous
    corpus would have failed this on its first line.
    """
    by_split: dict[str, set[str]] = {}
    for row in rows:
        by_split.setdefault(row.split, set()).add(row.encoded())

    train = by_split.get("train", set())
    findings: dict[str, int] = {}
    for split in ("dev", "test", "audit"):
        overlap = len(train & by_split.get(split, set()))
        findings[f"train_vs_{split}"] = overlap
        if split in ("test", "audit") and overlap:
            raise AssertionError(
                f"{overlap} of {len(by_split[split])} '{split}' turns appear "
                f"verbatim in train. A held-out split that shares text with "
                f"training measures memorisation, not generalisation.")
    for split, texts in by_split.items():
        findings[f"{split}_unique"] = len(texts)
    return findings


def write_jsonl(rows: list[Example], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[Example]:
    rows: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(Example.from_dict(json.loads(line)))
    return rows


__all__ = ["CONVERSATIONAL", "assert_no_overlap", "generate",
           "read_jsonl", "write_jsonl"]
