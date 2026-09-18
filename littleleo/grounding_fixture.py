"""Turns with real content attached, for grounding the labels against real models.

The routing corpus is built from abstract shapes — "this reconciliation", "this
proof" — with no payload, because that is what forces a model to read structure
rather than vocabulary. It is the right design for training a router and the
wrong one for asking whether a cheap model can actually do the work: a model
sent "give me only the total" with no table attached can only invent a number or
ask for one, and a judge comparing two such answers is grading hallucination
style.

So this fixture pairs each request with content that makes it answerable. Same
three tiers, same kinds of request, but now there is something to reason about.

Three tiers, one payload
------------------------

Each payload carries three requests, so the *only* thing that varies between
tiers is what is being asked of the same material:

``NO_MODEL``
    A lookup or an arithmetic step with a bounded answer. A script could do it,
    and crucially **the correct answer is known**, so these are checked exactly
    rather than by judgement.

``SMALL``
    A bounded transformation of the text. No reasoning chain.

``LARGE``
    A question whose answer requires holding a constraint against the content —
    finding which assumption fails, which figure is untrustworthy, what breaks
    under a condition the payload does not state.

The tier assignments remain *assertions*. What changes is that they are now
falsifiable: if a cheap model answers every LARGE request as well as a frontier
model does, the assertion was wrong and the experiment will say so.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingItem:
    payload: str
    request: str
    tier: int            # 0 NO_MODEL, 1 SMALL, 2 LARGE
    kind: str
    expected: str = ""   # exact answer, NO_MODEL items only

    def prompt(self) -> str:
        return f"{self.request}\n\n---\n{self.payload}\n---"


_LEDGER = """Invoice  Net      VAT     Gross
INV-1041  1,200.00   240.00   1,440.00
INV-1042    850.00   170.00   1,020.00
INV-1043    430.00    86.00     516.00
INV-1044  2,100.00   420.00   2,520.00"""

_LEDGER_MISMATCH = """Bank statement (received)      Ledger (recorded)
2026-08-03  INV-1041  1,440.00   2026-08-03  INV-1041  1,440.00
2026-08-07  INV-1042  1,020.00   2026-08-07  INV-1042  1,020.00
2026-08-19  INV-1043    516.00   2026-08-18  INV-1043    516.00
2026-08-28  REFUND     -204.00   (not recorded)
Statement closing balance: 2,772.00
Ledger closing balance:    2,976.00"""

_SNIPPET = """def apply_discount(total, rate):
    if rate > 1:
        rate = rate / 100
    return round(total - (total * rate), 2)

def checkout(items, rate):
    total = sum(i["price"] * i["qty"] for i in items)
    return apply_discount(total, rate)"""

_RETRY = """def send(payload, attempts=0):
    try:
        return transport.post(payload)
    except Timeout:
        if attempts < 5:
            return send(payload, attempts + 1)
        raise
    except Conflict:
        return send(payload, attempts)"""

_MIGRATION = """Plan: move `orders.status` from VARCHAR to an enum type.
1. Add column `status_new` (enum), nullable.
2. Backfill `status_new` from `status` in batches of 10,000.
3. Deploy application writing to both columns.
4. Verify counts match, then drop `status`.
5. Rename `status_new` to `status`.
The service runs continuously; there is no maintenance window."""

_PROOF = """Claim: the retry loop always terminates.
Lemma 1: each retry increments `attempts`.
Lemma 2: the loop exits when `attempts` reaches 5.
Therefore after at most 5 retries the loop exits. QED."""

_PARAGRAPH = """the quarterly numbers came in a bit under what we planned for, mostly
because two enterprise deals slipped into next quarter rather than anything going
wrong with demand. pipeline coverage is actually better than it was this time last
year and churn held flat at 3 percent. we spent less on infrastructure than
budgeted after the storage migration finished early."""

_EMAIL = """Dear Sir/Madam, Further to our telephone conversation of the 14th inst.,
I am writing to confirm that the consignment referenced above was despatched on
the 16th and should reach your premises no later than the 22nd. Kindly advise
should any discrepancy arise. Yours faithfully, A. Pereira"""

_NOTES = """standup 12 sep - raj: auth migration done, rollback tested
- priya: dashboard slow on large accounts, suspects the n+1 in billing
- sam: out thu/fri
- decision: hold the release until priya's fix lands
- next: raj to review the fix, sam to update the runbook when back"""

_LATENCY = """Endpoint          p50    p95    p99   req/s
/api/orders       12ms   40ms  310ms   1200
/api/orders/:id    8ms   19ms   22ms   3400
/api/search      140ms  820ms 2400ms     95
/api/health        1ms    2ms    2ms   5000"""

_CACHE = """The read path checks Redis first; on a miss it reads Postgres and writes
the result back to Redis with a 300-second TTL. Writes update Postgres and delete
the Redis key. Two application instances serve traffic behind a round-robin load
balancer."""

_SCHEDULE = """Job            Runs at   Duration  Depends on
extract        01:00     40 min    -
transform      01:30     25 min    extract
load           02:00     35 min    transform
reconcile      02:30     10 min    load
report         03:00      5 min    reconcile"""

_CONTRACT = """7.2 The Supplier shall deliver within thirty (30) days of the Order Date.
7.3 Where delivery is delayed beyond the period in 7.2, the Customer may terminate
    on written notice.
7.4 Clause 7.3 shall not apply where the delay arises from causes beyond the
    Supplier's reasonable control, provided the Supplier notifies the Customer
    within five (5) Business Days of becoming aware of such cause."""


#: Each payload with one request per tier. NO_MODEL items carry the exact answer.
ITEMS: list[GroundingItem] = [
    GroundingItem(_LEDGER, "Give me only the total of the Gross column, as a number and nothing else.", 0, "ledger", "5496.00"),
    GroundingItem(_LEDGER, "Rewrite this as a plain-English sentence for a non-finance reader.", 1, "ledger"),
    GroundingItem(_LEDGER, "If VAT should be 20% of Net throughout, identify any row where the recorded VAT is inconsistent and say what the Gross should have been.", 2, "ledger"),

    GroundingItem(_LEDGER_MISMATCH, "State only the numeric difference between the two closing balances.", 0, "reconciliation", "204.00"),
    GroundingItem(_LEDGER_MISMATCH, "Turn this into a short bulleted list of the entries on each side.", 1, "reconciliation"),
    GroundingItem(_LEDGER_MISMATCH, "Explain exactly where the closing balances diverge and which of the two you should trust, given the ledger is the system of record.", 2, "reconciliation"),

    GroundingItem(_SNIPPET, "How many functions are defined? Reply with the digit only.", 0, "code", "2"),
    GroundingItem(_SNIPPET, "Reformat this with consistent four-space indentation and add type hints.", 1, "code"),
    GroundingItem(_SNIPPET, "A caller passes rate=1 meaning 1%. What does checkout return for a 100.00 total, and why is that wrong?", 2, "code"),

    GroundingItem(_RETRY, "Name only the two exception types handled.", 0, "retry", "Timeout, Conflict"),
    GroundingItem(_RETRY, "Rewrite this using a loop instead of recursion, keeping the behaviour identical.", 1, "retry"),
    GroundingItem(_RETRY, "Does this always terminate? Identify precisely which path does not, and under what condition.", 2, "retry"),

    GroundingItem(_MIGRATION, "How many numbered steps are there? Reply with the digit only.", 0, "migration", "5"),
    GroundingItem(_MIGRATION, "Condense this plan to under forty words.", 1, "migration"),
    GroundingItem(_MIGRATION, "The service runs under continuous write load. Identify the first step at which this plan loses data or breaks writes, and why.", 2, "migration"),

    GroundingItem(_PROOF, "How many lemmas are stated? Reply with the digit only.", 0, "proof", "2"),
    GroundingItem(_PROOF, "Restate this claim and its argument in plainer language.", 1, "proof"),
    GroundingItem(_PROOF, "Is the claim proved? If not, name the specific lemma that fails and give a counterexample.", 2, "proof"),

    GroundingItem(_PARAGRAPH, "State only the churn percentage.", 0, "prose", "3"),
    GroundingItem(_PARAGRAPH, "Rewrite this in formal business English, preserving every fact.", 1, "prose"),
    GroundingItem(_PARAGRAPH, "Management wants to know whether this quarter's shortfall signals a demand problem. Answer using only what is stated here, and say what you cannot conclude.", 2, "prose"),

    GroundingItem(_EMAIL, "Give only the despatch date as it appears.", 0, "email", "16th"),
    GroundingItem(_EMAIL, "Rewrite this as a two-sentence modern email.", 1, "email"),
    GroundingItem(_EMAIL, "The consignment arrived on the 25th. Using only this letter, say whether the supplier breached what it promised and what remains ambiguous.", 2, "email"),

    GroundingItem(_NOTES, "Name only the person who is out on Thursday and Friday.", 0, "notes", "sam"),
    GroundingItem(_NOTES, "Turn these notes into a tidy bulleted summary.", 1, "notes"),
    GroundingItem(_NOTES, "Sam returns Monday and the release is still held. Work out what is now blocking the release and who must act first.", 2, "notes"),

    GroundingItem(_LATENCY, "State only the p99 for /api/search.", 0, "latency", "2400ms"),
    GroundingItem(_LATENCY, "Present this as a short prose summary.", 1, "latency"),
    GroundingItem(_LATENCY, "Which endpoint should be optimised first to reduce total user-visible waiting, and why is the obvious answer not the right one?", 2, "latency"),

    GroundingItem(_CACHE, "How many application instances are described? Reply with the digit only.", 0, "cache", "2"),
    GroundingItem(_CACHE, "Summarise this caching scheme in two sentences.", 1, "cache"),
    GroundingItem(_CACHE, "Describe a concrete interleaving of two requests that leaves Redis holding a stale value indefinitely.", 2, "cache"),

    GroundingItem(_SCHEDULE, "State only the start time of the reconcile job.", 0, "schedule", "02:30"),
    GroundingItem(_SCHEDULE, "Rewrite this schedule as a sentence per job.", 1, "schedule"),
    GroundingItem(_SCHEDULE, "If extract overruns by 25 minutes, which jobs are affected and does the report still start on time? Show the reasoning.", 2, "schedule"),

    GroundingItem(_CONTRACT, "How many clauses are quoted? Reply with the digit only.", 0, "contract", "3"),
    GroundingItem(_CONTRACT, "Rewrite these clauses in plain English.", 1, "contract"),
    GroundingItem(_CONTRACT, "The supplier was late due to a port strike and notified on the eighth business day. Can the customer terminate? Give the reasoning from the clauses.", 2, "contract"),
]

TIER_NAMES = {0: "NO_MODEL", 1: "SMALL", 2: "LARGE"}


def by_tier() -> dict[int, list[GroundingItem]]:
    out: dict[int, list[GroundingItem]] = {0: [], 1: [], 2: []}
    for item in ITEMS:
        out[item.tier].append(item)
    return out


__all__ = ["GroundingItem", "ITEMS", "TIER_NAMES", "by_tier"]
