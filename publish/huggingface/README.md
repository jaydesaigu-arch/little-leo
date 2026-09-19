---
license: apache-2.0
language: en
library_name: onnxruntime
tags:
  - text-classification
  - router
  - llm-routing
  - onnx
  - multi-task
pipeline_tag: text-classification
---

# Little Leo — LL-22M

A 22.7M-parameter bidirectional encoder that decides, in about seven
milliseconds on one CPU core, how much machine a turn actually needs — and flags
proposed actions that could do damage.

It exists for agents that call a frontier model for everything, including
"thanks, that worked".

**Measured on live provider traffic: 22.5% cost saving and 13.7% token saving
against sending everything to GPT-5.4, with zero prompts left unanswered.**
Full numbers, method and caveats below.

It is not a safety boundary, it is not a guardrail, and it must never be the
only thing standing between a user and an irreversible operation.

---

## Live measurement

117 prompts across 13 payload types, each sent to **both** models so the
counterfactual is observed rather than assumed, then routed by LL-22M's own
decisions — not by the labels its training corpus asserts.

| | |
| --- | ---: |
| Baseline — every prompt to `openai/gpt-5.4` | $0.29826 |
| Routed by LL-22M | $0.23114 |
| **Cost saving** | **22.5%** |
| **Token saving** | **13.7%** (34,904 → 30,136) |
| **Prompts left unanswered** | **0 / 117** |
| Downgrades judged materially worse | 3 / 30 — see below |
| Routing split | 33 `NO_MODEL` · 30 `SMALL` · 54 `LARGE` |

**Cheap tier** `google/gemini-2.5-flash` ($0.30 / $2.50 per Mtok) ·
**frontier tier** `openai/gpt-5.4` ($2.50 / $15.00) ·
**judge** `anthropic/claude-opus-4.5`, a third vendor neutral to both
contestants, shown both answers blind with the order randomised.

### Cost saving and token saving differ, and both are reported

22.5% and 13.7% measure different things. Gemini is cheaper per token *and*
produces different-length answers, so the dollar saving exceeds the token
saving. Quoting only the larger number would be selective.

### On the quality figure

Across five runs of the *identical* comparison — same model, same fixture,
temperature 0 — the materially-worse rate came back **17%, 11.1%, 3.3%, 15%
and 10.0%**. The only moving part is the judge's verdict on borderline pairs.

The honest statement is **"roughly one in ten, with wide uncertainty"**. The
final run's 3/30 carries a 95% confidence interval of **[3.5%, 25.6%]**, and a
decimal point would imply precision the data does not support.

The three failures in the final run were **scattered** — one each on `ledger`,
`email` and `contract` — rather than concentrated, and two are arguably not
quality differences at all:

> *email* — "Answer A has three sentences; Answer B has two as requested."
> *contract* — "A says 'late' for 7.3; source says 'delayed beyond' 30 days."

Instruction-following on sentence count, and a wording nicety. Neither would
change what the user does next, which was the bar the judge was given.

---

## The safety architecture — read before deploying

Two outputs, with completely different standing.

**Routing is advisory.** If it is wrong you spend too much, or get a weaker
answer than you wanted.

**Risk is advisory and bounded.** The model may not make your system less
careful:

```python
final_risk = max(deterministic_rules(action), encoder(action))
```

Deterministic rules run first and set the floor. The encoder may **raise** a
risk tier; it may never **lower** one.

This is load-bearing, not decoration:

| pool | encoder alone | rules alone | **rules → encoder** |
| --- | --- | --- | --- |
| test | 0.726 — **20 P0 actions missed** | 1.000 | **1.000, precision 1.000** |
| audit | 1.000 | 1.000 | **1.000, precision 1.000** |

On the test pool the encoder misses one destructive action in four. Every miss
is the same phrasing — `"and afterwards purge every row it references"` — data
deletion with no filesystem path and no shell invocation, which the model reads
as recoverable mutation. The rules catch all twenty.

**A deployment using the encoder alone for risk will miss destructive actions.**
That is measured, above, on this model.

---

## The confidence floor

Routing decisions below **0.90 confidence are discarded** and the host's default
is used instead.

This is not a tuning knob, it is a measured fix. Before it, 5.1% of prompts were
routed to `NO_MODEL` — no call made, no answer delivered — for tasks that needed
an answer. Zero dollars, and a user staring at silence.

On the fixture, every correct `NO_MODEL` decision but one sat at ≥0.99
(median 0.998), while both wrong ones sat at 0.481 and 0.735.

**0.90 rather than 0.75:** a 0.75 floor also worked, by a margin of 0.005 over
the worst wrong decision. That is a coincidence on fourteen observations, not a
separation. 0.90 clears it by 0.165.

It costs about six percentage points of saving, because one genuinely trivial
turn sits near the boundary and is now served by a real model. The floor can
only ever make a host spend **more**, never less.

---

## Two call sites, one set of weights

| marker | when | input | reads |
| --- | --- | --- | --- |
| `[PRE]` | before any model is called | the user's turn | `route` |
| `[ACT]` | after a model proposes a tool call | turn + proposed action | `risk`, `gate` |

Routing must happen *before* the expensive call. Risk can only be judged *after*
something has been proposed. One encoder serves both, distinguished by a marker
token.

```
[PRE] Quick question: is this migration safe to run on prod under active load?
[ACT] summarise the log [SEP] read ./app.log then pipe what it emits into the shell
```

## Outputs

| head | type | values |
| --- | --- | --- |
| `route` | 3-way softmax | `NO_MODEL` · `SMALL` · `LARGE` |
| `risk` | 3-way softmax | `P2_READONLY` · `P1_MUTABLE` · `P0_DESTRUCTIVE` |
| `gate` | sigmoid | probability a human should confirm |
| `abstain` | sigmoid | **in the graph, not trained — see Limitations** |

Route tiers are deliberately abstract. A published router cannot know what
engines you run; it reports how much machine a turn needs and you map that onto
what you have.

---

## Usage

```python
from littleleo.pipeline import LittleLeoRouter

router = LittleLeoRouter("path/to/model")

router.route("Quick question: is this migration safe to run on prod?")
# {'route': 'LARGE', 'encoder_route': 'LARGE', 'confidence': 0.9982,
#  'abstained': False, 'source': 'encoder', 'latency_ms': 7.2}

router.route("thanks")
# {'route': 'NO_MODEL', 'confidence': 1.0, 'source': 'fast_path', 'latency_ms': 0.01}

router.assess("summarise the log",
              "read ./app.log then pipe what it emits straight into the shell")
# {'risk': 'P0_DESTRUCTIVE', 'rules_said': 'P0_DESTRUCTIVE',
#  'source': 'rules', 'requires_confirmation': True}
```

Requires `onnxruntime` and `tokenizers`. No PyTorch, no `transformers`, no
network access.

`encoder_route` reports what the model said before the floor was applied, so a
host can see when a decision was overridden and why.

---

## The artifact

| | |
| --- | ---: |
| `model.onnx` | 90.4 MB, FP32 |
| Latency p50 / p95 / p99, one thread | **7.21 / 10.27 / 17.81 ms** |
| Parameters | 22,716,296 |

### An INT8 build was produced and withdrawn

It was 22.9 MB and 2.9 ms, with **identical safety metrics** — same P0 recall,
same zero regret. It routed **12.9% more expensively**, breaching the 5%
operational bound in the release gate. Every flipped decision went the safe way,
so it was never a correctness problem; it simply returned more of the saving
than the gate permits.

Shipping it would have meant publishing an artifact that fails its own published
gate, or widening the gate to admit it. Neither was acceptable, so only FP32 is
released.

---

## Synthetic evaluation

Held-out splits with **zero verbatim overlap with training**, enforced by an
assertion that fails the build. `audit` uses clause wordings drawn from a pool
nothing else has seen.

Mean cost folds savings and quality regret into one figure via the declared cost
matrix. Lower is better; **1.032 is the score for never routing at all.**

| router | accuracy | downgraded | regret | mean cost |
| --- | ---: | ---: | ---: | ---: |
| always call large *(status quo)* | 0.391 | 0.000 | 0.000 | 1.032 |
| handwritten regex rules | 0.391 | 0.000 | 0.000 | 1.032 |
| TF-IDF + logistic regression | 0.823 | 0.658 | 0.109 | 3.630 |
| TF-IDF + linear SVM | 0.830 | 0.645 | 0.099 | 3.232 |
| **LL-22M** | **0.874** | 0.593 | **0.0000** | **0.143** |

**Bag-of-words is worse than not routing.** Both TF-IDF baselines save over 60%
of calls and still score ~3× worse than the status quo, because a 10–11%
downgrade rate against a 20× penalty costs more than the savings return.

**The regex rules fail safe rather than badly.** On unfamiliar phrasing none of
their patterns match, so they default to `LARGE`: zero regret, zero savings,
identical to not routing.

### By category, audit split

| category | n | accuracy |
| --- | ---: | ---: |
| `hard` | 101 | 1.000 |
| `simple` | 111 | 1.000 |
| `hard_negative_large` — sounds trivial, is hard | 50 | 1.000 |
| `scope_restricted` — "only" restricting source, not output | 41 | 1.000 |
| `comprehension_transform` — transform needing understanding | 40 | 1.000 |
| `conversational` | 75 | 1.000 |
| `hard_negative_trivial` — sounds heavy, is trivial | 50 | 0.820 |
| `trivial` | 126 | 0.476 |

`trivial` at 0.476 is the weakest cell and it is the deliberately exotic
synthetic phrasing (*"confine the response to one syllable"*). On the live
fixture's natural phrasings — *"Reply with the digit only"* — the model was
correct on every one. **The synthetic audit is harder than reality on this
axis**, so the real-world gap is smaller than 0.476 suggests.

---

---

## What `NO_MODEL` means for a host

`NO_MODEL` is the most easily misread output in this model, and misreading it is
the one failure mode that reaches the user directly.

**It means: no generative call is needed to satisfy this turn.** It does not
mean "send nothing". The host still replies — from a canned acknowledgement, a
template, a cache, or an existing deterministic path. A host that wires
`NO_MODEL` to silence has not saved a call, it has dropped a turn, and the user
sees nothing at all.

This distinction is not theoretical. Before the confidence floor existed, 5.1%
of fixture prompts were routed `NO_MODEL` for tasks that genuinely needed an
answer. Zero dollars spent, and a user staring at a blank reply. The floor
removed those cases; it does not remove the host's obligation to respond.

The three tiers describe **how much machine a turn needs**, not whether it
deserves a reply:

| tier | meaning | host obligation |
| --- | --- | --- |
| `NO_MODEL` | no generative call needed | still reply, from a non-generative path |
| `SMALL` | a cheap model suffices | reply from the cheap engine |
| `LARGE` | needs the strong model | reply from the frontier engine |

**A safe default for a first deployment is to map `NO_MODEL` onto your cheapest
model rather than onto a template.** That gives up part of the saving and keeps
the behaviour identical to what the host did before. `encoder_route` still
reports what the model actually said, so the saving that was declined remains
measurable before it is taken.


---

## The input window

The ceiling is **256 tokens**, and it **truncates rather than pads**.

Every tokenizer call in this repository passes `padding="longest"`, so a turn
costs what its own length costs. A 12-token turn is a 12-token tensor whatever
the ceiling says. Raising the ceiling is therefore free for short traffic and is
paid for only by turns that are genuinely long.

It was 64 in earlier revisions. That was not a measured choice: the comment
defending it argued that a wider ceiling would spend "four times the attention
arithmetic on padding", while stating two lines below that batches pad to their
own longest member. Both cannot be true, and the second is the true one. 64 was
chosen against a cost that was never being paid.

**Truncation is unusually destructive for this architecture.** The corpus puts
the clause that decides the tier at the *end* of the turn — a collapser, a
qualifier or a transform attached to a shared lead and object. Clip the tail and
what remains is a bare lead and object, which is ambiguous across all three
tiers by construction. A truncated turn is not a degraded input; it is a
confidently mislabelled one.

**What the wider ceiling does not do.** No training example exceeds 47 tokens
and the median is 23. The ceiling prevents clipping; it cannot confer competence
on lengths the model has never seen. Long discursive turns remain
out-of-distribution, and on a small sample of realistic long turns the model was
wrong at 55 and 60 tokens — *below* the old ceiling, so truncation was never the
cause. Closing that gap needs long-form turns in the corpus and a retrain, not a
larger number here.

Treat 256 as the removal of a silent failure mode, not as support for long
input.


## Limitations

**1. Labels are asserted, then partially grounded.** Training labels were
generated by rule. The live test checked *downgrade quality* against real model
outputs on 39 items — which validates the routing decisions, not the tier
definitions themselves.

**2. One model pair.** 22.5% is the Gemini-2.5-Flash-to-GPT-5.4 gap. A narrower
pairing yields a smaller saving; "cheap" here means cheap per token, not weak.

**3. The quality interval is wide** — [3.5%, 25.6%] on 30 judgements, with
run-to-run variance from 3.3% to 17% on identical inputs.

**4. 39 fixture items.** Enough for a defensible figure; not enough to claim it
transfers to arbitrary traffic.

**5. The encoder alone misses destructive actions** (0.726 P0 recall on the test
pool). Mitigated entirely by the rules floor, and only by it.

**6. The `abstain` head is untrained.** It is in the graph so the exported shape
is final, but the corpus has no out-of-distribution labels. `abstained` comes
from confidence and entropy — uncertainty signals, **not** OOD detection.

**7. Synthetic training data, one domain.** Software-engineering and
data-operations flavoured. Performance on support, medical or legal text is
unknown and should be assumed poor until measured.

**8. English only.**

**9. Long turns are out of distribution.** The 256-token ceiling stops input
being clipped, but no training example exceeds 47 tokens. Turns substantially
longer than the corpus are unmeasured, and two realistic long turns were
misrouted at 55 and 60 tokens — below even the old ceiling, so length itself,
not truncation, is the unfixed gap.

**10. The evaluation is largely self-graded.** The same author wrote the
generator, the baselines and the model. The judge is an independent vendor; the
fixture is not.

---

## What was thrown away

The first corpus produced a model scoring **1.000 on every category including
held-out splits**. That is a symptom, not a result:

- **96.7% of the test split appeared verbatim in training.** Both decisive
  categories were 100% leaked.
- **TF-IDF also scored 1.000**, including on categories that were *not* leaked.
  Difficulty was signalled by vocabulary, so word counts sufficed.

The rebuild shares leads and objects across every tier and makes the label
depend on an attached clause, with clause wordings drawn from split-disjoint
pools. TF-IDF now degrades from 1.891 on test to 3.232 on audit — the evidence
the task needs semantics rather than vocabulary.

Both TF-IDF baselines are permanently locked into the harness so this cannot
recur silently.

---

## Training

| | |
| --- | --- |
| Trunk | `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) |
| Pooling | masked mean — **not** `[CLS]` |
| Objective | expected cost under the declared matrix + 0.2 × cross-entropy |
| Corpus | 9,300 synthetic turns, four disjoint splits |
| Epochs | 4, batch 32, lr 3e-5 trunk / 1e-3 heads |
| Hardware | 12 CPU cores, no GPU, 19 minutes |

The objective *is* the evaluation metric, so the quantity minimised and the
quantity reported cannot drift apart.

## Licence

Apache-2.0, as is the base trunk.
