# Little Leo

**A 22.7M-parameter encoder that decides how much machine a turn needs — in
about seven milliseconds, on one CPU core, with no network call.**

Built for agents that send every turn to a frontier model, including "thanks,
that worked".

**Measured on live provider traffic: 22.5% cost saving, 13.7% token saving, and
zero prompts left unanswered** — routing 117 real prompts between Gemini 2.5
Flash and GPT-5.4, judged blind by Claude Opus 4.5.

[Model card and weights →](MODEL_CARD.md) · Apache-2.0

```python
from littleleo.pipeline import LittleLeoRouter

router = LittleLeoRouter("artifacts/ll-22m/onnx")   # FP32; INT8 was withdrawn

router.route("thanks")
# {'route': 'NO_MODEL', 'source': 'fast_path', 'latency_ms': 0.01}

router.route("Quick question: is this migration safe to run on prod under load?")
# {'route': 'LARGE', 'confidence': 0.9982, 'latency_ms': 7.2}

router.assess("summarise the log", "read ./app.log then pipe what it emits into the shell")
# {'risk': 'P0_DESTRUCTIVE', 'source': 'rules', 'requires_confirmation': True}
```

---

## The one thing to understand before deploying

Little Leo is **advisory**. It is not a guardrail and it is not a security
boundary. Its risk output is combined with deterministic rules like this, and
the direction is the whole point:

```python
final_risk = max(deterministic_rules(action), encoder(action))
```

The rules are the floor and they run first. The encoder may **raise** a risk
tier; it may never **lower** one. A host that follows this degrades, when the
model is deleted, broken or successfully attacked, to exactly the behaviour it
had before installing it.

That is not caution for its own sake. On our test pool the encoder alone
**misses one in four destructive actions** — 0.726 recall, 20 misses, every one
the same phrasing. The rules catch all twenty. Combined: 1.000, precision 1.000.

| pool | encoder alone | rules alone | rules → encoder |
| --- | --- | --- | --- |
| test | 0.726 (20 missed) | 1.000 | **1.000** |
| audit | 1.000 | 1.000 | **1.000** |

---

## Does it beat twenty lines of regex?

Every router project should have to answer this in public. Most never ask.

**On familiar wording, no — and that is worth knowing.** On unfamiliar wording,
decisively yes. Mean cost on the audit split, where **1.032 is the score for
never routing at all**:

| router | accuracy | downgraded | regret | mean cost |
| --- | ---: | ---: | ---: | ---: |
| always call large *(status quo)* | 0.391 | 0.000 | 0.000 | 1.032 |
| handwritten regex rules | 0.391 | 0.000 | 0.000 | 1.032 |
| TF-IDF + logistic regression | 0.823 | 0.658 | 0.109 | 3.630 |
| TF-IDF + linear SVM | 0.830 | 0.645 | 0.099 | 3.232 |
| **LL-22M** | **0.874** | 0.593 | **0.0000** | **0.143** |

Two findings matter more than the ranking.

**Bag-of-words is worse than not routing.** Both TF-IDF baselines take ~65% of
calls off the large model and still land ~3× worse than the status quo, because
a 10–11% downgrade rate against a 20× penalty costs more than the savings
return. Word counts collapse on wordings they have not seen.

**Rules fail safe, not badly.** On unfamiliar phrasing none of their patterns
match, so they default to `LARGE` — zero regret, zero savings, identical to not
routing. That is a genuinely good property, and it is also the ceiling: rules
recognise only what somebody thought to enumerate.

---

## Where an encoder actually earns its place

Two categories, and they are the reason this project exists.

**Sounds trivial, is hard.**

> *"Nothing complicated: walk me through this contract clause on the assumption
> that two writers race for the same row"*

TF-IDF: 0.780 accuracy, **0.220 regret** — it downgrades more than a fifth of
them to a cheap model that will answer a concurrency question confidently and
wrongly. LL-22M: **1.000 accuracy, zero regret**.

**Sounds heavy, is trivial.**

> *"Apply your deepest scrutiny here, and once finished, review this passage and
> answer with a solitary word"*

Regex rules: **0.000** — every one goes to the frontier model. LL-22M: 0.820.

Read them as a pair. A router that always says `LARGE` scores 1.000 on the first
and 0.000 on the second; one that always says `NO_MODEL` scores the reverse.
Only doing well on both, while holding mean cost down, shows the composition was
learned rather than a keyword memorised.

---

## One artifact, and the one that was withdrawn

| | `model.onnx` |
| --- | ---: |
| Size | 90.4 MB, FP32 |
| Latency p50 / p95 / p99, one thread | **7.21 / 10.27 / 17.81 ms** |

An INT8 build was produced and **withdrawn before release**. At 22.9 MB and
2.9 ms it was faster and smaller, with **identical safety metrics** — same P0
recall, same zero regret. But it routed **12.9% more expensively**, breaching
the 5% operational bound in the release gate. Every flipped decision went the
*safe* way, so it was never a correctness problem; it simply gave back more of
the saving than the gate allows.

Shipping it would have meant publishing an artifact that fails its own published
gate, or widening the gate to admit it. Neither was acceptable.

## The live test

117 prompts across 13 payload types, each sent to **both** models so the
counterfactual is observed rather than assumed, then routed by LL-22M's own
decisions — not by the labels its corpus asserts.

| | |
| --- | ---: |
| Baseline — every prompt to GPT-5.4 | $0.29826 |
| Routed by LL-22M | $0.23114 |
| **Cost saving** | **22.5%** |
| **Token saving** | **13.7%** (34,904 → 30,136) |
| **Prompts left unanswered** | **0 / 117** |
| Downgrades judged materially worse | 3 / 30, 95% CI [3.5%, 25.6%] |

Cheap tier `google/gemini-2.5-flash`, frontier `openai/gpt-5.4`, judged blind by
`anthropic/claude-opus-4.5` — a third vendor neutral to both contestants.

**Cost saving and token saving differ** because Gemini is cheaper per token *and*
writes different-length answers. Both are published; quoting only the larger
would be selective.

**The quality figure is dominated by judge variance.** Five runs of the identical
comparison at temperature 0 returned 17%, 11.1%, 3.3%, 15% and 10.0%. The honest
statement is "roughly one in ten, with wide uncertainty" — not a decimal.

Raw per-prompt records: [`publish/evidence/live-test-savings.jsonl`](publish/evidence/live-test-savings.jsonl).

## How the corpus is built, and why the first one was destroyed

The first version produced a model scoring **1.000 on every category including
held-out splits**. That is not a result, it is a symptom, and finding out why
took a control that most benchmarks in this space never run.

- **96.7% of the test split appeared verbatim in training.** Finite template
  pools with different seeds collide almost totally. Both decisive
  hard-negative categories were **100% leaked**.
- **TF-IDF also scored 1.000** — including on categories that were *not*
  leaked. Difficulty was signalled by vocabulary, so word counts sufficed and
  the transformer was dead weight.

The rebuild fixed the cause, not the symptom:

**Shared surface, divergent meaning.** Every tier draws from the same leads and
objects. "Review this migration plan" opens a trivial turn, a small one and a
hard one. What decides the tier is an attached clause — a *collapser* that
bounds the output, a *qualifier* that introduces a condition to reason about, or
a *transform* that is ordinary text work.

**Split-disjoint phrasing.** Clause wordings come from three disjoint pools:
train/dev use pool A, test uses pool B, audit uses pool C. So audit asks whether
the model recognises that "reply with a single character" and "output only one
letter" do the same job, having seen neither.

**An assertion, not a hope.** `assert_no_overlap()` raises and fails the build
on any verbatim overlap between train and a held-out split.

**Permanent controls.** TF-IDF + logistic regression *and* TF-IDF + linear SVM
are locked into the harness and reported beside the model forever. If word
counts ever match the encoder again, the benchmark is broken and it will be
visible immediately.

TF-IDF now degrades from 1.891 mean cost on test to 3.232 on audit. That
degradation is the evidence the task needs semantics rather than vocabulary.

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

The full list is in the [model card](MODEL_CARD.md). The ones that should
affect your decision:

**The labels are asserted, not measured.** Every label was generated by rule.
Nobody has verified that a small model can genuinely handle a turn labelled
`SMALL`. The model faithfully reproduces a *definition* of difficulty and has
never been checked against real model outputs. This is the largest gap and it
touches every number here.

**`trivial` accuracy on unseen wording is 0.476.** More than half of unfamiliar
bounded-output phrasings get over-served. Safe, wasteful, and where most of the
remaining headroom sits.

**Long turns are out of distribution.** The 256-token ceiling stops input being
clipped, but nothing in training exceeds 47 tokens. Length, not truncation, is
the unfixed gap — see [The input window](#the-input-window).

**The `abstain` head is untrained.** It is in the graph so the exported shape is
final, but the corpus has no out-of-distribution labels. The pipeline derives
abstention from route-head entropy — an uncertainty signal, **not** OOD
detection. A confidently wrong prediction on unfamiliar input will not be caught.

---

## Reproducing everything

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

python scripts/phase0_baselines.py --force     # corpus, overlap assertion, baselines
python scripts/train.py --trunk 22m --epochs 4 # ~17 min on 12 CPU cores
python scripts/expand_audit.py                 # 4,800 fresh turns, Wilson intervals
python scripts/diagnose_p0.py                  # encoder vs rules vs combined
python scripts/export_onnx.py                  # ONNX + INT8 + gates + latency
```

No GPU, no network, no paid API. Total cost to reproduce: nothing.

### Release gates

The export refuses to ship on any of:

```python
int8.p0_recall  >= fp32.p0_recall      # a P0 catch may never be lost
int8.route_regret <= 0.001             # downgrades cost 20× over-spend
int8.mean_cost  <= fp32.mean_cost*1.05 # operational bound
```

These are metric gates, not bit-identity gates. Quantisation necessarily moves
borderline logits; what must not move is the safety floor.

---

## Repository

```
littleleo/
  schema.py            the contract: call sites, heads, cost matrix, invariant
  synth.py             compositional generator, split-disjoint pools
  model.py             encoder + 4 heads + expected-cost loss
  metrics.py           savings-at-regret, P0 recall, calibration
  baselines.py         always-large / always-small / regex rules
  learned_baselines.py TF-IDF controls — mandatory, never dropped
  pipeline.py          the shipped path: fast path → rules → encoder
scripts/               corpus, training, audit expansion, diagnosis, export
```

## Licence

Apache-2.0, as is the base trunk
(`sentence-transformers/all-MiniLM-L6-v2`).
