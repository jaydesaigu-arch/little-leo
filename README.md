# Little Leo

**A 22.7M-parameter encoder that decides how much machine a turn needs — in
about three milliseconds, on one CPU core, with no network call.**

Built for agents that send every turn to a frontier model, including "thanks,
that worked". On held-out wording it has never seen, it takes **69% of traffic
off the large model with zero measured downgrades**.

[Model card and weights →](MODEL_CARD.md) · Apache-2.0

```python
from littleleo.pipeline import LittleLeoRouter

router = LittleLeoRouter("artifacts/ll-22m/onnx")

router.route("thanks")
# {'route': 'NO_MODEL', 'source': 'fast_path', 'latency_ms': 0.01}

router.route("Quick question: is this migration safe to run on prod under load?")
# {'route': 'LARGE', 'confidence': 0.9988, 'latency_ms': 2.9}

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
decisively yes. Mean cost on the audit split, where **1.189 is the score for
never routing at all**:

| router | accuracy | downgraded | regret | mean cost |
| --- | ---: | ---: | ---: | ---: |
| always call large *(status quo)* | 0.291 | 0.000 | 0.000 | 1.189 |
| handwritten regex rules | 0.291 | 0.000 | 0.000 | 1.189 |
| TF-IDF + logistic regression | 0.847 | 0.822 | 0.119 | 4.822 |
| TF-IDF + linear SVM | 0.817 | 0.808 | 0.106 | 4.327 |
| **LL-22M** | **0.867** | **0.586** | **0.0000** | **0.158** |

Two findings matter more than the ranking.

**Bag-of-words is worse than not routing.** Both TF-IDF baselines save over 80%
of calls and still land 3.6× worse than the status quo, because a 10–12%
downgrade rate against a 20× penalty costs more than the savings return. Word
counts collapse on wordings they have not seen.

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

TF-IDF: 0.625 accuracy, **0.375 regret** — it downgrades more than a third of
them to a cheap model that will answer a concurrency question confidently and
wrongly. LL-22M: **1.000 accuracy, zero regret** on 600 unseen-wording samples,
95% CI [0.0000, 0.0064].

**Sounds heavy, is trivial.**

> *"Apply your deepest scrutiny here, and once finished, review this passage and
> answer with a solitary word"*

Regex rules: **0.000** — every one goes to the frontier model. LL-22M: 0.767.

Read them as a pair. A router that always says `LARGE` scores 1.000 on the first
and 0.000 on the second; one that always says `NO_MODEL` scores the reverse.
Only doing well on both, while holding mean cost down, shows the composition was
learned rather than a keyword memorised.

---

## Two artifacts, one honest trade

| | `model.int8.onnx` | `model.onnx` |
| --- | --- | --- |
| Size | **22.9 MB** | 90.4 MB |
| Latency p50 / p95 | **2.94 / 4.28 ms** | 7.74 / 9.52 ms |
| P0 recall vs PyTorch | identical | identical |
| Routing regret vs PyTorch | identical (0.0000) | identical (0.0000) |
| Routing mean cost | 0.1751 | **0.1582** |

INT8 costs **10.7% more in routing efficiency** for 2.6× the speed and a quarter
of the size. Both safety properties are unchanged, and measured regret is
0.0000 for both — so every one of those flipped decisions is in the *safe*
direction, over-serving rather than under-serving.

Pick on your constraint. Neither is the "real" one.

---

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

TF-IDF now degrades from 1.891 mean cost on test to 4.327 on audit. That
degradation is the evidence the task needs semantics rather than vocabulary.

---

## Limitations

The full list is in the [model card](MODEL_CARD.md). The three that should
affect your decision:

**The labels are asserted, not measured.** Every label was generated by rule.
Nobody has verified that a small model can genuinely handle a turn labelled
`SMALL`. The model faithfully reproduces a *definition* of difficulty and has
never been checked against real model outputs. This is the largest gap and it
touches every number here.

**`trivial` accuracy on unseen wording is 0.550.** Nearly half of unfamiliar
bounded-output phrasings get over-served. Safe, wasteful, and where most of the
remaining headroom sits.

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
