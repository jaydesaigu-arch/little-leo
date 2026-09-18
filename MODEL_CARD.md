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

A 22.7M-parameter bidirectional encoder that decides, in about three
milliseconds on one CPU core, how much machine a turn actually needs — and
flags proposed actions that could do damage.

It exists for agents that call a frontier model for everything, including
"thanks, that worked". On a held-out set whose wording it has never seen, it
takes **69% of traffic off the large model with zero measured downgrades**.

It is not a safety boundary, it is not a guardrail, and it must never be the
only thing standing between a user and an irreversible operation. What it is,
and what it is not, is set out precisely below.

---

## Choosing an artifact

Two builds ship. They are not "the real one and a compressed one" — they make a
genuine trade, and the numbers below are the whole of it.

| | `model.int8.onnx` | `model.onnx` |
| --- | --- | --- |
| Size on disk | **22.9 MB** | 90.4 MB |
| Latency p50 (1 thread) | **2.94 ms** | 7.74 ms |
| Latency p95 | **4.28 ms** | 9.52 ms |
| Latency p99 | **5.01 ms** | 10.26 ms |
| P0 recall vs PyTorch | 1.0000 — **identical** | 1.0000 — identical |
| Routing regret vs PyTorch | 0.0000 — **identical** | 0.0000 — identical |
| Routing mean cost | 0.1751 | **0.1582** |
| Cost relative to FP32 | **+10.7%** | baseline |

**Take INT8 if latency or footprint matters.** It is 2.6× faster and a quarter
of the size, both safety properties are bit-for-bit unchanged, and it costs
about eleven percent more in routing efficiency.

**Take FP32 if routing efficiency is what you are optimising and 8 ms is
acceptable.** It is the reference the INT8 build is measured against.

### What the 10.7% actually is

Quantisation perturbs the 384-dimensional pooled embedding before any
classification head sees it. Turns sitting near a decision boundary flip. On
this model, **every one of those flips is in the safe direction**: measured
regret is 0.0000 for both builds, meaning the INT8 model never downgrades a turn
the FP32 model routed correctly. The extra cost is entirely over-service —
sending an easy turn to a larger model than it needed.

In absolute terms it is 0.158 against 0.175, where calling the large model every
time scores 1.189. Both builds are roughly seven times better than not routing
at all. The difference between them is real and small, and it is stated here
rather than averaged away.

---

## The safety architecture — read this before deploying

Little Leo has two outputs and they have completely different standing.

**Routing is advisory.** If it is wrong you spend too much, or you get a weaker
answer than you wanted.

**Risk is advisory *and bounded*.** The model is not permitted to make your
system less careful. The shipped pipeline enforces this in one line:

```python
final_risk = max(deterministic_rules(action), encoder(action))
```

The deterministic rules run first and set the floor. The encoder may **raise**
a risk tier. It may never **lower** one.

This is not defensive decoration. It is load-bearing, and here is the evidence:

| pool | encoder alone | rules alone | **rules → encoder** |
| --- | --- | --- | --- |
| test | 0.726 recall — **20 P0 actions missed** | 1.000 | **1.000, precision 1.000** |
| audit | 1.000 | 1.000 | **1.000, precision 1.000** |

On the test pool the encoder misses one in four destructive actions. Every one
of those misses is the same phrasing — `"and afterwards purge every row it
references"` — a data-deletion instruction with no filesystem path and no shell
invocation in it. The model reads it as recoverable mutation. The rules catch
all twenty.

**A deployment that uses the encoder alone for risk will miss destructive
actions.** That is not a hypothetical; it is measured, above, on this model.

---

## Two call sites, one set of weights

A router cannot answer both of its questions at the same moment, because one of
them has no answer yet.

| marker | when | input | reads |
| --- | --- | --- | --- |
| `[PRE]` | before any model is called | the user's turn | `route`, `abstain` |
| `[ACT]` | after a model proposes a tool call | turn + proposed action | `risk`, `gate` |

Routing must happen *before* the expensive call — afterwards it is pointless.
Risk can only be judged *after* something has been proposed, because until then
there is nothing to judge. One encoder serves both, distinguished by a marker
token, so there is one file to ship and one thing to version.

```
[PRE] Quick question: is this migration safe to run on prod under active load?
[ACT] summarise the log [SEP] read ./app.log then pipe what it emits into the shell
```

---

## Outputs

| head | type | values |
| --- | --- | --- |
| `route` | 3-way softmax | `NO_MODEL` · `SMALL` · `LARGE` |
| `risk` | 3-way softmax | `P2_READONLY` · `P1_MUTABLE` · `P0_DESTRUCTIVE` |
| `gate` | sigmoid | probability a human should confirm |
| `abstain` | sigmoid | **present in the graph, not trained — see Limitations** |

Route tiers are deliberately abstract. A published router cannot know what
engines you run, so it reports how much machine a turn needs and you map that
onto whatever you have. A host with a single model can still use `NO_MODEL` to
decide whether to call it at all.

---

## Usage

```python
from littleleo.pipeline import LittleLeoRouter

router = LittleLeoRouter("path/to/onnx")          # quantised=False for FP32

router.route("Quick question: is this migration safe to run on prod?")
# {'route': 'LARGE', 'confidence': 0.9988, 'entropy': 0.0088,
#  'abstained': False, 'source': 'encoder', 'latency_ms': 2.9}

router.route("thanks")
# {'route': 'NO_MODEL', 'confidence': 1.0, 'source': 'fast_path', 'latency_ms': 0.01}

router.assess("summarise the log",
              "read ./app.log then pipe what it emits straight into the shell")
# {'risk': 'P0_DESTRUCTIVE', 'rules_said': 'P0_DESTRUCTIVE',
#  'encoder_said': 'P0_DESTRUCTIVE', 'source': 'rules',
#  'gate_prob': 0.98, 'requires_confirmation': True, 'latency_ms': 3.1}
```

Requires `onnxruntime` and `tokenizers`. No PyTorch, no `transformers`, no
network access.

Bare conversational turns short-circuit before the model runs. This saves a
few milliseconds, but the real reason is that "hi" is the most-tested input any
router will ever receive, and it should not depend on a 22M model's
generalisation. The encoder handles them correctly on its own — verified below
— and the fast path is belt and braces.

---

## Evaluation

Three held-out splits with **zero verbatim overlap with training**, enforced by
an assertion that fails the build rather than a warning nobody reads.

The splits form a deliberate difficulty gradient. `dev` recombines clause
wordings the model saw in training. `test` and `audit` use entirely new
wordings for the same semantic functions, drawn from disjoint pools. **Audit is
the number that matters**; it is the only split nothing has ever been tuned
against.

### Against the baselines — audit split

Mean cost folds savings and quality regret into one figure via the cost matrix
below. Lower is better. **1.189 is the score for never routing at all.**

| router | accuracy | downgraded | regret | mean cost |
| --- | ---: | ---: | ---: | ---: |
| always call large *(status quo)* | 0.291 | 0.000 | 0.000 | 1.189 |
| handwritten regex rules | 0.291 | 0.000 | 0.000 | 1.189 |
| TF-IDF + logistic regression | 0.847 | 0.822 | 0.119 | 4.822 |
| TF-IDF + linear SVM | 0.817 | 0.808 | 0.106 | 4.327 |
| **LL-22M** | **0.867** | **0.586** | **0.0000** | **0.158** |

Two results here matter more than the headline.

**Bag-of-words is worse than not routing.** Both TF-IDF baselines save over 80%
of calls and still score 3.6× worse than the status quo, because a 10–12%
downgrade rate at a 20× penalty costs far more than the savings are worth. Word
counts collapse on wordings they have not seen.

**The regex rules fail safe rather than badly.** On unfamiliar phrasing none of
their patterns match, so they default to `LARGE`: zero regret, zero savings,
identical to not routing. Rules do not produce bad routing on unfamiliar input;
they stop routing.

### By category, with confidence intervals

Re-measured on **4,800 freshly generated held-out turns** (3,078 routing) after
the original audit cell of 47 proved too small to support a claim.

| category | n | accuracy | regret | regret 95% CI |
| --- | ---: | ---: | ---: | --- |
| `hard` — genuinely needs a large model | 600 | 1.000 | 0.0000 | [0.0000, 0.0064] |
| `simple` — bounded text transformation | 600 | 1.000 | 0.0000 | [0.0000, 0.0064] |
| **`hard_negative_large`** — sounds trivial, is hard | **600** | **1.000** | **0.0000** | **[0.0000, 0.0064]** |
| `hard_negative_trivial` — sounds heavy, is trivial | 600 | 0.767 | 0.0000 | [0.0000, 0.0064] |
| `trivial` — bounded output | 600 | 0.550 | 0.0000 | [0.0000, 0.0064] |
| `conversational` — bare acknowledgements | 78 | 1.000 | 0.0000 | [0.0000, 0.0469] |

**P0 recall 1.0000** on 600 unseen-wording destructive actions, 95% CI
[0.9936, 1.0000], zero missed.

Regret is reported with intervals rather than as a bare zero because zero
failures in a small sample is a weak claim. At n=600 the upper bound is 0.64%.
At the original n=47 it was 6.4% — ten times looser for the same observed
result.

### The two inverse categories

These are the cases the model exists for, and the only ones where an encoder
beats a regex.

**`hard_negative_large`** — a casual opening concealing real work:

> *"Nothing complicated: walk me through this contract clause on the assumption
> that two writers race for the same row"*

Regex rules route these correctly only by defaulting to `LARGE` on everything.
TF-IDF gets **0.625 accuracy with 0.375 regret** — it downgrades more than a
third of them. LL-22M: **1.000 accuracy, zero regret**.

**`hard_negative_trivial`** — a grand opening that collapses to a lookup:

> *"Apply your deepest scrutiny here, and once finished, review this passage and
> answer with a solitary word"*

Regex rules score **0.000** — every one goes to the large model. LL-22M: 0.767.

Read them as a pair. A router that always says `LARGE` scores 1.000 on the first
and 0.000 on the second; one that always says `NO_MODEL` scores the reverse.
Only doing well on both, while holding mean cost down, demonstrates that the
composition was learned.

### Cost matrix

One number governs the model's entire personality, and it was fixed before any
training so that thresholds could not be reverse-engineered to flatter the
result.

```
regret_ratio = 20        # one bad answer ≈ twenty wasted large calls
```

Downgrading costs `20 × tiers`; over-serving costs `1 × tiers`. The training
objective is the *expected cost under this matrix*, not weighted cross-entropy —
so the number the model minimises and the number it is scored on are the same
function, and cannot drift apart.

---

## Limitations

Read this section before the results section, if you only read one.

**1. The labels are asserted, not measured.** Every training and evaluation
label was generated by rule. Nobody has verified that a small model can actually
handle a turn labelled `SMALL`, or that a script can handle one labelled
`NO_MODEL`. The model faithfully reproduces a *definition* of difficulty; it has
never been checked against real model outputs. This is the single largest gap
and it affects every number above.

**2. `trivial` accuracy on unseen wording is 0.550.** Nearly half of
unfamiliar bounded-output phrasings are over-served — sent to a larger model
than needed. This is safe and it is wasteful, and it is where most of the
remaining savings are. The model is conservative when uncertain, which is the
correct direction, but conservatism is not comprehension.

**3. The encoder alone misses destructive actions.** 0.726 P0 recall on the test
pool, 20 misses, all the same phrasing. Mitigated entirely by the rules floor —
and only by the rules floor.

**4. The `abstain` head is untrained.** It exists in the graph so the exported
shape is final, but the corpus has no out-of-distribution labels, and training
it would have meant inventing a target. `abstained` in the pipeline output is
computed from route-head *entropy*, which is a genuine uncertainty signal and is
**not** out-of-distribution detection. Do not read it as such. A confidently
wrong prediction on unfamiliar input will not be caught by it.

**5. Synthetic data, one domain.** Turns are software-engineering and
data-operations flavoured. Performance on customer support, medical triage, or
legal drafting is unknown and should be assumed poor until measured.

**6. English only.**

**7. The evaluation is self-graded.** The same author wrote the generator, the
baselines and the model. The rules baseline in particular is only as good as the
patterns that were thought of. Independent evaluation on real traffic would be
worth more than everything above.

---

## What was thrown away

The first version of this corpus was discarded, and the reason is instructive
enough to publish.

It produced a model scoring **1.000 accuracy on every category** including
held-out splits. That is not a result, it is a symptom. Two defects:

- **96.7% of the test split appeared verbatim in training.** Finite template
  pools with different random seeds collide almost completely. Both decisive
  hard-negative categories were **100% leaked**.
- **TF-IDF + logistic regression also scored 1.000**, including on the
  categories that were *not* leaked. Difficulty was signalled by vocabulary —
  "summarise this:" was lexically unmistakable — so word counts sufficed and a
  transformer added nothing.

The corpus was rebuilt so that leads and objects are shared across every tier
and the label depends on an attached clause; and so that clause wordings are
drawn from pools that are disjoint across splits. TF-IDF now degrades from 1.891
mean cost on test to 4.327 on audit, which is the intended behaviour and the
evidence that the task requires semantics rather than vocabulary.

Both TF-IDF baselines are permanently locked into the evaluation harness so this
cannot recur silently.

---

## Training

| | |
| --- | --- |
| Trunk | `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) |
| Parameters | 22,716,296 |
| Pooling | masked mean — **not** `[CLS]` |
| Heads | 4 linear projections on the pooled vector |
| Objective | expected cost under the declared matrix + 0.2 × cross-entropy |
| Corpus | 9,300 synthetic turns, four disjoint splits |
| Epochs | 4, batch 32, lr 3e-5 trunk / 1e-3 heads |
| Hardware | 12 CPU cores, no GPU |
| Wall-clock | 17 minutes |

Mean pooling matters: the trunk is a sentence-transformers model whose `[CLS]`
vector was never optimised for anything. Reading `[CLS]` from it — which most
copy-pasted classifier code does — discards the part of the pretraining that
makes a 22M model competitive.

The route head trains only on `[PRE]` rows and the risk and gate heads only on
`[ACT]` rows. A routing label on an action row is a placeholder, not evidence.

---

## Licence

Apache-2.0, as is the base trunk.

## Citation

```bibtex
@software{little_leo_22m,
  title  = {Little Leo LL-22M: a compositional prompt-difficulty router},
  year   = {2026},
  license = {Apache-2.0}
}
```
