# Sentinel-22M — specification

> Status: **specified, nothing trained.** Phase 0 is a ~$1 go/no-go gate and must
> run before any other work. Nothing below is a commitment to train; it is a
> commitment to *what would have to be true* before training is justified.

---

## 1. One sentence

Sentinel-22M is a 22.7M-parameter bidirectional encoder that reads the **output of
a tool call** and says whether the agent should **continue, retry, or stop**.

It is the sibling of LL-22M. LL-22M decides *how much model a turn needs*, before
anything runs. Sentinel decides *whether the last step actually worked*, after
something ran. Same trunk, same licence, same safety posture, different question
and a different call site.

---

## 2. The failure this exists for

**Exit code 0 with a stack trace in stdout.**

A regex on stderr scores that a success. The agent proceeds on a corrupt
observation, and every subsequent step is reasoning about a result that does not
exist. The damage is not the failed command; the damage is the six steps after it.

The same shape recurs:

- a write that reports success but truncated
- an empty result that is correct, next to an empty result that means the query broke
- a partial file write followed by a clean exit
- a retryable network blip wearing the same words as a permanent auth failure

None of these are separable by pattern-matching on the text alone — which is the
claim this project has to earn in Phase 2, and may well fail there.

---

## 3. Contract

### Classes

| class | ordinal | meaning | agent does |
| --- | ---: | --- | --- |
| `SUCCESS` | 0 | the step did what it said | continue |
| `TRANSIENT_ERROR` | 1 | failed, plausibly recoverable | retry, possibly with changes |
| `FATAL_HALT` | 2 | failed, or succeeded corruptly; do not build on this | stop, surface to a human |

Ordered `SUCCESS < TRANSIENT_ERROR < FATAL_HALT`. The ordering is load-bearing: it
is what makes "escalate-only" a meaningful statement rather than a slogan.

### The safety invariant

```python
final = max(deterministic_rules(output), model(output))
```

**The model may escalate. It may never downgrade.** A host that deletes Sentinel
degrades to exactly the behaviour it had before installing it: whatever its own
exit-code rules already said. The model can only ever make the host stop *more*
often, never less.

This is the same invariant LL-22M ships, and LL-149M is the reason to keep it.
Under adversarial conditions LL-149M's encoder-alone P0 recall collapsed to 0.370
while `regex→encoder` held at 1.000 recall and 1.000 precision. The rules floor had
previously been insurance nobody had ever claimed on. Now it has been claimed on.

### What Sentinel is not

Not a security boundary. Not a replacement for the host's own exit-code handling.
Not a judge of whether the *plan* is right — only of whether the *last observation*
is trustworthy.

---

## 4. Build

| | |
| --- | --- |
| Student trunk | `sentence-transformers/all-MiniLM-L6-v2` (22.7M, Apache-2.0) |
| Teacher | **Qwen** (Apache-2.0), single-token output + `logprobs` |
| Window | 256 tokens |
| Classes | 3, ordered |
| Licence | Apache-2.0 end to end |

### Why an Apache-2.0 teacher, specifically

OpenAI and Google terms restrict using model outputs to train competing models.
Distilling from either contaminates an Apache-2.0 release — the weights would carry
an obligation the licence does not disclose. Claude exposes no logprobs, so it
cannot be a soft-label teacher at all, whatever its terms said.

Qwen is Apache-2.0, exposes `logprobs`, and is cheap. That is the whole argument.

### One head, not four

Stagnation detection needs multi-step history. Payload compression needs ~50k
tokens of context. Parameter sanity-checking needs ~100. These are incompatible
scopes, and bundling them makes every cheap decision pay the expensive one's
context cost — which, per §5, *is* the cost.

One head. Ship it. Then argue about the second.

---

## 5. Latency budget — and the number that was wrong

The original target was **8 ms at 1024 tokens**. That was off by **60–250×**.

Measured on this machine, one core: ModernBERT-base took **73.20 ms at 43 tokens**.
At 1024 tokens the same model is ~1.8 s. LL-22M's 7.21 ms p50 is also at ~43
tokens, not at its 256-token ceiling.

**Sequence length dominates latency, not parameter count.** Every latency figure in
this project is meaningless without the token count beside it, and any figure
quoted without one should be treated as unmeasured.

The design consequence: Sentinel is a **per-step supervisor**, not a hot-path
component. It runs once per tool call, against an agent step that already took
multiple seconds. **500 ms is an acceptable budget.** That is what buys the
256-token window; it is not a window that would survive a hot-path requirement.

Measure on a quiet machine. This box has ~50% timing jitter — re-measure a known
anchor before trusting any new latency figure.

---

## 6. Distillation is training. There is no weight capture.

To be explicit, because the opposite is an easy thing to assume:

**Qwen's tensors cannot be copied into a 22M encoder.** Different architecture,
different width, different vocabulary. Nothing is transplanted.

What is free is the **trunk** — MiniLM's pretrained weights, exactly as in LL-22M.
Teacher knowledge arrives *only* through gradient descent on teacher-labelled
examples. The teacher is a labelling function, not a donor.

Which sets the critical path: **training is cheap (~20 min on CPU); trace
collection is the bottleneck.** Budget accordingly — see §9.

### Soft labels, not hard

Hard labels teach argmax. Argmax teaches confidence everywhere. Confidence
everywhere is **exactly the LL-149M failure**: its wrong `NO_MODEL` calls sat at
**0.998 and 0.989**, far above the 0.90 confidence floor, and six users got
silence. LL-22M's wrong calls sat at 0.481 and 0.735, and the floor caught them.

A memorised model is wrong with conviction, and against conviction every
confidence-threshold defence silently stops working. Soft labels transfer the
teacher's *hedging*, which is the only thing that makes a downstream confidence
floor mean anything.

This is why Phase 0 gates on calibration and not only on accuracy. A teacher that
is 0.99-confident and 70% right is worse than useless: distillation transfers the
confidence along with the errors.

---

## 7. Training objective

Extends the cost-matrix pattern already in `littleleo/schema.py`:

```
L = Σ_t q(t) · Σ_p student(p) · C[t, p]  +  λ · KL(q ‖ student)
```

- `q` — Qwen's distribution over the three classes (not its argmax)
- `C` — the declared cost matrix, written **before training**, as in LL-22M
- first term — expected cost marginalised over teacher *uncertainty*
- second term — preserves the student's shape, not just its decisions

The cost matrix must be declared in `sentinel/schema.py` before the first training
run. Thresholds derived from a matrix chosen afterwards are thresholds chosen to
flatter the model.

**The objective is the eval metric.** This is the most portable finding from
LL-149M: it scored *higher accuracy* than LL-22M (0.8855 vs 0.8737) while costing
**130.6% more** (0.3300 vs 0.1431), because the cost matrix charges a downgrade 20×
an over-service and the bigger model traded cheap errors for expensive ones. An
accuracy table would have shipped it. Sentinel does not get an accuracy table.

---

## 8. Data

### Splits — as in `littleleo/synth.py`

Split-disjoint **by tool family**, not by row:

| split | pool |
| --- | --- |
| train, dev | tool families **A** |
| test | tool families **B** |
| audit | tool families **C** — never seen by anything |

`assert_no_overlap()` fails the build rather than warning. A warning printed into a
long log is a warning nobody reads, and the previous Little Leo corpus reached
96.7% verbatim train/test overlap before that guard existed.

Note that `dev` being drawn from pool A means dev saturates early and is a poor
early-stopping signal. See §9, Phase 1 — this exact property cost LL-149M its run.

### Sources, in order of preference

1. **Leo's own execution history** — real, licence-clean, in-domain
2. **Public trace corpora** — SWE-bench, ToolBench, WebArena. **Check
   redistribution terms before use**; a corpus that cannot be redistributed can
   still be trained on, but it changes what the model card is allowed to claim.
3. **Frontier-generated edge cases** — to fill gaps, not to form the bulk

Target 15–20k traces.

---

## 9. Phases

### Phase 0 — teacher check. ~$1. Run this first.

Script: `sentinel/scripts/phase0_teacher_check.py`

Hand-label **100 real tool results**. Label them **before** seeing Qwen's answers,
or you are measuring your own anchoring rather than the teacher's agreement.

**Over-sample hard cases** — exit-0-with-traceback, empty-but-successful, partial
writes. A sample that is 90% clean successes will score 0.95 agreement and teach
you nothing; the script refuses a skewed sample unless explicitly overridden, for
exactly that reason.

Input, one JSON object per line:

```json
{"tool":"bash","args":"pytest -q","output":"...","human":"FATAL_HALT","note":"exit 0, traceback in stdout"}
```

The script queries Qwen with `max_tokens=1`, `logprobs=true`, `top_logprobs=20`,
maps the returned tokens `A`/`B`/`C` onto the three classes, extracts the
probability distribution, and gates on:

1. **Teacher–human agreement ≥ 0.85**
2. **Confidence is informative** — accuracy must rise with confidence
3. **No class below 0.70 recall**

Fail any one → **do not distil.** There is no partial credit: a teacher failing
gate 2 produces a confidently-wrong student, which is the single failure mode this
whole design exists to avoid.

#### Two OpenRouter facts, found the hard way in a trial run

**`provider: {"require_parameters": true}` is mandatory.** OpenRouter load-balances
one model id across several upstream providers, and they do not all implement
`top_logprobs`. One that does not returns a perfectly good *letter* with an empty
`logprobs` block. In a six-row trial, three rows came back with no distribution at
all — silently, because the answer itself was correct. A soft-label pipeline that
quietly collects hard labels for half its corpus is precisely the failure this
phase exists to catch. The flag is pinned in `ask()`; the run also gates on
`answers_usable` so a residual drop cannot pass unnoticed.

**Pin one provider before quoting calibration numbers.** Even with logprobs
available, consecutive calls were served by Parasail, GMICloud and StreamLake —
differently-quantised builds of nominally the same model. That is noise for
agreement and *not* noise for ECE, because the confidences being compared then come
from different models. `--provider` pins one and disables fallbacks; the script
warns whenever a run was served by more than one.

### Phase 1 — traces. 2–4 days. **The bottleneck.**

Collect 15–20k traces under the split discipline in §8. This is the critical path;
every other phase is hours, and this one is days.

**Checkpoint per epoch, and early-stop on `audit`, not `dev`.** `scripts/train.py`
currently checkpoints only after the final epoch and early-stops on dev — which is
pool A and saturates at epoch 1. LL-149M's run therefore flew blind and landed on a
memorised checkpoint with zero training loss by epoch 2. Fix the harness before the
first Sentinel run, not after it.

### Phase 2 — controls first, then train. 1 day.

**Run the controls before the training run, not after:**

- exit-code regex
- TF-IDF + LogisticRegression
- TF-IDF + LinearSVC

If TF-IDF solves it, the task is lexical and a transformer is dead weight. That is
a real result that cost an afternoon instead of a training run, and it is worth
publishing either way.

For calibration of expectations: on LL-22M the regex control scored 0.391 and
**both TF-IDF baselines came out worse than not routing at all.** That is the bar a
control has to clear to kill a project, and it has been cleared before.

### Phase 3 — export and gates. 1–2 days.

Reuse `scripts/export_onnx.py` and its gates unchanged — metric invariance rather
than bit-level identity, the zero-regret bound, relative mean-cost drift.

**Plus one new gate: calibration.**

- ECE below a declared bound
- a published risk–coverage curve

**Enforce it, do not merely report it.** Almost no model card reports calibration at
all; a model that *gates* on it is making a claim its competitors are not. This is
the differentiator, and it is worth nothing if it is advisory.

---

## 10. The risk to watch

**Real traces are the binding constraint.**

Distillation fixes *labels*. That is genuinely worth doing — it solves LL-22M model
card limitation #1, where every label was asserted by a rule, meaning the model
could only ever learn the rule.

Distillation does nothing for *inputs*. Train on synthetic traces and you get a
beautifully calibrated classifier **of synthetic traces**, with an honest
calibration curve for a distribution nobody runs.

**Before Phase 1 starts, establish how many real tool-execution traces Leo can
actually supply.** If the answer is a few hundred, the design changes:

- fewer classes (possibly binary: trustworthy / not)
- a tighter declared domain (one tool family, named in the card)
- honest scope in the model card, stating the trace count

That is a smaller project, not a failed one. Discovering it in Phase 1 is cheap.
Discovering it in Phase 3 is not.

---

## 11. Decisions already taken — do not relitigate

- **The name is Sentinel-22M**, not AgentSentinel-100M. The artifact is 22.7M and
  the name has to match the thing.
- **500 ms, not 8 ms.** Per-step supervisor, not hot path. See §5.
- **Distillation is training.** No weight capture. See §6.
- **One head first.** See §4.
- **Apache-2.0 teacher, mandatory.** See §4.
- **Soft labels, mandatory.** See §6.
- **Escalate-only, mandatory.** See §3.
- **Controls before training.** See §9, Phase 2.

---

## 12. Related

- `../MODEL_CARD.md` — LL-22M, the shipping sibling
- `../LL-149M-FINDINGS.md` — why bigger failed, in detail
- `../littleleo/schema.py` — the contract pattern this spec follows
- `../littleleo/synth.py` — split-disjoint pools and `assert_no_overlap()`
- `../scripts/export_onnx.py` — the release gates Phase 3 reuses
