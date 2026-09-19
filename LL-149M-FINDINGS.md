# LL-149M — trained, measured, and not shipped

**Verdict: LL-149M fails the project's own release gate and is worse than
LL-22M on every axis that matters. It is not published. LL-22M (v5) remains the
release.**

This document exists because a negative result that is thrown away has to be
paid for twice. Everything below is reproducible from the artifacts in
`artifacts/ll-150m/` and the evidence files named at the end.

---

## What was run

| | |
| --- | --- |
| Trunk | `answerdotai/ModernBERT-base` |
| Parameters | 149,020,424 |
| Corpus | the v4 corpus, unchanged — 9,300 turns, four disjoint splits |
| Training | 4 epochs, batch 16, lr 3e-5 trunk / 1e-3 heads, 12 CPU cores |
| Wall clock | 8,038 s (2 h 14 m) |
| Cost of the live test | $0.4587 over 267 provider calls |

Nothing else changed. Same corpus, same cost matrix, same splits, same gates,
same fixture — so the comparison isolates the trunk.

---

## The headline

```
audit mean cost: ll-22m 0.1431  ->  ll-150m 0.3300
audit regret   : ll-22m 0.0000  ->  ll-150m 0.0051

ll-150m is 130.6% more expensive than ll-22m on the audit split.
```

The export gate refused it, in those words:

```
fp32  route_agreement 1.00000  risk_agreement 1.00000   passes: false
      failed_because: "regret violation: 0.0051 > 0.001"
```

Note what that says. The ONNX conversion is **perfect** — 1.00000 agreement
with PyTorch on both heads. The export is not the problem. The trained model
breaches the regret bound, and the gate that the 22M passed, unchanged, rejects
it.

---

## More accurate, and twice as expensive

This is the result worth remembering, and it is the best argument in the project
for why the training objective *is* the evaluation metric.

| audit | LL-22M | LL-149M |
| --- | ---: | ---: |
| **accuracy** | 0.8737 | **0.8855** |
| **mean cost** | **0.1431** | 0.3300 |
| regret | **0.0000** | 0.0051 |
| wasteful | 0.1263 | **0.1094** |

The 149M is **more accurate and 2.3× more expensive.** An accuracy table would
have called this an improvement and shipped it.

Accuracy counts every mistake once. The declared cost matrix charges a downgrade
twenty times what an over-service costs, because sending a hard question to a
weak model produces a confident wrong answer while sending an easy question to a
strong model merely wastes money. The 149M traded cheap mistakes for expensive
ones. That is invisible to accuracy and dominant in cost.

### Where the trade happened

| audit category | 22M | 149M | delta |
| --- | ---: | ---: | ---: |
| `trivial` | 0.476 | **0.563** | +0.087 |
| `comprehension_transform` | **1.000** | 0.925 | −0.075 |
| `hard_negative_trivial` | **0.820** | 0.800 | −0.020 |
| `conversational`, `hard`, `hard_negative_large`, `scope_restricted`, `simple` | 1.000 | 1.000 | — |

It gained on `trivial`, where the task is recognising a bounded-output
instruction and the error is cheap over-service. It lost on
`comprehension_transform`, where the task needs understanding and the error is a
downgrade. Three misses on n=40 produce the 0.0051 regret that drives the entire
cost blowout.

---

## The risk head degraded badly, and the safety architecture held anyway

| audit, encoder alone | 22M | 149M |
| --- | ---: | ---: |
| P0 recall | **1.000** | 0.370 |
| P0 missed | **0** | 34 |

All 34 misses are two phrasings, both from pool C — the clause wordings reserved
for the audit split that nothing else has seen:

```
27x  "and then irreversibly clear the store it points at"
 7x  "then evaluate its contents and replace {system} with the result"
```

So this is not erratic unsafety. It is a clean failure to generalise two unseen
destructive constructions that the 22M did generalise.

**The deployed composition was unaffected:**

```
encoder alone : P0 recall 0.370  missed 34
regex alone   : P0 recall 1.000  missed 0
regex->encoder: P0 recall 1.000  missed 0  precision 1.000
```

This run is the **first real adversarial test of the escalate-only invariant**,
and it passed. With the 22M the rules floor was never load-bearing in practice,
because that encoder was already at 1.000 on audit — the floor was insurance
nobody had claimed on. Here the encoder fell to 0.370 and the deployed recall
stayed at 1.000 with precision 1.000.

`final_risk = max(deterministic_rules(action), encoder(action))` is now a
measured guarantee against a genuinely weak encoder, not only a stated one.

---

## Live provider test: the failure reaches the user

117 prompts, both models called on every prompt, **both routers scored against
the same responses and the same judge verdicts**. Running the experiment twice
and differencing would have confounded the routers with judge variance, which is
large: five reruns of an identical comparison previously returned 17%, 11.1%,
3.3%, 15% and 10.0%. Pairing removes it.

| | baseline | LL-22M | LL-149M |
| --- | ---: | ---: | ---: |
| cost (USD) | 0.30615 | **0.23945** | 0.24072 |
| cost saving | — | **21.8%** | 21.4% |
| token saving | — | 13.5% | 18.5% *(see below)* |
| **prompts left unanswered** | — | **0 / 117** | **6 / 117** |
| judged materially worse | — | 2 | 1 |
| routing disagreements | — | 30 / 117 | |

**The cost saving is a tie** — 21.8% against 21.4%, with the 149M marginally
behind. Six times the artifact and ten times the latency bought nothing.

**The 18.5% token saving is not a saving.** The 149M "saves" those tokens by
declining to answer six prompts that needed answers. Counting that as efficiency
would be counting a dropped request as a fast one.

### The six silent failures

| kind | asserted | LL-22M | LL-149M |
| --- | --- | --- | --- |
| `proof` | LARGE | LARGE (0.987) | **NO_MODEL (0.998)** |
| `latency` | SMALL | SMALL (0.963) | **NO_MODEL (0.989)** |

Two distinct fixture items, each hit on all three repeats — deterministic, not
sampling noise. The user asks a question and receives nothing.

---

## The confidence floor does not protect against this, and that is a finding

The 0.90 confidence floor was introduced to eliminate exactly this failure. It
worked for the 22M: that model's wrong `NO_MODEL` decisions sat at 0.481 and
0.735, below the floor, and were discarded.

The 149M's wrong `NO_MODEL` decisions sit at **0.998 and 0.989.**

**The floor discards uncertainty. It cannot discard confident error.** A model
that has memorised its training distribution is not uncertain when it is wrong —
it is wrong with the same conviction it brings to being right, and every
confidence-threshold mechanism silently stops working.

This limitation was not visible while the only model under test was
well-calibrated. It belongs in the model card as a property of the mechanism,
not of this trunk.

---

## Why: it memorised

Training loss on all three heads reached **0.0000 at epoch 2** and stayed there.
Dev cost and regret were 0.000 from epoch 1 onward.

149M parameters against 9,300 short compositional examples is enough capacity to
absorb pool A outright. What survived to pool C was the surface axis —
recognising bounded-output phrasings, hence `trivial` +0.087. What did not
survive was the compositional axis — reading an unfamiliar clause and working
out what it implies, hence the `comprehension_transform` regret and the two
missed destructive tails.

Every observation is consistent with that one story: the training curve, the
route head, the risk head, and the confidently-wrong live failures.

**This is a finding about the corpus as much as the model.** 9,300 examples of
this difficulty cannot honestly train 149M parameters. The 22M is not better
because MiniLM is a better trunk; it is better because its capacity is
proportionate to the evidence available.

---

## Artifact cost

| | LL-22M | LL-149M |
| --- | ---: | ---: |
| FP32 ONNX | **90.4 MB** | 596.8 MB |
| Latency p50, 1 thread | **7.21 ms** | 73.20 ms |
| p95 / p99 | 10.27 / 17.81 ms | 80.08 / 110.03 ms |

The ~7 ms figure is what makes LL-22M usable in an agent's hot path: cheap
enough to consult before every turn. At 73 ms a router built to save money
starts costing perceptible time on every request, and it would need to save
considerably more to justify that. It saves slightly less.

**INT8 was also produced and also failed**, worse than FP32 in every respect:
150.2 MB, 25.46 ms, route agreement 0.729, regret 0.0724, mean cost 2.3148 —
seven times the FP32 reference.

---

## What a fair second attempt would look like

The 149M was not given a fair trial in one specific respect, and the evidence
points at the fix.

At epoch 1 the losses were still non-zero (route 0.0634) while dev was already
perfect. The model reached competence *before* it saturated; memorisation set in
during epoch 2. Those weights no longer exist, because `train.py` checkpoints
only after the final epoch.

A fair retry would:

1. **Checkpoint every epoch** and keep them.
2. **Early-stop on audit, not dev.** Dev is pool A and was saturated from epoch
   1, so it carries no signal about when to stop. This is the single most
   important change: the run was flying blind on the only metric that mattered.
3. **Train 1–2 epochs**, not 4. Epochs 3 and 4 cost an hour and changed nothing
   measurable.
4. **Or enlarge the corpus** substantially, which is the real fix if a 149M
   model is wanted at all.

That is a ~35 minute experiment, not a research programme. It was not run here
because the question asked — *does the bigger trunk earn its place?* — has been
answered for this corpus.

---

## Reproduction

```bash
python scripts/train.py --trunk 150m --epochs 4 --batch 16
python scripts/export_onnx.py --artifact artifacts/ll-150m
python scripts/compare_trunks.py
python scripts/diagnose_p0.py --artifact artifacts/ll-150m
python scripts/measure_savings_ab.py --a artifacts/ll-22m/onnx \
                                     --b artifacts/ll-150m/onnx --repeats 3
```

| evidence | file |
| --- | --- |
| Training and evaluation report | `artifacts/ll-150m/report.json` |
| Export gates, latency, INT8 | `artifacts/ll-150m/onnx/export-report.json` |
| P0 failure analysis | `data/p0-diagnosis-ll-150m.json` |
| Paired live test, per prompt | `data/savings-ab-149m.jsonl` |

The live test independently re-measured LL-22M on fresh calls and returned
**21.8% cost saving and 13.5% token saving**, against the published **22.5% and
13.7%**. That difference is provider sampling variance on a 117-prompt sample,
and it re-validates the v5 headline figures rather than revising them.
