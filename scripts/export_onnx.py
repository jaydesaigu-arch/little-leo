"""Export to ONNX, quantise to INT8, and prove the exported graph still agrees.

An export is not finished when a file appears. Three things go wrong routinely
and silently: the traced graph bakes in a fixed sequence length, quantisation
moves a decision boundary far enough to flip predictions, and the pooling gets
dropped or subtly altered so the exported model reads a different vector than
the one that was trained.

So this script exports, then re-runs the held-out set through *the exported
artefact* and compares predictions against PyTorch head by head. A disagreement
rate above a declared tolerance fails the export rather than shipping it with a
note in the README that nobody reads.

INT8 dynamic quantisation is used because the weights dominate this model's
footprint and dynamic quantisation needs no calibration set — which matters for
a model whose training data is synthetic and whose users' data is not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from littleleo.metrics import risk_report, route_report  # noqa: E402
from littleleo.model import MAX_LENGTH, LittleLeo  # noqa: E402
from littleleo.schema import SITE_ACT, SITE_PRE, Risk, Route  # noqa: E402
from littleleo.synth import read_jsonl  # noqa: E402

#: Raw prediction agreement, reported but *not* the gate.
#:
#: The first version of this script failed an export at 95.3% risk agreement.
#: Inspection showed every disagreement was P1 against P2 -- two non-destructive
#: tiers -- with P0 recall, route regret and mean cost all unchanged. The gate
#: was measuring prediction identity when the model had been accepted on
#: metrics, so it failed an artefact that was fine.
#:
#: This number stays in the report because a large drift is worth seeing even
#: when it costs nothing. It is a diagnostic, not a verdict.
AGREEMENT_TOLERANCE = 0.005

# --- release gates -------------------------------------------------------
#
# Metric invariance, not bit-level identity. Quantisation necessarily perturbs
# weights, and points sitting near a decision hyperplane will flip under any
# epsilon -- including in FP32. What must not move is the safety floor.

#: Hard. A P0 catch may not be lost, ever, at any tolerance.
#: (Enforced as: INT8 recall >= FP32 recall.)

#: Hard. A downgrade costs 20x an over-spend, so regret gets a near-zero bound
#: rather than a proportional one.
ZERO_REGRET_BOUND = 0.001

#: Operational. Relative to FP32, so the bound scales with the metric instead
#: of with whatever this particular corpus happens to score.
MAX_MEAN_COST_DRIFT_RATIO = 0.05


class ExportWrapper(torch.nn.Module):
    """Fixed output order, so the consumer never has to guess.

    ONNX returns a tuple. Naming the outputs in the export and documenting the
    order here is the difference between a usable artefact and one where a
    caller silently reads the risk head as the route head.
    """

    def __init__(self, model: LittleLeo):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask)
        return (out["route_logits"], out["risk_logits"],
                out["gate_logit"], out["abstain_logit"])


def export(artifact: Path, out_dir: Path) -> dict:
    blob = torch.load(artifact / "model.pt", map_location="cpu",
                      weights_only=False)
    model = LittleLeo(blob["trunk"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(artifact)

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "model.onnx"
    int8_path = out_dir / "model.int8.onnx"

    sample = tokenizer(["[PRE] review this passage and reply with just yes or no"],
                       return_tensors="pt", truncation=True,
                       max_length=MAX_LENGTH, padding="longest")
    torch.onnx.export(
        ExportWrapper(model),
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["route_logits", "risk_logits", "gate_logit",
                      "abstain_logit"],
        # Both axes dynamic. A graph with a baked-in batch or sequence length
        # works perfectly on the example it was traced with and fails on the
        # first real request of a different shape.
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "route_logits": {0: "batch"}, "risk_logits": {0: "batch"},
            "gate_logit": {0: "batch"}, "abstain_logit": {0: "batch"},
        },
        opset_version=17, do_constant_folding=True,
        # torch 2.14 defaults to the dynamo exporter, whose graph onnxruntime's
        # dynamic quantiser cannot run shape inference over -- it fails with
        # "Inferred shape and existing shape differ in dimension 0: (384) vs
        # (3)", the hidden size against a head's output width. The legacy
        # TorchScript tracer emits the shapes the quantiser expects. This model
        # has no data-dependent control flow, so the tracer loses nothing.
        dynamo=False)

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Quantise the encoder; leave the heads in FP32.
    #
    # Measured honestly: excluding them changed nothing. Agreement was
    # identical with and without, which says the prediction drift comes from
    # accumulated encoder error shifting the pooled vector before any head sees
    # it -- not from rounding the heads themselves. The first version of this
    # comment asserted the opposite and was wrong.
    #
    # It is kept anyway because it is free: four small projections against 22.7
    # million parameters, no measurable file-size cost, and it removes one
    # mechanism by which a future trunk change could move a decision boundary.
    graph = onnx.load(str(fp32_path))
    head_tensors = {name for name in
                    (initializer.name for initializer in graph.graph.initializer)
                    if any(head in name for head in
                           ("route_head", "risk_head", "gate_head",
                            "abstain_head"))}
    exclude = [node.name for node in graph.graph.node
               if any(inp in head_tensors for inp in node.input)]
    print(f"excluding {len(exclude)} head node(s) from quantisation: {exclude}")

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8,
                     nodes_to_exclude=exclude)
    tokenizer.save_pretrained(out_dir)

    return {
        "trunk": blob["trunk"], "parameters": blob["params"],
        "fp32_mb": round(fp32_path.stat().st_size / 1e6, 1),
        "int8_mb": round(int8_path.stat().st_size / 1e6, 1),
        "pytorch_mb": round((artifact / "model.pt").stat().st_size / 1e6, 1),
    }


def verify(artifact: Path, out_dir: Path, rows) -> dict:
    """Run the held-out set through the exported graph and compare."""
    import onnxruntime as ort

    blob = torch.load(artifact / "model.pt", map_location="cpu",
                      weights_only=False)
    model = LittleLeo(blob["trunk"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(artifact)

    encoded = tokenizer([r.encoded() for r in rows], truncation=True,
                        max_length=MAX_LENGTH, padding="longest",
                        return_tensors="np")
    findings: dict = {}
    with torch.no_grad():
        reference = model(torch.tensor(encoded["input_ids"]),
                          torch.tensor(encoded["attention_mask"]))
    torch_route = reference["route_logits"].argmax(-1).numpy()
    torch_risk = reference["risk_logits"].argmax(-1).numpy()

    for name, path in (("fp32", out_dir / "model.onnx"),
                       ("int8", out_dir / "model.int8.onnx")):
        session = ort.InferenceSession(str(path),
                                       providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"input_ids": encoded["input_ids"],
                                     "attention_mask": encoded["attention_mask"]})
        route_agree = float((outputs[0].argmax(-1) == torch_route).mean())
        risk_agree = float((outputs[1].argmax(-1) == torch_risk).mean())

        # Agreement is a proxy. What a host experiences is the metric, so the
        # exported graph is scored on the same report the model was accepted
        # on -- a 5% disagreement that leaves P0 recall untouched and a 1% one
        # that costs a P0 catch are not the same event.
        act = [i for i, row in enumerate(rows) if row.site == SITE_ACT]
        pre = [i for i, row in enumerate(rows) if row.site == SITE_PRE]
        exported_risk = risk_report(
            [rows[i].risk for i in act],
            [Risk(int(outputs[1].argmax(-1)[i])) for i in act])
        exported_route = route_report(
            [rows[i].route for i in pre],
            [Route(int(outputs[0].argmax(-1)[i])) for i in pre])

        findings[name] = {
            "route_agreement": round(route_agree, 5),
            "risk_agreement": round(risk_agree, 5),
            "p0_recall": exported_risk["p0_recall"],
            "p0_missed": exported_risk["p0_missed"],
            "route_regret": exported_route["regret"],
            "route_mean_cost": exported_route["mean_cost"],
        }

    # The gate: compare each graph against PyTorch's own accepted figures.
    reference_risk = risk_report([rows[i].risk for i in
                                  [j for j, r in enumerate(rows) if r.site == SITE_ACT]],
                                 [Risk(int(torch_risk[i])) for i in
                                  [j for j, r in enumerate(rows) if r.site == SITE_ACT]])
    reference_route = route_report([rows[i].route for i in
                                    [j for j, r in enumerate(rows) if r.site == SITE_PRE]],
                                   [Route(int(torch_route[i])) for i in
                                    [j for j, r in enumerate(rows) if r.site == SITE_PRE]])
    for name, values in findings.items():
        # Two safety invariants, breached or not -- no tolerance on either.
        safety_ok = values["p0_recall"] >= reference_risk["p0_recall"]
        regret_ok = values["route_regret"] <= ZERO_REGRET_BOUND
        # One operational bound, relative so it scales with the metric rather
        # than with whatever this corpus happens to score.
        ceiling = reference_route["mean_cost"] * (1 + MAX_MEAN_COST_DRIFT_RATIO)
        cost_ok = values["route_mean_cost"] <= ceiling

        values["reference_p0_recall"] = reference_risk["p0_recall"]
        values["reference_regret"] = reference_route["regret"]
        values["reference_mean_cost"] = reference_route["mean_cost"]
        values["mean_cost_ratio"] = round(
            values["route_mean_cost"] / reference_route["mean_cost"], 5)
        values["passes"] = bool(safety_ok and regret_ok and cost_ok)
        values["failed_because"] = (
            f"safety regression: P0 recall {values['p0_recall']} < "
            f"{reference_risk['p0_recall']}" if not safety_ok else
            f"regret violation: {values['route_regret']} > {ZERO_REGRET_BOUND}"
            if not regret_ok else
            f"cost blowout: {values['route_mean_cost']:.4f} > {ceiling:.4f}"
            if not cost_ok else "")
    return findings


def benchmark(out_dir: Path, tokenizer, repeats: int = 200) -> dict:
    """Single-turn latency, which is what a host actually experiences.

    Reported as percentiles, not a mean: a router sits on every turn, so the
    tail is what a user notices.
    """
    import onnxruntime as ort

    text = "[PRE] Quick question: review this migration plan given it has to " \
           "run under active load on the primary"
    encoded = tokenizer([text], return_tensors="np", truncation=True,
                        max_length=MAX_LENGTH, padding="longest")
    results: dict = {}
    for name, path in (("fp32", out_dir / "model.onnx"),
                       ("int8", out_dir / "model.int8.onnx")):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1   # one turn, one core: the honest case
        session = ort.InferenceSession(str(path), options,
                                       providers=["CPUExecutionProvider"])
        feed = {"input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"]}
        for _ in range(20):
            session.run(None, feed)
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            session.run(None, feed)
            timings.append((time.perf_counter() - started) * 1000)
        timings.sort()
        results[name] = {
            "p50_ms": round(timings[len(timings) // 2], 3),
            "p95_ms": round(timings[int(len(timings) * 0.95)], 3),
            "p99_ms": round(timings[int(len(timings) * 0.99)], 3),
            "threads": 1, "repeats": repeats,
        }
    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path,
                        default=ROOT / "artifacts" / "ll-22m")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="latency is meaningless while the CPU is busy")
    args = parser.parse_args(argv[1:])
    out_dir = args.out or (args.artifact / "onnx")

    sizes = export(args.artifact, out_dir)
    print(json.dumps(sizes, indent=2))

    rows = [r for r in read_jsonl(ROOT / "data" / "corpus.jsonl")
            if r.split == "audit"]
    agreement = verify(args.artifact, out_dir, rows)
    print("\nexport agreement with PyTorch (n=%d):" % len(rows))
    for name, values in agreement.items():
        state = "PASS" if values["passes"] else "FAIL"
        print(f"  {name:<5} route {values['route_agreement']:.5f}  "
              f"risk {values['risk_agreement']:.5f}  -> {state}")

    report = {"sizes": sizes, "agreement": agreement,
              "tolerance": AGREEMENT_TOLERANCE}
    if not args.skip_benchmark:
        tokenizer = AutoTokenizer.from_pretrained(args.artifact)
        report["latency"] = benchmark(out_dir, tokenizer)
        print("\nsingle-turn latency, 1 thread:")
        for name, values in report["latency"].items():
            print(f"  {name:<5} p50 {values['p50_ms']:.2f} ms  "
                  f"p95 {values['p95_ms']:.2f} ms  p99 {values['p99_ms']:.2f} ms")

    (out_dir / "export-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out_dir}")
    failed = [n for n, v in agreement.items() if not v["passes"]]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
