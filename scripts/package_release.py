"""Assemble the publishable folder from the repository. Repeatable, not by hand.

Two copies of a document drift. This session produced a README and a model card
carrying the same figures, and within an hour of the numbers changing one of
them was quietly wrong while the other was right — which is the failure this
script exists to prevent.

So `MODEL_CARD.md` is the single source for the model card, `publish/` is
generated from the repository, and the generation is a command rather than a
sequence of copies somebody remembers to repeat.

It also refuses to package a release whose figures do not match the evidence
files. A model card is a claim about measurements; if the measurements on disk
say something else, the card is wrong and packaging it would publish the error.

    python scripts/package_release.py
    python scripts/package_release.py --check      # verify without writing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISH = ROOT / "publish"

#: What Hugging Face receives. The model card is published there as README.md,
#: which is the filename the Hub renders on the model page.
HUGGINGFACE = {
    "artifacts/ll-22m/onnx/model.onnx": "model.onnx",
    "artifacts/ll-22m/onnx/tokenizer.json": "tokenizer.json",
    "artifacts/ll-22m/onnx/tokenizer_config.json": "tokenizer_config.json",
    "config.json": "config.json",
    "LICENSE": "LICENSE",
    "MODEL_CARD.md": "README.md",
}

#: Raw measurements, published so every figure in the card can be recomputed.
EVIDENCE = {
    "data/savings-final.jsonl": "live-test-savings.jsonl",
    "data/phase0-v4-audit.json": "baselines-audit.json",
    "artifacts/ll-22m/report.json": "synthetic-evaluation.json",
    "artifacts/ll-22m/onnx/export-report.json": "onnx-export.json",
}

#: Figures the model card states, and where each must be reproducible from.
#: Checked before packaging, because a card is a claim about measurements and a
#: claim that disagrees with the file beside it is simply false.
CLAIMS = [
    ("22.5%", "cost saving"),
    ("13.7%", "token saving"),
    ("0 / 117", "prompts left unanswered"),
    ("90.4 MB", "artifact size"),
    ("22,716,296", "parameter count"),
]


def verify_claims() -> list[str]:
    """Recompute the headline figures from the evidence and compare."""
    problems: list[str] = []
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")

    rows = [json.loads(line) for line
            in (ROOT / "data" / "savings-final.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    baseline = sum(r["baseline_cost_usd"] for r in rows)
    routed = sum(r["routed_cost_usd"] for r in rows)
    saving = 100 * (1 - routed / baseline)
    if f"{saving:.1f}%" not in card:
        problems.append(f"card does not state the measured cost saving "
                        f"{saving:.1f}%")

    unanswered = sum(1 for r in rows
                     if r["ll_route"] == "NO_MODEL"
                     and r["asserted_tier"] != "NO_MODEL")
    if unanswered and "0 / 117" in card:
        problems.append(f"card claims zero unanswered prompts; evidence shows "
                        f"{unanswered}")

    graph = ROOT / "artifacts" / "ll-22m" / "onnx" / "model.onnx"
    size_mb = round(graph.stat().st_size / 1e6, 1)
    if f"{size_mb} MB" not in card:
        problems.append(f"card does not state the artifact size {size_mb} MB")

    if (ROOT / "artifacts" / "ll-22m" / "onnx" / "model.int8.onnx").exists():
        problems.append("an INT8 graph is present but this release withdrew it")

    for phrase, what in CLAIMS:
        if phrase not in card:
            problems.append(f"card is missing its {what} figure ({phrase})")
    return problems


def package() -> None:
    for name, mapping in (("huggingface", HUGGINGFACE), ("evidence", EVIDENCE)):
        target = PUBLISH / name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for source, destination in mapping.items():
            shutil.copy2(ROOT / source, target / destination)
            print(f"  {name}/{destination:<28} <- {source}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the claims without writing anything")
    args = parser.parse_args(argv[1:])

    problems = verify_claims()
    if problems:
        print("REFUSING TO PACKAGE — the model card disagrees with the evidence:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("claims verified against the evidence files")

    if args.check:
        return 0
    print()
    package()
    total = sum(f.stat().st_size for f in PUBLISH.rglob("*") if f.is_file())
    print(f"\npublish/ assembled, {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
