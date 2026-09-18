"""Train Little Leo, and report the two cells that decide whether it shipped.

    python scripts/train.py --trunk 22m --epochs 4
    python scripts/train.py --trunk 150m --epochs 3 --batch 16

Nothing about this script is clever. The only things worth noticing are that the
route head is trained solely on PRE rows and the risk and gate heads solely on
ACT rows — a routing label on an action row is a placeholder, not evidence — and
that evaluation reports per-category regret rather than an aggregate, because
the categories that matter are a twentieth of the corpus and an aggregate hides
them completely.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from littleleo.metrics import risk_report, route_report  # noqa: E402
from littleleo.model import (MAX_LENGTH, TRUNK_22M, TRUNK_150M,  # noqa: E402
                             LittleLeo, expected_cost_loss, gate_loss,
                             risk_cost_matrix, route_cost_matrix)
from littleleo.schema import (DEFAULT_REGRET_RATIO, SITE_ACT,  # noqa: E402
                              SITE_PRE, Risk, Route)
from littleleo.synth import read_jsonl  # noqa: E402

TRUNKS = {"22m": TRUNK_22M, "150m": TRUNK_150M}


class TurnDataset(Dataset):
    """Tokenised turns, padded to the longest member rather than to the ceiling.

    The ceiling (``MAX_LENGTH``) only truncates; the actual tensor width is
    whatever the data needs. Any turn that the ceiling would clip is counted and
    reported before it happens, because a silently shortened turn loses exactly
    the qualifying clause that makes a casual-sounding question a hard one.
    """

    def __init__(self, rows, tokenizer, max_length=MAX_LENGTH, label=""):
        self.rows = rows
        texts = [r.encoded() for r in rows]
        # Measure before truncating. A ceiling that silently clips the end off
        # a turn would remove exactly the part that distinguishes "is this
        # migration safe" from "is this migration safe to run on prod, given
        # the constraints below" -- and the model would be blamed for it.
        raw = tokenizer(texts, truncation=False)["input_ids"]
        longest = max(len(ids) for ids in raw)
        clipped = sum(1 for ids in raw if len(ids) > max_length)
        if clipped:
            print(f"  !! {label or 'dataset'}: {clipped}/{len(raw)} turns exceed "
                  f"max_length={max_length} (longest {longest}); raise the "
                  f"ceiling or these are being cut")
        encoded = tokenizer(texts, truncation=True, max_length=max_length,
                            padding="longest", return_tensors="pt")
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.padded_to = int(self.input_ids.shape[1])
        self.route = torch.tensor([int(r.route) for r in rows])
        self.risk = torch.tensor([int(r.risk) for r in rows])
        self.gate = torch.tensor([int(r.gate) for r in rows])
        self.is_pre = torch.tensor([r.site == SITE_PRE for r in rows])
        self.is_act = torch.tensor([r.site == SITE_ACT for r in rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index],
                "attention_mask": self.attention_mask[index],
                "route": self.route[index], "risk": self.risk[index],
                "gate": self.gate[index], "is_pre": self.is_pre[index],
                "is_act": self.is_act[index]}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    route_pred, risk_pred, route_true, risk_true = [], [], [], []
    pre_index, act_index, cursor = [], [], 0
    for batch in loader:
        outputs = model(batch["input_ids"].to(device),
                        batch["attention_mask"].to(device))
        r = outputs["route_logits"].argmax(-1).cpu()
        k = outputs["risk_logits"].argmax(-1).cpu()
        for i in range(len(r)):
            if batch["is_pre"][i]:
                route_pred.append(Route(int(r[i])))
                route_true.append(Route(int(batch["route"][i])))
                pre_index.append(cursor + i)
            if batch["is_act"][i]:
                risk_pred.append(Risk(int(k[i])))
                risk_true.append(Risk(int(batch["risk"][i])))
                act_index.append(cursor + i)
        cursor += len(r)
    return route_true, route_pred, risk_true, risk_pred, pre_index


def collapse_check(predictions, head: str, classes: int = 3,
                   threshold: float = 0.95) -> str:
    """Flag a head that has given up and is predicting one class for everything.

    This is the characteristic failure of a multi-task model whose losses are
    not balanced: one head keeps learning while another discovers that always
    answering with the majority class is a decent local minimum. It shows up as
    a respectable aggregate loss and a head that has stopped being a classifier.

    Caught per epoch rather than at the end, because the fix — moving a task
    weight — has to happen while there is still training left to do.
    """
    if not predictions:
        return ""
    counts = [0] * classes
    for p in predictions:
        counts[int(p)] += 1
    top = max(counts)
    share = top / len(predictions)
    if share < threshold:
        return ""
    return (f"COLLAPSE? {head} predicted class {counts.index(top)} for "
            f"{share:.1%} of rows")


def per_category(rows, pre_index, route_true, route_pred) -> dict:
    """Accuracy and regret by generator category. This is the real report."""
    buckets: dict[str, dict] = {}
    for position, row_index in enumerate(pre_index):
        source = rows[row_index].source
        bucket = buckets.setdefault(source, {"n": 0, "ok": 0, "regret": 0})
        bucket["n"] += 1
        bucket["ok"] += int(route_pred[position] == route_true[position])
        bucket["regret"] += int(route_pred[position] < route_true[position])
    return {name: {"n": b["n"],
                   "accuracy": round(b["ok"] / b["n"], 3),
                   "regret": round(b["regret"] / b["n"], 3)}
            for name, b in sorted(buckets.items())}


def evaluate(model, rows, tokenizer, device, batch: int, label: str) -> dict:
    loader = DataLoader(TurnDataset(rows, tokenizer, label=label), batch_size=batch)
    route_true, route_pred, risk_true, risk_pred, pre_index = predict(
        model, loader, device)
    return {
        "split": label,
        "route": route_report(route_true, route_pred),
        "risk": risk_report(risk_true, risk_pred),
        "by_category": per_category(rows, pre_index, route_true, route_pred),
        "padded_to": loader.dataset.padded_to,
        # Underscored: consumed by the collapse check, not part of the report.
        "_route_pred": route_pred,
        "_risk_pred": risk_pred,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trunk", choices=sorted(TRUNKS), default="22m")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--regret-ratio", type=float, default=DEFAULT_REGRET_RATIO)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args(argv[1:])

    trunk = TRUNKS[args.trunk]
    device = torch.device("cpu")
    torch.manual_seed(20260918)
    torch.set_num_threads(12)

    rows = read_jsonl(ROOT / "data" / "corpus.jsonl")
    train_rows = [r for r in rows if r.split == "train"]
    dev_rows = [r for r in rows if r.split == "dev"]
    test_rows = [r for r in rows if r.split == "test"]
    audit_rows = [r for r in rows if r.split == "audit"]

    print(f"trunk: {trunk}")
    tokenizer = AutoTokenizer.from_pretrained(trunk)
    model = LittleLeo(trunk).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {total_params/1e6:.1f}M")

    loader = DataLoader(TurnDataset(train_rows, tokenizer),
                        batch_size=args.batch, shuffle=True)
    head_names = ("route_head", "risk_head", "gate_head", "abstain_head")
    head_params = [p for n, p in model.named_parameters()
                   if any(h in n for h in head_names)]
    trunk_params = [p for n, p in model.named_parameters()
                    if not any(h in n for h in head_names)]
    optimiser = torch.optim.AdamW(
        [{"params": trunk_params, "lr": args.lr},
         {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
    steps = len(loader) * args.epochs
    schedule = get_linear_schedule_with_warmup(optimiser, int(steps * 0.1), steps)

    route_costs = route_cost_matrix(args.regret_ratio)
    risk_costs = risk_cost_matrix()

    print(f"training: {len(train_rows)} rows, {len(loader)} steps/epoch, "
          f"{args.epochs} epochs")
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Tracked separately, because one aggregate number cannot tell a model
        # that is learning both tasks from one that has traded a dead risk head
        # for a slightly better route head.
        totals = {"route": 0.0, "risk": 0.0, "gate": 0.0}
        for step, batch in enumerate(loader, 1):
            optimiser.zero_grad()
            outputs = model(batch["input_ids"].to(device),
                            batch["attention_mask"].to(device))
            parts = {
                "route": expected_cost_loss(outputs["route_logits"],
                                            batch["route"], route_costs,
                                            batch["is_pre"]),
                "risk": expected_cost_loss(outputs["risk_logits"],
                                           batch["risk"], risk_costs,
                                           batch["is_act"]),
                "gate": gate_loss(outputs["gate_logit"], batch["gate"],
                                  batch["is_act"]),
            }
            loss = parts["route"] + parts["risk"] + parts["gate"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            for name, value in parts.items():
                totals[name] += float(value.detach())
        dev = evaluate(model, dev_rows, tokenizer, device, args.batch, "dev")
        n = len(loader)
        print(f"  epoch {epoch}: loss route {totals['route']/n:.4f} "
              f"risk {totals['risk']/n:.4f} gate {totals['gate']/n:.4f}  |  "
              f"dev cost {dev['route']['mean_cost']:.3f} "
              f"regret {dev['route']['regret']:.3f} "
              f"P0rec {dev['risk']['p0_recall']:.3f}  "
              f"[{time.monotonic()-started:.0f}s]")
        for warning in (collapse_check(dev["_route_pred"], "route_head"),
                        collapse_check(dev["_risk_pred"], "risk_head")):
            if warning:
                print(f"    !! {warning}")

    reports = {name: evaluate(model, subset, tokenizer, device, args.batch, name)
               for name, subset in (("test", test_rows), ("audit", audit_rows))}
    # The raw prediction lists are working state for the collapse check, not
    # findings. Thousands of enum values in the report file would bury the
    # numbers a reader actually came for.
    serialisable = {name: {k: v for k, v in report.items()
                           if not k.startswith("_")}
                    for name, report in reports.items()}

    out = Path(args.out) / f"ll-{args.trunk}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "trunk": trunk,
                "params": total_params}, out / "model.pt")
    tokenizer.save_pretrained(out)
    (out / "report.json").write_text(
        json.dumps({"trunk": trunk, "parameters": total_params,
                    "regret_ratio": args.regret_ratio,
                    "train_seconds": round(time.monotonic() - started, 1),
                    "reports": serialisable}, indent=2), encoding="utf-8")

    print()
    for name, report in reports.items():
        r, k = report["route"], report["risk"]
        print(f"== {name} ==")
        print(f"  route: acc {r['accuracy']:.3f}  downgraded {r['downgraded']:.3f}  "
              f"regret {r['regret']:.3f}  spend {r['large_call_spend']:.3f}  "
              f"mean_cost {r['mean_cost']:.3f}")
        print(f"  risk : P0 recall {k['p0_recall']:.3f}  prec {k['p0_precision']:.3f}  "
              f"missed {k['p0_missed']}  under {k['under_called']}")
        for category, values in report["by_category"].items():
            marker = "  <<<" if "hard_negative" in category else ""
            print(f"    {category:<26} n={values['n']:<5} "
                  f"acc={values['accuracy']:.3f} regret={values['regret']:.3f}{marker}")
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
