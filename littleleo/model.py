"""The encoder, its heads, and a loss that is the evaluation metric.

Architecture is unremarkable on purpose: a pretrained bidirectional encoder,
mean-pooled, with small linear heads. The two decisions worth explaining are the
pooling and the loss.

**Mean pooling, not ``[CLS]``.** The 22M trunk is a sentence-transformers model,
trained with mean pooling as its sentence representation; its ``[CLS]`` vector
was never optimised for anything. Reading ``[CLS]`` from it — which is what most
copy-pasted classifier code does — throws away the part of the pretraining that
makes a 22M model competitive at all.

**The loss is the cost matrix.** Cross-entropy treats every mistake as equally
bad. Here they are emphatically not: routing a hard turn to a small model
produces a confident wrong answer that a person may act on, while routing an
easy turn to a large model wastes a fraction of a cent. So instead of picking
class weights by hand, the loss computes the *expected cost* of the predicted
distribution under the cost matrix declared in :mod:`littleleo.schema`:

    L = E_{p ~ softmax(logits)} [ cost(true, p) ]

The number the model minimises during training and the number it is scored on
afterwards are then the same function. There is no separate set of loss weights
to tune, and no opportunity for the two to drift apart — which is the usual way
a model ends up optimising something subtly different from what is reported.

A small cross-entropy term rides alongside it purely for gradient conditioning:
expected cost alone is flat wherever the model is confidently wrong, and CE
supplies a gradient there.

**Head 4 is not trained in Phase 1.** The abstain head exists so the exported
graph has its final shape, but the corpus contains no out-of-distribution
labels, so training it would mean inventing a target. Until a real OOD set
exists, abstention is computed at inference from the route head's entropy, and
that is stated wherever the head is used rather than quietly papered over.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from .schema import Risk, Route, risk_cost, route_cost

#: The 22M trunk. Apache-2.0, six layers, 384 hidden, mean-pooled.
TRUNK_22M = "sentence-transformers/all-MiniLM-L6-v2"

#: The 150M trunk, swapped in for LL-150M with no other change.
TRUNK_150M = "answerdotai/ModernBERT-base"

#: Truncation ceiling, set from the corpus rather than from habit.
#:
#: The longest encoded turn in the corpus is 59 tokens and the median is 23, so
#: 256 — the number that gets copied into every classifier script — spends
#: roughly four times the attention arithmetic on padding that the mask then
#: discards. Barely noticeable on a 22M trunk; the difference between a
#: forty-minute and a multi-hour run on a 150M one.
#:
#: This is a *ceiling*, not a target: batches pad to their own longest member.
#: A different trunk means a different tokenizer, so the training script checks
#: whether this ceiling actually truncates anything and says so loudly rather
#: than silently clipping the end off a turn.
MAX_LENGTH = 64


def route_cost_matrix(regret_ratio: float) -> torch.Tensor:
    """``C[t, p]`` — the declared cost of predicting *p* when the truth is *t*."""
    return torch.tensor(
        [[route_cost(Route(t), Route(p), regret_ratio) for p in range(3)]
         for t in range(3)], dtype=torch.float32)


def risk_cost_matrix() -> torch.Tensor:
    return torch.tensor(
        [[risk_cost(Risk(t), Risk(p)) for p in range(3)] for t in range(3)],
        dtype=torch.float32)


class LittleLeo(nn.Module):
    """One encoder, four heads, two call sites."""

    def __init__(self, trunk: str = TRUNK_22M, dropout: float = 0.1):
        super().__init__()
        self.trunk_name = trunk
        self.config = AutoConfig.from_pretrained(trunk)
        self.encoder = AutoModel.from_pretrained(trunk)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.route_head = nn.Linear(hidden, 3)
        self.risk_head = nn.Linear(hidden, 3)
        self.gate_head = nn.Linear(hidden, 1)
        self.abstain_head = nn.Linear(hidden, 1)  # Phase 4; see module docstring

    def pool(self, hidden_states: torch.Tensor,
             attention_mask: torch.Tensor) -> torch.Tensor:
        """Masked mean over real tokens. Padding must not dilute the vector."""
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self.pool(outputs.last_hidden_state, attention_mask))
        return {
            "route_logits": self.route_head(pooled),
            "risk_logits": self.risk_head(pooled),
            "gate_logit": self.gate_head(pooled).squeeze(-1),
            "abstain_logit": self.abstain_head(pooled).squeeze(-1),
        }


def expected_cost_loss(logits: torch.Tensor, targets: torch.Tensor,
                       cost: torch.Tensor, mask: torch.Tensor,
                       ce_weight: float = 0.2) -> torch.Tensor:
    """Expected cost under the predicted distribution, plus a CE conditioner.

    *mask* selects the rows this head is responsible for. A routing label on an
    ACT-site row is a placeholder, not evidence, and training on it would teach
    the route head that every proposed tool call is a SMALL turn.
    """
    if mask.sum() == 0:
        return logits.sum() * 0.0
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    probabilities = F.softmax(selected_logits, dim=-1)
    row_costs = cost.to(selected_logits.device)[selected_targets]
    expected = (probabilities * row_costs).sum(dim=-1).mean()
    normaliser = cost.max().clamp(min=1e-9)
    conditioner = F.cross_entropy(selected_logits, selected_targets)
    return expected / normaliser + ce_weight * conditioner


def gate_loss(logit: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
              positive_weight: float = 8.0) -> torch.Tensor:
    """Binary confirmation signal, weighted against missing a confirmation.

    Asymmetric for the same reason everything else here is: an unnecessary
    confirmation is an irritation, a missing one is an incident. 8.0 is the
    ratio of those two in the declared risk costs, rounded down.
    """
    if mask.sum() == 0:
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logit[mask], targets[mask].float(),
        pos_weight=torch.tensor(positive_weight, device=logit.device))


def abstention_from_entropy(route_logits: torch.Tensor) -> torch.Tensor:
    """Normalised predictive entropy of the route head, in ``[0, 1]``.

    A stand-in for the untrained abstain head. It is a genuine uncertainty
    signal — a model spreading mass across all three tiers is telling you
    something — but it is not out-of-distribution detection, and a host should
    not read it as such until Phase 4 replaces it.
    """
    probabilities = F.softmax(route_logits, dim=-1)
    entropy = -(probabilities * probabilities.clamp(min=1e-9).log()).sum(dim=-1)
    return entropy / torch.log(torch.tensor(float(route_logits.shape[-1])))


__all__ = ["LittleLeo", "MAX_LENGTH", "TRUNK_150M", "TRUNK_22M",
           "abstention_from_entropy", "expected_cost_loss", "gate_loss",
           "risk_cost_matrix", "route_cost_matrix"]
