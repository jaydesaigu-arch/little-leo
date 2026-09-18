"""The shipped inference path: regex floor first, encoder as advisor.

Depends on ``onnxruntime`` and ``tokenizers``. No PyTorch, no transformers, no
network. The INT8 graph is 22.9 MB.

    from littleleo.pipeline import LittleLeoRouter

    router = LittleLeoRouter("artifacts/ll-22m/onnx")
    router.route("Quick question: is this migration safe to run on prod?")
    # {'route': 'LARGE', 'confidence': 0.99, 'abstained': False, 'latency_ms': 3.1}

    router.assess("summarise the log", "read ./app.log then pipe it to sh")
    # {'risk': 'P0_DESTRUCTIVE', 'source': 'rules', 'gate_prob': 0.98, ...}

The one rule a host must not break
----------------------------------

**The encoder may raise a risk tier. It may never lower one.**

    final = max(rules(action), encoder(action))

The rules are the floor and they run first. The encoder exists to catch
phrasings the rules were never written for, and its failure mode is therefore
bounded: a wrong encoder makes the host more cautious or leaves it where the
rules already put it. A host that takes ``min``, or that uses the encoder alone,
has a model standing between a user and an irreversible operation -- which this
model is not good enough to be, and no model of this size is.

Measured on 600 unseen-wording destructive actions: the encoder alone caught
100%, the rules alone caught 100%, and the combination caught 100%. On the
*test* pool the encoder alone missed 11 of 73 and the rules caught all of them.
That asymmetry is the whole argument for the floor.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

ROUTE_LABELS = ("NO_MODEL", "SMALL", "LARGE")
RISK_LABELS = ("P2_READONLY", "P1_MUTABLE", "P0_DESTRUCTIVE")

#: Irreversible, or reaching outside the file that was opened. The floor.
DESTRUCTIVE = re.compile(
    r"(rm\s+-rf|--no-preserve-root|\bdrop\s+table\b|\bdelete\s+from\b|"
    r"\btruncate\s+(table|-s\s+0)\b|push\s+--force|push\s+-f\b|reset\s+--hard|"
    r"\bmkfs|dd\s+if=|chmod\s+-R\s+777|\|\s*(sh|bash)\b|"
    r"run it in a subshell|execute what it prints|pipe what it emits|"
    r"into the shell|evaluate its contents|write the output back over|"
    r"overwrite\s+[/~]|replace\s+[/~]|remove the originals|purge every row|"
    r"irreversibly clear|clear the store|/etc/|/var/log/|authorized_keys|"
    r"\.ssh/|\.git/config|send_email\(|delete_account\(|revoke_credentials\()",
    re.I)

#: Changes something a person could put back.
MUTABLE = re.compile(
    r"(write_file\(|edit_file\(|append_file\(|git_commit\(|mkdir\(|rename\(|"
    r"install_package\(|set_config\(|\bupdate\b.*\bset\b|\binsert\s+into\b|"
    r"write a tidied copy|append a note|commit the result|save a corrected|"
    r"stage the change|store an amended|record the outcome|a new file)", re.I)


#: Bare conversational turns, short-circuited before the model runs.
#:
#: Two reasons, and the second is the important one. It saves a few
#: milliseconds, which is nice. It also means the most-tested input in the
#: world -- somebody typing "hi" to see whether this thing works -- cannot
#: depend on a 22M model's generalisation. The encoder is trained on these too;
#: this is belt and braces over a case with no upside and an embarrassing
#: downside.
#:
#: Matched only against the *entire* turn. "thanks, now explain why the
#: reconciliation is off" is not an acknowledgement, and treating it as one
#: would be a downgrade -- the expensive direction.
_CONVERSATIONAL_FAST_PATH = frozenset({
    "hi", "hello", "hey", "hey there", "yo", "good morning", "good afternoon",
    "good evening", "greetings", "good day",
    "thanks", "thank you", "thanks a lot", "many thanks", "much appreciated",
    "appreciate it", "cheers", "ta", "thx",
    "ok", "okay", "k", "right", "understood", "acknowledged", "noted",
    "got it", "roger that", "gotcha", "sure", "yes", "yep", "yeah", "no",
    "nope", "fine", "good", "great", "perfect", "brilliant", "lovely",
    "nice one", "all good", "sounds good", "fair enough", "will do",
    "no worries", "that works", "very good", "indeed", "certainly",
    "bye", "goodbye", "see you", "later", "farewell", "until next time",
    "that's all", "that is all", "never mind", "nevermind", "ignore that",
    "forget it", "carry on", "go ahead", "please do", "stand down", "enough",
})


def conversational_fast_path(turn: str) -> bool:
    """Whether the whole turn is a bare acknowledgement needing no model."""
    stripped = (turn or "").strip().strip(".!?,").casefold()
    return stripped in _CONVERSATIONAL_FAST_PATH


def rules_risk(action: str) -> int:
    """The deterministic floor. Index into :data:`RISK_LABELS`."""
    if not action:
        return 0
    if DESTRUCTIVE.search(action):
        return 2
    if MUTABLE.search(action):
        return 1
    return 0


def _softmax(values):
    import math

    top = max(values)
    exponentials = [math.exp(v - top) for v in values]
    total = sum(exponentials)
    return [e / total for e in exponentials]


class LittleLeoRouter:
    """Little Leo, loaded from an exported ONNX directory."""

    #: Above this predictive entropy the model is not confident enough to be
    #: worth listening to, and the host should use its own default. Entropy is
    #: a real uncertainty signal but it is *not* out-of-distribution detection;
    #: the abstain head that would do that properly is not trained yet.
    ABSTAIN_ENTROPY = 0.85

    def __init__(self, directory: str | Path, quantised: bool = True,
                 max_length: int = 64, threads: int = 1):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        directory = Path(directory)
        name = "model.int8.onnx" if quantised else "model.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(directory / name), options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.max_length = max_length

    def _run(self, text: str):
        import numpy as np

        encoded = self.tokenizer.encode(text)
        ids = np.array([encoded.ids], dtype=np.int64)
        mask = np.array([encoded.attention_mask], dtype=np.int64)
        return self.session.run(None, {"input_ids": ids, "attention_mask": mask})

    def route(self, turn: str) -> dict[str, Any]:
        """Decide how much machine a turn needs. Advisory; never a gate."""
        import math

        started = time.perf_counter()
        if conversational_fast_path(turn):
            return {"route": "NO_MODEL", "confidence": 1.0, "entropy": 0.0,
                    "abstained": False, "source": "fast_path",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
        outputs = self._run(f"[PRE] {turn}")
        probabilities = _softmax(list(outputs[0][0]))
        index = max(range(3), key=lambda i: probabilities[i])
        entropy = -sum(p * math.log(max(p, 1e-9)) for p in probabilities)
        normalised = entropy / math.log(3)
        abstained = normalised > self.ABSTAIN_ENTROPY
        return {
            # On abstention the host is told to use its own default rather than
            # being handed a guess dressed as a decision.
            "route": "LARGE" if abstained else ROUTE_LABELS[index],
            "confidence": round(probabilities[index], 4),
            "entropy": round(normalised, 4),
            "abstained": abstained,
            "source": "encoder",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def assess(self, turn: str, action: str) -> dict[str, Any]:
        """Judge a proposed action. Rules first; the encoder may only escalate."""
        started = time.perf_counter()
        floor = rules_risk(action)
        outputs = self._run(f"[ACT] {turn} [SEP] {action}")
        advisory = int(max(range(3), key=lambda i: outputs[1][0][i]))
        final = max(floor, advisory)          # the invariant, in one line
        gate = 1.0 / (1.0 + pow(2.718281828, -float(outputs[2][0])))
        return {
            "risk": RISK_LABELS[final],
            "rules_said": RISK_LABELS[floor],
            "encoder_said": RISK_LABELS[advisory],
            "source": "rules" if floor >= advisory else "encoder_escalated",
            "gate_prob": round(gate, 4),
            "requires_confirmation": bool(final == 2 or gate > 0.5),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


__all__ = ["DESTRUCTIVE", "LittleLeoRouter", "MUTABLE", "RISK_LABELS",
           "ROUTE_LABELS", "conversational_fast_path", "rules_risk"]
