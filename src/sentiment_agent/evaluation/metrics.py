from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

LABELS = ("positive", "neutral", "negative")


def classification_metrics(gold: Sequence[str], predicted: Sequence[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1.mean()),
        "micro_f1": float(f1_score(
            gold, predicted, labels=LABELS, average="micro", zero_division=0,
        )),
        "weighted_f1": float(f1_score(
            gold, predicted, labels=LABELS, average="weighted", zero_division=0,
        )),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(LABELS)
        },
    }
