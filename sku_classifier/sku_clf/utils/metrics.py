from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def probabilities_to_labels(outputs: np.ndarray, threshold: float) -> np.ndarray:
    if outputs.size == 0:
        return np.array([], dtype=np.int64)
    labels = []
    negative_label = outputs.shape[1]
    for row in outputs:
        active = np.where(row >= threshold)[0]
        if active.size == 0:
            labels.append(negative_label)
        else:
            labels.append(int(active[np.argmax(row[active])]))
    return np.array(labels, dtype=np.int64)


def targets_to_labels(targets: np.ndarray) -> np.ndarray:
    if targets.size == 0:
        return np.array([], dtype=np.int64)
    negative_label = targets.shape[1]
    labels = []
    for row in targets:
        if row.sum() <= 0:
            labels.append(negative_label)
        else:
            labels.append(int(np.argmax(row)))
    return np.array(labels, dtype=np.int64)


def calculate_metrics(outputs: np.ndarray, hard_targets: np.ndarray, threshold: float = 0.5) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if outputs.size == 0 or hard_targets.size == 0:
        return empty_metrics(), []

    num_classes = hard_targets.shape[1]
    true_labels = targets_to_labels(hard_targets)
    pred_labels = probabilities_to_labels(outputs, threshold)
    labels = list(range(num_classes + 1))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, average="weighted", zero_division=0
    )
    support = len(true_labels)
    fp = int(((pred_labels != num_classes) & (true_labels == num_classes)).sum())
    fn = int(((pred_labels == num_classes) & (true_labels != num_classes)).sum())
    negatives = int((true_labels == num_classes).sum())
    positives = int((true_labels != num_classes).sum())

    metrics = {
        "support": support,
        "positive_support": positives,
        "negative_support": negatives,
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "false_positive_rate": fp / negatives if negatives else 0.0,
        "false_negative_rate": fn / positives if positives else 0.0,
    }

    try:
        metrics["roc_auc_macro"] = roc_auc_score(hard_targets, outputs, average="macro")
    except ValueError:
        metrics["roc_auc_macro"] = np.nan

    per_class = []
    p, r, f1, s = precision_recall_fscore_support(true_labels, pred_labels, labels=labels, average=None, zero_division=0)
    for idx, label in enumerate(labels):
        row = {
            "class_index": label,
            "class_name": "NEGATIVE" if label == num_classes else f"class_{label}",
            "support": int(s[idx]),
            "precision": p[idx],
            "recall": r[idx],
            "f1": f1[idx],
        }
        if label < num_classes:
            try:
                row["roc_auc"] = roc_auc_score(hard_targets[:, label], outputs[:, label])
            except ValueError:
                row["roc_auc"] = np.nan
        else:
            row["roc_auc"] = np.nan
        per_class.append(row)

    return metrics, per_class


def empty_metrics() -> Dict[str, Any]:
    return {
        "support": 0,
        "positive_support": 0,
        "negative_support": 0,
        "accuracy": 0.0,
        "precision_macro": 0.0,
        "recall_macro": 0.0,
        "f1_macro": 0.0,
        "precision_weighted": 0.0,
        "recall_weighted": 0.0,
        "f1_weighted": 0.0,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "roc_auc_macro": np.nan,
    }
