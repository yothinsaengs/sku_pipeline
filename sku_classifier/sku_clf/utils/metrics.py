from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    unique = np.unique(y_true)
    if unique.size < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def safe_macro_auc(targets: np.ndarray, outputs: np.ndarray) -> float:
    aucs = []
    for class_index in range(targets.shape[1]):
        auc = safe_binary_auc(targets[:, class_index], outputs[:, class_index])
        if not np.isnan(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else np.nan


def normalize_thresholds(threshold: float | np.ndarray | List[float], num_classes: int) -> np.ndarray:
    if isinstance(threshold, (list, tuple, np.ndarray)):
        thresholds = np.asarray(threshold, dtype=np.float32)
        if thresholds.shape[0] != num_classes:
            raise ValueError(f"Expected {num_classes} thresholds, got {thresholds.shape[0]}")
        return thresholds
    return np.full(num_classes, float(threshold), dtype=np.float32)


def probabilities_to_labels(outputs: np.ndarray, threshold: float | np.ndarray | List[float]) -> np.ndarray:
    if outputs.size == 0:
        return np.array([], dtype=np.int64)
    labels = []
    negative_label = outputs.shape[1]
    thresholds = normalize_thresholds(threshold, negative_label)
    for row in outputs:
        active = np.where(row >= thresholds)[0]
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


def calculate_metrics(
    outputs: np.ndarray,
    hard_targets: np.ndarray,
    threshold: float | np.ndarray | List[float] = 0.5,
    macro_min_support: int = 1,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if outputs.size == 0 or hard_targets.size == 0:
        return empty_metrics(), []

    num_classes = hard_targets.shape[1]
    true_labels = targets_to_labels(hard_targets)
    pred_labels = probabilities_to_labels(outputs, threshold)
    labels = list(range(num_classes + 1))
    support_by_label = {label: int((true_labels == label).sum()) for label in labels}
    macro_labels = [label for label in labels if support_by_label[label] >= macro_min_support]
    if not macro_labels:
        macro_labels = labels

    precision_macro_all, recall_macro_all, f1_macro_all, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, average="macro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=macro_labels, average="macro", zero_division=0
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
        "precision_macro_all": precision_macro_all,
        "recall_macro_all": recall_macro_all,
        "f1_macro_all": f1_macro_all,
        "macro_min_support": macro_min_support,
        "macro_included_class_count": len(macro_labels),
        "macro_ignored_class_count": len(labels) - len(macro_labels),
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "false_positive_rate": fp / negatives if negatives else 0.0,
        "false_negative_rate": fn / positives if positives else 0.0,
    }

    metrics["roc_auc_macro"] = safe_macro_auc(hard_targets, outputs)

    per_class = []
    p, r, f1, s = precision_recall_fscore_support(true_labels, pred_labels, labels=labels, average=None, zero_division=0)
    for idx, label in enumerate(labels):
        row = {
            "class_index": label,
            "class_name": "NEGATIVE" if label == num_classes else f"class_{label}",
            "support": int(s[idx]),
            "included_in_macro": int(s[idx]) >= macro_min_support,
            "precision": p[idx],
            "recall": r[idx],
            "f1": f1[idx],
        }
        if label < num_classes:
            row["roc_auc"] = safe_binary_auc(hard_targets[:, label], outputs[:, label])
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
        "precision_macro_all": 0.0,
        "recall_macro_all": 0.0,
        "f1_macro_all": 0.0,
        "macro_min_support": 1,
        "macro_included_class_count": 0,
        "macro_ignored_class_count": 0,
        "precision_weighted": 0.0,
        "recall_weighted": 0.0,
        "f1_weighted": 0.0,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "roc_auc_macro": np.nan,
    }
