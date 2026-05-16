import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
from typing import Dict, Any, List

def calculate_metrics(outputs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    # outputs: [N, C] probabilities, targets: [N, C] binary
    preds = (outputs > threshold).astype(np.int32)
    
    # Aggregated metrics (Macro)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    acc = accuracy_score(targets, preds)
    
    metrics = {
        'acc': acc,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1
    }
    
    # Per-class metrics
    num_classes = targets.shape[1]
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(targets, preds, average=None, zero_division=0)
    
    for i in range(num_classes):
        metrics[f'precision_class_{i}'] = p_class[i]
        metrics[f'recall_class_{i}'] = r_class[i]
        metrics[f'f1_class_{i}'] = f1_class[i]
        
        # ROC AUC
        try:
            metrics[f'auc_class_{i}'] = roc_auc_score(targets[:, i], outputs[:, i])
        except:
            metrics[f'auc_class_{i}'] = 0.5
            
    return metrics

def get_confusion_matrix(outputs: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
    # For multi-label, a standard confusion matrix doesn't perfectly apply if classes overlap.
    # But since each ROI in this POC is assumed to have one positive class OR be negative:
    # We can create an (N+1)x(N+1) matrix where index N is "Negative".
    
    num_pos = targets.shape[1]
    preds = (outputs > threshold).astype(np.int32)
    
    # Convert to single label [0, num_pos]
    # 0..num_pos-1: positive classes
    # num_pos: negative (all zeros)
    
    def to_single_label(arr):
        labels = []
        for row in arr:
            if row.sum() == 0:
                labels.append(num_pos)
            else:
                labels.append(np.argmax(row))
        return np.array(labels)
    
    true_labels = to_single_label(targets)
    pred_labels = to_single_label(preds)
    
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_pos + 1)))
    return cm
