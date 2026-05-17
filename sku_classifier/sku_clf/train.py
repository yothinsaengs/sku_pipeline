import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from sku_clf.data.dataset import SKUExperimentDataset
from sku_clf.data.sampler import BatchRatioSampler
from sku_clf.logging_utils import append_log
from sku_clf.models.backbone import get_model
from sku_clf.models.loss import get_loss_fn
from sku_clf.utils.metrics import calculate_metrics


def resolve_device(config: Dict[str, Any]) -> torch.device:
    device_name = config.get("training", {}).get("device", "auto")
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def make_loader(dataset: SKUExperimentDataset, config: Dict[str, Any], training: bool) -> DataLoader:
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    num_workers = int(config.get("training", {}).get("num_workers", 0))
    if training:
        ratio = config.get("sampling", {}).get("pos_neg_ratio", "1:1")
        sampler = BatchRatioSampler(dataset.samples, batch_size, ratio)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def train_fraction_run(
    config: Dict[str, Any],
    run_dir: Path,
    positive_ids: Sequence[int],
    positive_names: Sequence[str],
    train_samples: Sequence[Dict[str, Any]],
    val_samples: Sequence[Dict[str, Any]],
    test_samples: Sequence[Dict[str, Any]],
    fraction: float,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    with (run_dir / "resolved_config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    train_ds = SKUExperimentDataset(train_samples, positive_ids, config, is_training=True)
    val_ds = SKUExperimentDataset(val_samples, positive_ids, config, is_training=False)
    test_ds = SKUExperimentDataset(test_samples, positive_ids, config, is_training=False)

    device = resolve_device(config)
    model = get_model(config).to(device)
    pos_weights = compute_class_pos_weights(train_samples, positive_ids, config).to(device)
    criterion = get_loss_fn(config, pos_weights=pos_weights)
    optimizer = build_optimizer(model, config)
    epochs = int(config.get("training", {}).get("epochs", 50))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    threshold = float(config.get("threshold", 0.5))
    metrics_cfg = config.get("metrics", {})
    macro_min_support = int(metrics_cfg.get("macro_min_support", 5))
    threshold_candidates = [float(value) for value in metrics_cfg.get("tune_thresholds", {}).get("candidates", [0.1, 0.3, 0.5, 0.7, 0.9])]
    tune_thresholds_enabled = bool(metrics_cfg.get("tune_thresholds", {}).get("enabled", True))
    scales = [int(size) for size in config.get("input", {}).get("sizes", [56, 112, 224])]
    eval_scales = list(scales)
    if metrics_cfg.get("evaluate_dynamic_scale", True):
        eval_scales.append("dynamic")
    sanity_check_batch(train_ds, config, scales, positive_ids, fraction)
    write_pretrain_summary(
        run_dir=run_dir,
        model=model,
        config=config,
        positive_ids=positive_ids,
        positive_names=positive_names,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        fraction=fraction,
    )
    per_class_interval = int(config.get("logging", {}).get("per_class_interval", 5))
    best_val_f1 = -1.0
    epoch_rows = []
    final_test_rows = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_ds, config, criterion, optimizer, device, epoch, epochs, scales)
        scheduler.step()

        val_scale_rows = []
        test_scale_rows = []
        per_class_rows = []
        for scale in eval_scales:
            val_loss, val_outputs, val_targets = evaluate_outputs(model, val_ds, criterion, device, scale)
            tuned_thresholds = tune_per_class_thresholds(
                val_outputs,
                val_targets,
                threshold_candidates,
                macro_min_support,
                default_threshold=threshold,
            ) if tune_thresholds_enabled else np.full(len(positive_ids), threshold, dtype=np.float32)
            val_metrics, val_per_class = calculate_metrics(val_outputs, val_targets, tuned_thresholds, macro_min_support=macro_min_support)
            test_loss, test_outputs, test_targets = evaluate_outputs(model, test_ds, criterion, device, scale)
            test_metrics, test_per_class = calculate_metrics(test_outputs, test_targets, tuned_thresholds, macro_min_support=macro_min_support)
            val_row = prefixed_row(val_metrics, "val")
            test_row = prefixed_row(test_metrics, "test")
            row = {
                "epoch": epoch,
                "negative_fraction": fraction,
                "scale": scale,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                **threshold_row(tuned_thresholds, "threshold"),
                **val_row,
                **test_row,
            }
            epoch_rows.append(row)
            val_scale_rows.append((scale, val_metrics))
            test_scale_rows.append((scale, test_metrics))

            if epoch % per_class_interval == 0:
                per_class_rows.extend(format_per_class(epoch, fraction, scale, "val", val_per_class, positive_names))
                per_class_rows.extend(format_per_class(epoch, fraction, scale, "test", test_per_class, positive_names))
                per_class_rows.extend(format_thresholds(epoch, fraction, scale, tuned_thresholds, positive_names))

        pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
        if per_class_rows:
            pd.DataFrame(per_class_rows).to_csv(run_dir / f"per_class_metrics_epoch_{epoch:03d}.csv", index=False)

        mean_val_f1 = float(np.mean([metrics["f1_macro"] for _, metrics in val_scale_rows])) if val_scale_rows else 0.0
        mean_test_f1 = float(np.mean([metrics["f1_macro"] for _, metrics in test_scale_rows])) if test_scale_rows else 0.0
        append_log(
            run_dir.parent,
            f"fraction={fraction} epoch={epoch} train_loss={train_loss:.6f} "
            f"mean_val_f1_macro={mean_val_f1:.6f} mean_test_f1_macro={mean_test_f1:.6f}",
        )
        if mean_val_f1 > best_val_f1:
            best_val_f1 = mean_val_f1
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, run_dir / "checkpoints" / "best_model.pth")
        torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, run_dir / "checkpoints" / "last_model.pth")

    best_path = run_dir / "checkpoints" / "best_model.pth"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    for scale in eval_scales:
        val_loss, val_outputs, val_targets = evaluate_outputs(model, val_ds, criterion, device, scale)
        tuned_thresholds = tune_per_class_thresholds(
            val_outputs,
            val_targets,
            threshold_candidates,
            macro_min_support,
            default_threshold=threshold,
        ) if tune_thresholds_enabled else np.full(len(positive_ids), threshold, dtype=np.float32)
        loss, test_outputs, test_targets = evaluate_outputs(model, test_ds, criterion, device, scale)
        metrics, _ = calculate_metrics(test_outputs, test_targets, tuned_thresholds, macro_min_support=macro_min_support)
        final_test_rows.append({"negative_fraction": fraction, "scale": scale, "test_loss": loss, **threshold_row(tuned_thresholds, "threshold"), **metrics})
    pd.DataFrame(final_test_rows).to_csv(run_dir / "final_test_metrics_by_scale.csv", index=False)
    with (run_dir / "run_summary.json").open("w") as f:
        json.dump({"negative_fraction": fraction, "best_val_f1_macro": best_val_f1}, f, indent=2)
    return {"negative_fraction": fraction, "best_val_f1_macro": best_val_f1, "final_test_rows": final_test_rows}


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]):
    training_cfg = config.get("training", {})
    lr = float(training_cfg.get("lr", 0.0003))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    name = training_cfg.get("optimizer", "adamw")
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def compute_class_pos_weights(samples: Sequence[Dict[str, Any]], positive_ids: Sequence[int], config: Dict[str, Any]) -> torch.Tensor:
    weight_cfg = config.get("loss", {}).get("class_pos_weight", {})
    counts = np.zeros(len(positive_ids), dtype=np.float32)
    id_to_idx = {class_id: index for index, class_id in enumerate(positive_ids)}
    for sample in samples:
        if sample.get("role") == "positive" and sample.get("class_id") in id_to_idx:
            counts[id_to_idx[sample["class_id"]]] += 1
    weights = np.ones(len(positive_ids), dtype=np.float32)
    if weight_cfg.get("enabled", True) and len(counts) > 0:
        nonzero = counts[counts > 0]
        if nonzero.size == 0:
            raise ValueError("Cannot compute class positive weights: no positive train samples")
        max_count = float(nonzero.max())
        for index, count in enumerate(counts):
            if count <= 0:
                weights[index] = float(weight_cfg.get("max_weight", 10.0))
            elif weight_cfg.get("strategy", "inv_sqrt") == "inverse":
                weights[index] = max_count / float(count)
            else:
                weights[index] = np.sqrt(max_count / float(count))
        weights = np.minimum(weights, float(weight_cfg.get("max_weight", 10.0)))
    return torch.tensor(weights, dtype=torch.float32)


def count_by_role(samples: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "total": len(samples),
        "positive": sum(1 for sample in samples if sample.get("role") == "positive"),
        "negative": sum(1 for sample in samples if sample.get("role") == "negative"),
        "negative_box": sum(1 for sample in samples if sample.get("source") == "negative_box"),
        "background": sum(1 for sample in samples if sample.get("source") == "background"),
    }


def count_images(samples: Sequence[Dict[str, Any]]) -> int:
    return len({sample.get("stem") for sample in samples})


def count_boxes_by_class(samples: Sequence[Dict[str, Any]], positive_ids: Sequence[int], positive_names: Sequence[str]) -> str:
    id_to_name = {class_id: positive_names[index] for index, class_id in enumerate(positive_ids)}
    counts = {}
    for sample in samples:
        class_id = sample.get("class_id")
        if sample.get("role") == "positive":
            counts[class_id] = counts.get(class_id, 0) + 1
    return "|".join(f"{class_id}:{id_to_name.get(class_id, f'class_{class_id}')}={count}" for class_id, count in sorted(counts.items()))


def model_size_summary(model: torch.nn.Module) -> Dict[str, float]:
    param_count = sum(param.numel() for param in model.parameters())
    trainable_param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    param_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return {
        "parameter_count": param_count,
        "trainable_parameter_count": trainable_param_count,
        "buffer_count": buffer_count,
        "weight_size_mb": round((param_bytes + buffer_bytes) / (1024 * 1024), 3),
    }


def write_pretrain_summary(
    run_dir: Path,
    model: torch.nn.Module,
    config: Dict[str, Any],
    positive_ids: Sequence[int],
    positive_names: Sequence[str],
    train_samples: Sequence[Dict[str, Any]],
    val_samples: Sequence[Dict[str, Any]],
    test_samples: Sequence[Dict[str, Any]],
    fraction: float,
) -> None:
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    ratio = config.get("sampling", {}).get("pos_neg_ratio", "1:1")
    sampler = BatchRatioSampler(list(train_samples), batch_size, ratio)
    model_stats = model_size_summary(model)
    class_pos_weights = compute_class_pos_weights(train_samples, positive_ids, config).cpu().numpy()
    rows = []
    for split_name, samples in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        role_counts = count_by_role(samples)
        rows.append({
            "negative_fraction": fraction,
            "split": split_name,
            "image_count": count_images(samples),
            "sample_count": role_counts["total"],
            "positive_sample_count": role_counts["positive"],
            "negative_sample_count": role_counts["negative"],
            "negative_box_sample_count": role_counts["negative_box"],
            "background_sample_count": role_counts["background"],
            "positive_class_count": len(positive_ids),
            "positive_classes": "|".join(f"{class_id}:{positive_names[index]}" for index, class_id in enumerate(positive_ids)),
            "class_pos_weights": "|".join(f"{positive_ids[index]}:{positive_names[index]}={class_pos_weights[index]:.6f}" for index in range(len(positive_ids))),
            "positive_box_counts_by_class": count_boxes_by_class(samples, positive_ids, positive_names),
            "batch_size": batch_size,
            "train_batch_count": sampler.num_batches if split_name == "train" else math.ceil(len(samples) / batch_size),
            "train_sampler_pos_per_batch": sampler.num_pos_per_batch if split_name == "train" else "",
            "train_sampler_neg_per_batch": sampler.num_neg_per_batch if split_name == "train" else "",
            **model_stats,
        })
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "pretrain_summary.csv", index=False)
    lines = [
        f"Pretrain summary | fraction={fraction}",
        f"  classes: {len(positive_ids)} positive outputs",
        f"  train images/samples: {rows[0]['image_count']} images, {rows[0]['sample_count']} samples "
        f"({rows[0]['positive_sample_count']} pos, {rows[0]['negative_sample_count']} neg)",
        f"  val images/samples: {rows[1]['image_count']} images, {rows[1]['sample_count']} samples "
        f"({rows[1]['positive_sample_count']} pos, {rows[1]['negative_sample_count']} neg)",
        f"  test images/samples: {rows[2]['image_count']} images, {rows[2]['sample_count']} samples "
        f"({rows[2]['positive_sample_count']} pos, {rows[2]['negative_sample_count']} neg)",
        f"  batch: size={batch_size}, train_batches={sampler.num_batches}, pos_per_batch={sampler.num_pos_per_batch}, neg_per_batch={sampler.num_neg_per_batch}",
        f"  class_pos_weight: enabled={config.get('loss', {}).get('class_pos_weight', {}).get('enabled', True)}",
        f"  model: params={model_stats['parameter_count']}, trainable={model_stats['trainable_parameter_count']}, weight_size_mb={model_stats['weight_size_mb']}",
    ]
    text = "\n".join(lines) + "\n"
    (run_dir / "pretrain_summary.txt").write_text(text)
    append_log(run_dir.parent, text)
    append_log(run_dir.parent, f"Pretrain summary saved: {run_dir / 'pretrain_summary.csv'}")
    print(text, flush=True)


def sanity_check_batch(dataset: SKUExperimentDataset, config: Dict[str, Any], scales: Sequence[int], positive_ids: Sequence[int], fraction: float) -> None:
    if not positive_ids:
        raise ValueError("No positive model outputs configured")
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    sampler = BatchRatioSampler(dataset.samples, batch_size, config.get("sampling", {}).get("pos_neg_ratio", "1:1"))
    indices = list(iter(sampler))[:batch_size]
    if not indices:
        raise ValueError(f"Sampler returned no indices for fraction {fraction}")
    for scale in scales:
        dataset.set_fixed_size(scale)
        batch = [dataset[index] for index in indices]
        inputs = torch.stack([item[0] for item in batch], dim=0)
        targets = torch.stack([item[1] for item in batch], dim=0)
        hard_targets = torch.stack([item[2] for item in batch], dim=0)
        expected_shape = (len(batch), 3, scale, scale)
        if tuple(inputs.shape) != expected_shape:
            raise ValueError(f"Bad input tensor shape for scale {scale}: got {tuple(inputs.shape)}, expected {expected_shape}")
        expected_target_shape = (len(batch), len(positive_ids))
        if tuple(targets.shape) != expected_target_shape or tuple(hard_targets.shape) != expected_target_shape:
            raise ValueError(f"Bad target shape for fraction {fraction}: got {tuple(targets.shape)} and {tuple(hard_targets.shape)}, expected {expected_target_shape}")
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            raise ValueError(f"Non-finite tensor values found in first-batch sanity check for fraction {fraction}")


def train_one_epoch(model, dataset: SKUExperimentDataset, config: Dict[str, Any], criterion, optimizer, device, epoch: int, epochs: int, scales: Sequence[int]) -> float:
    model.train()
    losses = []
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    sampler = BatchRatioSampler(dataset.samples, batch_size, config.get("sampling", {}).get("pos_neg_ratio", "1:1"))
    indices = list(iter(sampler))
    batches = [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]
    for batch_indices in tqdm(batches, desc=f"Epoch {epoch}/{epochs}", leave=False):
        dataset.set_fixed_size(int(np.random.choice(scales)))
        batch = [dataset[index] for index in batch_indices]
        inputs = torch.stack([item[0] for item in batch], dim=0)
        targets = torch.stack([item[1] for item in batch], dim=0)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate_outputs(model, dataset: SKUExperimentDataset, criterion, device, scale: int | str):
    dataset.set_fixed_size(scale)
    if scale == "dynamic":
        return evaluate_outputs_dynamic(model, dataset, criterion, device)
    loader = make_loader(dataset, dataset.config, training=False)
    model.eval()
    losses = []
    outputs_all = []
    targets_all = []
    with torch.no_grad():
        for inputs, targets, hard_targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(float(loss.item()))
            outputs_all.append(outputs.cpu().numpy())
            targets_all.append(hard_targets.numpy())
    outputs_np = np.concatenate(outputs_all, axis=0) if outputs_all else np.array([])
    targets_np = np.concatenate(targets_all, axis=0) if targets_all else np.array([])
    return float(np.mean(losses)) if losses else 0.0, outputs_np, targets_np


def evaluate_outputs_dynamic(model, dataset: SKUExperimentDataset, criterion, device):
    model.eval()
    losses = []
    outputs_all = []
    targets_all = []
    with torch.no_grad():
        for index in range(len(dataset)):
            inputs, targets, hard_targets = dataset[index]
            inputs = inputs.unsqueeze(0).to(device)
            targets = targets.unsqueeze(0).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(float(loss.item()))
            outputs_all.append(outputs.cpu().numpy())
            targets_all.append(hard_targets.unsqueeze(0).numpy())
    outputs_np = np.concatenate(outputs_all, axis=0) if outputs_all else np.array([])
    targets_np = np.concatenate(targets_all, axis=0) if targets_all else np.array([])
    return float(np.mean(losses)) if losses else 0.0, outputs_np, targets_np


def tune_per_class_thresholds(
    outputs: np.ndarray,
    hard_targets: np.ndarray,
    candidates: Sequence[float],
    min_support: int,
    default_threshold: float,
) -> np.ndarray:
    if outputs.size == 0 or hard_targets.size == 0:
        return np.full(0, default_threshold, dtype=np.float32)
    thresholds = np.full(outputs.shape[1], default_threshold, dtype=np.float32)
    for class_index in range(outputs.shape[1]):
        y_true = hard_targets[:, class_index]
        support = int(y_true.sum())
        if support < min_support or np.unique(y_true).size < 2:
            continue
        best_threshold = default_threshold
        best_f1 = -1.0
        for candidate in candidates:
            y_pred = (outputs[:, class_index] >= candidate).astype(np.int32)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = candidate
        thresholds[class_index] = best_threshold
    return thresholds


def threshold_row(thresholds: np.ndarray, prefix: str) -> Dict[str, float]:
    return {f"{prefix}_class_{index}": float(value) for index, value in enumerate(thresholds)}


def prefixed_row(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def format_per_class(
    epoch: int,
    fraction: float,
    scale: int,
    split: str,
    rows: List[Dict[str, Any]],
    positive_names: Sequence[str],
) -> List[Dict[str, Any]]:
    formatted = []
    for row in rows:
        class_index = int(row["class_index"])
        class_name = "NEGATIVE" if class_index >= len(positive_names) else positive_names[class_index]
        formatted.append({
            "epoch": epoch,
            "negative_fraction": fraction,
            "scale": scale,
            "split": split,
            **row,
            "class_name": class_name,
        })
    return formatted


def format_thresholds(
    epoch: int,
    fraction: float,
    scale: int,
    thresholds: np.ndarray,
    positive_names: Sequence[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "epoch": epoch,
            "negative_fraction": fraction,
            "scale": scale,
            "split": "threshold",
            "class_index": index,
            "class_name": positive_names[index],
            "support": "",
            "included_in_macro": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "roc_auc": "",
            "selected_threshold": float(threshold),
        }
        for index, threshold in enumerate(thresholds)
    ]
