import json
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
    criterion = get_loss_fn(config)
    optimizer = build_optimizer(model, config)
    epochs = int(config.get("training", {}).get("epochs", 50))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    threshold = float(config.get("threshold", 0.5))
    scales = [int(size) for size in config.get("input", {}).get("sizes", [56, 112, 224])]
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
        for scale in scales:
            val_loss, val_metrics, val_per_class = evaluate(model, val_ds, criterion, device, threshold, scale)
            test_loss, test_metrics, test_per_class = evaluate(model, test_ds, criterion, device, threshold, scale)
            val_row = prefixed_row(val_metrics, "val")
            test_row = prefixed_row(test_metrics, "test")
            row = {
                "epoch": epoch,
                "negative_fraction": fraction,
                "scale": scale,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                **val_row,
                **test_row,
            }
            epoch_rows.append(row)
            val_scale_rows.append((scale, val_metrics))
            test_scale_rows.append((scale, test_metrics))

            if epoch % per_class_interval == 0:
                per_class_rows.extend(format_per_class(epoch, fraction, scale, "val", val_per_class, positive_names))
                per_class_rows.extend(format_per_class(epoch, fraction, scale, "test", test_per_class, positive_names))

        pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
        if per_class_rows:
            pd.DataFrame(per_class_rows).to_csv(run_dir / f"per_class_metrics_epoch_{epoch:03d}.csv", index=False)

        mean_val_f1 = float(np.mean([metrics["f1_macro"] for _, metrics in val_scale_rows])) if val_scale_rows else 0.0
        if mean_val_f1 > best_val_f1:
            best_val_f1 = mean_val_f1
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, run_dir / "checkpoints" / "best_model.pth")
        torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, run_dir / "checkpoints" / "last_model.pth")

    best_path = run_dir / "checkpoints" / "best_model.pth"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    for scale in scales:
        loss, metrics, _ = evaluate(model, test_ds, criterion, device, threshold, scale)
        final_test_rows.append({"negative_fraction": fraction, "scale": scale, "test_loss": loss, **metrics})
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


def evaluate(model, dataset: SKUExperimentDataset, criterion, device, threshold: float, scale: int):
    dataset.set_fixed_size(scale)
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
    metrics, per_class = calculate_metrics(outputs_np, targets_np, threshold)
    return float(np.mean(losses)) if losses else 0.0, metrics, per_class


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
