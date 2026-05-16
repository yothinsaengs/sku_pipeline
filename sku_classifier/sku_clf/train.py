import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional

from sku_clf.data.dataset import get_dataloaders
from sku_clf.models.backbone import get_model
from sku_clf.models.loss import get_loss_fn
from sku_clf.utils.logger import RunLogger
from sku_clf.utils.metrics import calculate_metrics, get_confusion_matrix
from sku_clf.utils.plots import plot_loss_curve, plot_metrics_curves, plot_confusion_matrix, plot_aug_preview

def train(config: Dict[str, Any], data_path: str):
    train_loader, val_loader, test_loader = get_dataloaders(config, data_path)
    return train_with_dataloaders(config, train_loader, val_loader, test_loader)

def train_with_dataloaders(config: Dict[str, Any], train_loader, val_loader, test_loader):
    # Setup Logger
    logger = RunLogger(config)
    logger.log("Starting training session")
    
    # Device
    device_name = config['training'].get('device', 'auto')
    if device_name == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_name)
    logger.log(f"Using device: {device}")
    
    logger.log(f"Data loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples")
    logger.log(f"Batch sampler pos:neg ratio: {config.get('sampling', {}).get('pos_neg_ratio', '1:1')}")
    if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'num_pos_per_batch'):
        logger.log(
            f"Expected train batch mix: {train_loader.sampler.num_pos_per_batch} positive, "
            f"{train_loader.sampler.num_neg_per_batch} negative; sampler uses replacement when needed"
        )
    
    # Model
    model = get_model(config).to(device)
    
    # Loss, Optimizer, Scheduler
    criterion = get_loss_fn(config)
    
    optimizer_name = config['training'].get('optimizer', 'adamw')
    lr = config['training'].get('lr', 0.0003)
    wd = config['training'].get('weight_decay', 0.01)
    
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    epochs = config['training'].get('epochs', 50)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Early Stopping
    early_stopping_config = config['training'].get('early_stopping', {})
    best_val_f1 = -1
    patience = early_stopping_config.get('patience', 10)
    patience_counter = 0
    
    # Aug Preview
    if config['logging'].get('aug_preview', {}).get('enabled'):
        batch, _ = next(iter(train_loader))
        plot_aug_preview(batch, os.path.join(logger.run_dir, 'aug_preview.png'))
    
    # Training Loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Multiscale logic
        current_size = config['input'].get('max_size', 224)
        if config['training'].get('multiscale', False):
            # Sample a random size from 128 to max_size in steps of 32
            sizes = list(range(128, config['input'].get('max_size', 224) + 1, 32))
            current_size = np.random.choice(sizes)
            train_loader.dataset.set_target_size(current_size)
            logger.log(f"Epoch {epoch+1} | Multiscale active | Input size: {current_size}x{current_size}")
        else:
            logger.log(f"Epoch {epoch+1} | Input size: {current_size}x{current_size}")

        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Class Mixing
            mix_config = config['augmentation'].get('class_mixing', {})
            if mix_config.get('enabled') and np.random.random() < mix_config.get('prob', 0.3):
                from sku_clf.data.augment import mixup_data, cutmix_data
                strategy = mix_config.get('strategy', 'both')
                
                if strategy == 'mixup' or (strategy == 'both' and np.random.random() < 0.5):
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mix_config.get('mixup', {}).get('alpha', 0.4), device)
                else:
                    inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, mix_config.get('cutmix', {}).get('alpha', 1.0), device)
                
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (i + 1)})
            
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device, config['threshold'])
        val_losses.append(val_loss)
        
        # Logging
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'val_loss': val_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        logger.save_metrics(epoch_log)
        logger.log(f"Epoch {epoch+1} | train_loss: {epoch_train_loss:.4f} | val_loss: {val_loss:.4f} | val_f1: {val_metrics['f1_macro']:.4f}")
        
        # Checkpoint
        is_best = val_metrics['f1_macro'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['f1_macro']
            patience_counter = 0
            logger.save_checkpoint(model, epoch + 1, is_best=True)
        else:
            patience_counter += 1
            
        logger.save_checkpoint(model, epoch + 1, is_best=False)
        
        # Scheduler Step
        scheduler.step()
        
        # Plots (every few epochs)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            plot_loss_curve(train_losses, val_losses, os.path.join(logger.run_dir, 'loss_curve.png'))
            plot_metrics_curves(logger.metrics_history, logger.run_dir)
            
        # Early Stopping
        if early_stopping_config.get('enabled') and patience_counter >= patience:
            logger.log(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Final Evaluation on Test Set
    logger.log("Training complete. Running final evaluation on test set...")
    # Load best model
    best_path = os.path.join(logger.run_dir, 'checkpoints', 'best_model.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics, outputs_all, targets_all = evaluate_epoch(model, test_loader, criterion, device, config['threshold'], return_all=True)
    
    # Save final results
    summary = {
        'test_metrics': test_metrics,
        'config': config
    }
    logger.save_summary(summary)
    
    # Final Plots
    pos_class_names = [c['name'] for c in config['classes'] if c['role'] == 'positive']
    cm = get_confusion_matrix(outputs_all, targets_all, config['threshold'])
    plot_confusion_matrix(cm, pos_class_names, os.path.join(logger.run_dir, 'confusion_matrix.png'))
    
    logger.log(f"Final Test F1: {test_metrics['f1_macro']:.4f}")

def evaluate_epoch(model, loader, criterion, device, threshold, return_all=False):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
    all_outputs = np.concatenate(all_outputs, axis=0) if all_outputs else np.array([])
    all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
    
    metrics = calculate_metrics(all_outputs, all_targets, threshold)
    
    if return_all:
        return avg_loss, metrics, all_outputs, all_targets
    return avg_loss, metrics
