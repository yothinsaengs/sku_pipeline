import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any

def plot_loss_curve(train_losses: List[float], val_losses: List[float], save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str):
    plt.figure(figsize=(12, 10))
    # cm is (N+1)x(N+1), class_names should be N names + "Negative"
    full_names = class_names + ["Negative"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=full_names, yticklabels=full_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def plot_metrics_curves(metrics_history: List[Dict[str, Any]], save_dir: str):
    df = pd.DataFrame(metrics_history)
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    if 'val_acc' in df.columns:
        plt.plot(df['val_acc'], label='Val Acc')
    if 'val_f1_macro' in df.columns:
        plt.plot(df['val_f1_macro'], label='Val F1 (Macro)')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'val_metrics.png'))
    plt.close()

def plot_aug_preview(batch_images: np.ndarray, save_path: str, num_samples: int = 16):
    # batch_images: [B, 3, H, W] tensor or numpy
    if isinstance(batch_images, np.ndarray):
        imgs = batch_images
    else:
        imgs = batch_images.cpu().numpy()
        
    num_samples = min(num_samples, len(imgs))
    rows = int(np.sqrt(num_samples))
    cols = (num_samples + rows - 1) // rows
    
    plt.figure(figsize=(15, 15))
    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)
        img = np.transpose(imgs[i], (1, 2, 0))
        # Denormalize if needed, assuming [0, 1]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
