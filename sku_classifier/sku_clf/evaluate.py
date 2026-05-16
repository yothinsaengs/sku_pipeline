import torch
import os
import json
import numpy as np
from typing import Dict, Any, Optional

from sku_clf.data.dataset import SKUROIDataset, get_transforms
from sku_clf.models.backbone import get_model
from sku_clf.models.loss import get_loss_fn
from sku_clf.utils.metrics import calculate_metrics, get_confusion_matrix
from sku_clf.utils.plots import plot_confusion_matrix
from sku_clf.utils.io import extract_dataset, resolve_data_paths
import glob

def evaluate(config: Dict[str, Any], checkpoint_path: str, data_path: Optional[str] = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Data Path
    if data_path is None:
        # Try to use path from config if available or error
        data_path = config['dataset'].get('images_dir', 'dataset')
    
    data_dir = extract_dataset(data_path)
    images_dir, labels_dir = resolve_data_paths(data_dir)
    
    # List all images
    img_exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    image_files = []
    for ext in img_exts:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    image_files.sort()
    
    valid_images = []
    valid_labels = []
    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(labels_dir, basename + ".txt")
        if os.path.exists(lbl_path):
            valid_images.append(img_path)
            valid_labels.append(lbl_path)
            
    dataset = SKUROIDataset(valid_images, valid_labels, config, transform=get_transforms(config, False), is_training=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['training'].get('batch_size', 32), shuffle=False)
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = calculate_metrics(all_outputs, all_targets, config['threshold'])
    print(f"Evaluation Metrics: {metrics}")
    
    # Save results
    save_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(save_dir, 'eval_results.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # CM
    pos_class_names = [c['name'] for c in config['classes'] if c['role'] == 'positive']
    cm = get_confusion_matrix(all_outputs, all_targets, config['threshold'])
    plot_confusion_matrix(cm, pos_class_names, os.path.join(save_dir, 'eval_confusion_matrix.png'))
