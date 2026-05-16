import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple
import glob

class SKUROIDataset(Dataset):
    def __init__(self, 
                 image_paths: List[str], 
                 label_paths: List[str], 
                 config: Dict[str, Any], 
                 transform=None,
                 is_training: bool = True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.config = config
        self.transform = transform
        self.is_training = is_training
        
        self.classes_config = config['classes']
        self.id_to_role = {c['id']: c['role'] for c in self.classes_config}
        self.id_to_name = {c['id']: c['name'] for c in self.classes_config}
        
        # Map positive class IDs to [0, num_pos - 1]
        self.pos_classes = [c for c in self.classes_config if c['role'] == 'positive']
        self.pos_id_to_idx = {c['id']: i for i, c in enumerate(self.pos_classes)}
        self.num_pos_classes = len(self.pos_classes)
        
        self.min_size = config['input'].get('min_size', 4)
        self.max_size = config['input'].get('max_size', 224)
        self.context_margin = config['input'].get('context_margin', 0.15)
        self.pad_color = config['input'].get('pad_color', 114)
        self.target_size = self.max_size
        
        self.samples = self._prepare_samples()

    def set_target_size(self, size: int):
        self.target_size = size

    def _prepare_samples(self) -> List[Dict[str, Any]]:
        samples = []

        for img_path, lbl_path in zip(self.image_paths, self.label_paths):
            if not os.path.exists(lbl_path):
                continue
            
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            image_bboxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                image_bboxes.append([x_center, y_center, width, height])

                if class_id not in self.id_to_role:
                    continue
                role = self.id_to_role[class_id]
                
                samples.append({
                    'image_path': img_path,
                    'bbox': [x_center, y_center, width, height],
                    'class_id': class_id,
                    'role': role
                })

            # Add synthetic negatives if enabled
            extra_negatives = self.config.get('extra_negatives', {})
            if self.is_training and extra_negatives.get('enabled'):
                synthetic_cfg = extra_negatives.get('synthetic_cutpaste', {})
                if synthetic_cfg.get('enabled'):
                    samples.append({
                        'image_path': img_path,
                        'image_bboxes': image_bboxes,
                        'role': 'synthetic'
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def _letterbox(self, img: np.ndarray, new_shape: Tuple[int, int], color: int = 114) -> np.ndarray:
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color))
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        role = sample['role']
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        if role == 'synthetic':
            from sku_clf.data.extra_neg import generate_synthetic_negative
            crop = generate_synthetic_negative(img, sample['image_bboxes'], self.config)
            crop = self._letterbox(crop, (self.target_size, self.target_size), color=self.pad_color)
            class_id = -1
        else:
            bbox = sample['bbox']
            class_id = sample['class_id']
            xc, yc, bw, bh = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
            bw_ext, bh_ext = bw * (1 + self.context_margin), bh * (1 + self.context_margin)
            x1, y1 = max(0, int(xc - bw_ext / 2)), max(0, int(yc - bh_ext / 2))
            x2, y2 = min(w, int(xc + bw_ext / 2)), min(h, int(yc + bh_ext / 2))
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] < self.min_size or crop.shape[1] < self.min_size:
                crop = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            else:
                crop = self._letterbox(crop, (self.target_size, self.target_size), color=self.pad_color)
        
        if self.transform:
            augmented = self.transform(image=crop)
            crop = augmented['image']
        
        target = np.zeros(self.num_pos_classes, dtype=np.float32)
        if role == 'positive':
            target[self.pos_id_to_idx[class_id]] = 1.0
        
        return torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0, torch.from_numpy(target)

def get_dataloaders(config: Dict[str, Any], data_path: str):
    from sku_clf.utils.io import extract_dataset, resolve_data_paths
    from sku_clf.data.augment import get_transforms
    from sku_clf.data.sampler import BatchRatioSampler
    from torch.utils.data import DataLoader
    import random

    data_dir = extract_dataset(data_path)
    images_dir, labels_dir = resolve_data_paths(data_dir)
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
    
    num_images = len(valid_images)
    indices = list(range(num_images))
    random.seed(config['dataset'].get('split_seed', 42))
    random.shuffle(indices)
    
    train_split = config['dataset']['split']['train']
    val_split = config['dataset']['split']['val']
    num_train = int(num_images * train_split)
    num_val = int(num_images * val_split)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]
    
    def create_split(idxs):
        return [valid_images[i] for i in idxs], [valid_labels[i] for i in idxs]

    train_imgs, train_lbls = create_split(train_indices)
    val_imgs, val_lbls = create_split(val_indices)
    test_imgs, test_lbls = create_split(test_indices)
    
    train_ds = SKUROIDataset(train_imgs, train_lbls, config, transform=get_transforms(config, True), is_training=True)
    val_ds = SKUROIDataset(val_imgs, val_lbls, config, transform=get_transforms(config, False), is_training=False)
    test_ds = SKUROIDataset(test_imgs, test_lbls, config, transform=get_transforms(config, False), is_training=False)
    
    batch_size = config['training']['batch_size']
    ratio = config['sampling'].get('pos_neg_ratio', "1:1")
    train_sampler = BatchRatioSampler(train_ds.samples, batch_size, ratio)
    # Loaders
    num_workers = config['training'].get('num_workers', 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    
    return train_loader, val_loader, test_loader
