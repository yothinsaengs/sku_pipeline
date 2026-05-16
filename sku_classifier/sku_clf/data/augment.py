import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import cv2
import numpy as np
import torch
from typing import Dict, Any

def get_transforms(config: Dict[str, Any], is_training: bool = True):
    if not is_training:
        return A.Compose([
            # No normalization for now, just to tensor
            # The dataset already converts to float and divides by 255
            # ToTensorV2()
        ])

    aug_config = config['augmentation']
    
    # Safe augmentations
    safe_augs = []
    if aug_config['safe'].get('horizontal_flip'):
        safe_augs.append(A.HorizontalFlip(p=0.5))
    if aug_config['safe'].get('vertical_flip'):
        safe_augs.append(A.VerticalFlip(p=0.5))
    
    jitter = aug_config['safe'].get('color_jitter', {})
    if jitter:
        safe_augs.append(A.ColorJitter(
            brightness=jitter.get('brightness', 0.2),
            contrast=jitter.get('contrast', 0.2),
            saturation=jitter.get('saturation', 0.2),
            hue=jitter.get('hue', 0.05),
            p=0.5
        ))
    
    rot_deg = aug_config['safe'].get('random_rotation_deg', 0)
    if rot_deg > 0:
        safe_augs.append(A.Rotate(limit=rot_deg, p=0.5))

    # Risky augmentations
    risky_augs = []
    for risk in aug_config.get('random_mix', []):
        name = risk['name']
        prob = risk['prob']
        params = risk['params']
        
        if name == 'aggressive_color_jitter':
            risky_augs.append(A.ColorJitter(**params, p=prob))
        elif name == 'slight_perspective':
            risky_augs.append(A.Perspective(**params, p=prob))
        elif name == 'gaussian_blur':
            risky_augs.append(A.GaussianBlur(**params, p=prob))

    # Random Occlusion
    occ_config = aug_config.get('random_occlusion', {})
    if occ_config.get('enabled'):
        risky_augs.append(RandomOcclusion(
            prob=occ_config.get('prob', 0.3),
            max_count=occ_config.get('max_count', 4),
            shapes=occ_config.get('shapes', ["rect", "ellipse", "polygon"]),
            scale_range=occ_config.get('scale_range', [0.05, 0.25]),
            fill=occ_config.get('fill', "random")
        ))
    
    return A.Compose(safe_augs + risky_augs)

# Custom transform for Random Occlusion
class RandomOcclusion(A.ImageOnlyTransform):
    def __init__(self, 
                 prob=0.3, 
                 max_count=4, 
                 shapes=["rect", "ellipse", "polygon"], 
                 scale_range=[0.05, 0.25], 
                 fill="random",
                 always_apply=False, 
                 p=1.0):
        super(RandomOcclusion, self).__init__(always_apply, p)
        self.prob = prob
        self.max_count = max_count
        self.shapes = shapes
        self.scale_range = scale_range
        self.fill = fill

    def apply(self, image, **params):
        if random.random() > self.prob:
            return image
        
        img = image.copy()
        h, w = img.shape[:2]
        count = random.randint(1, self.max_count)
        
        for _ in range(count):
            shape_type = random.choice(self.shapes)
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            sw, sh = int(w * scale), int(h * scale)
            
            x = random.randint(0, w - sw)
            y = random.randint(0, h - sh)
            
            if self.fill == "random":
                color = [random.randint(0, 255) for _ in range(3)]
            elif self.fill == "black":
                color = [0, 0, 0]
            elif self.fill == "mean":
                color = [int(x) for x in img.mean(axis=(0,1))]
            else:
                color = [114, 114, 114]

            if shape_type == "rect":
                cv2.rectangle(img, (x, y), (x + sw, y + sh), color, -1)
            elif shape_type == "ellipse":
                cv2.ellipse(img, (x + sw // 2, y + sh // 2), (sw // 2, sh // 2), 0, 0, 360, color, -1)
            elif shape_type == "polygon":
                num_pts = random.randint(3, 6)
                pts = np.array([[random.randint(x, x + sw), random.randint(y, y + sh)] for _ in range(num_pts)])
                cv2.fillPoly(img, [pts], color)

        return img
        
def mixup_data(x, y, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, device='cpu'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    y_a, y_b = y, y[index]
    
    # Generate bbox
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exact area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
