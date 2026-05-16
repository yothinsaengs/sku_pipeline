import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Any

def get_background_patch(image: np.ndarray, bboxes: List[List[float]], scale_range: Tuple[float, float]) -> np.ndarray:
    # bboxes: [[xc, yc, w, h], ...] normalized
    h, w = image.shape[:2]
    
    # Simple strategy: try random crops until we find one with low overlap with bboxes
    # or just use regions we know are background.
    
    max_tries = 10
    for _ in range(max_tries):
        scale = random.uniform(scale_range[0], scale_range[1])
        sw, sh = int(w * scale), int(h * scale)
        
        x1 = random.randint(0, w - sw)
        y1 = random.randint(0, h - sh)
        x2, y2 = x1 + sw, y1 + sh
        
        # Check overlap
        overlap = False
        for bbox in bboxes:
            bxc, byc, bw, bh = bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h
            bx1, by1 = bxc - bw/2, byc - bh/2
            bx2, by2 = bxc + bw/2, byc + bh/2
            
            # Intersection
            ix1, iy1 = max(x1, bx1), max(y1, by1)
            ix2, iy2 = min(x2, bx2), min(y2, by2)
            
            if ix1 < ix2 and iy1 < iy2:
                overlap = True
                break
        
        if not overlap:
            return image[y1:y2, x1:x2]
            
    # If no background found, return a random crop (fallback)
    return image[0:100, 0:100]

def generate_synthetic_negative(image: np.ndarray, bboxes: List[List[float]], config: Dict[str, Any]) -> np.ndarray:
    extra_cfg = config['extra_negatives']['synthetic_cutpaste']
    scale_range = extra_cfg.get('scale_range', [0.05, 0.3])
    
    patch = get_background_patch(image, bboxes, scale_range)
    bg_crop = get_background_patch(image, bboxes, scale_range)
    
    # Resize patch to fit bg_crop or vice versa
    # For CutPaste, we usually paste a small patch onto a larger crop
    ph, pw = patch.shape[:2]
    bh, bw = bg_crop.shape[:2]
    
    # Ensure bg_crop is large enough
    if bh < 32 or bw < 32:
        return np.zeros((224, 224, 3), dtype=np.uint8)
        
    # Resize patch if too large
    if ph > bh or pw > bw:
        r = min(bh/ph, bw/pw) * 0.5
        patch = cv2.resize(patch, (int(pw*r), int(ph*r)))
        ph, pw = patch.shape[:2]
        
    res = bg_crop.copy()
    px = random.randint(0, bw - pw)
    py = random.randint(0, bh - ph)
    
    # Shape masking
    shape = random.choice(extra_cfg.get('shapes', ['rect', 'ellipse']))
    mask = np.zeros((ph, pw), dtype=np.uint8)
    if shape == 'rect':
        mask[:] = 255
    elif shape == 'ellipse':
        cv2.ellipse(mask, (pw//2, ph//2), (pw//2, ph//2), 0, 0, 360, 255, -1)
    else: # polygon
        pts = np.array([[random.randint(0, pw), random.randint(0, ph)] for _ in range(5)])
        cv2.fillPoly(mask, [pts], 255)
        
    for c in range(3):
        res[py:py+ph, px:px+pw, c] = res[py:py+ph, px:px+pw, c] * (1 - mask/255) + patch[:, :, c] * (mask/255)
        
    return res
