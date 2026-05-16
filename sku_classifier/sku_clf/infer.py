import torch
import cv2
import numpy as np
import os
from typing import Dict, Any, Optional, List
from sku_clf.models.backbone import get_model
from sku_clf.data.dataset import SKUROIDataset # For letterbox

def infer(config: Dict[str, Any], 
          checkpoint_path: str, 
          image: Optional[str] = None, 
          boxes: Optional[str] = None, 
          crops: Optional[str] = None,
          save_vis: Optional[str] = None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    pos_classes = [c for c in config['classes'] if c['role'] == 'positive']
    pos_names = [c['name'] for c in pos_classes]
    
    max_size = config['input'].get('max_size', 224)
    context_margin = config['input'].get('context_margin', 0.15)
    pad_color = config['input'].get('pad_color', 114)
    threshold = config.get('threshold', 0.5)

    def preprocess_crop(crop):
        # Resize and letterbox
        # We can reuse SKUROIDataset._letterbox by making it a static or standalone utility
        # For now, let's just implement a quick version or import
        # I'll just copy it here for simplicity in this module
        h, w = crop.shape[:2]
        r = min(max_size / h, max_size / w)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = (max_size - new_unpad[0]) / 2, (max_size - new_unpad[1]) / 2
        if (w, h) != new_unpad:
            crop = cv2.resize(crop, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_color, pad_color, pad_color))
        
        # To tensor
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return tensor

    results = []
    
    if crops:
        # Mode B: Pre-cropped
        img = cv2.imread(crops)
        tensor = preprocess_crop(img).to(device)
        with torch.no_grad():
            outputs = model(tensor).cpu().numpy()[0]
        
        for i, conf in enumerate(outputs):
            if conf > threshold:
                results.append({'class': pos_names[i], 'confidence': float(conf)})
        print(f"Crops {crops}: {results}")
        
    elif image and boxes:
        # Mode A: Image + Boxes
        full_img = cv2.imread(image)
        h, w = full_img.shape[:2]
        
        with open(boxes, 'r') as f:
            lines = f.readlines()
            
        vis_img = full_img.copy() if save_vis else None
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5: continue
            
            # YOLO: class, xc, yc, bw, bh
            xc, yc, bw, bh = map(float, parts[1:])
            xc, yc, bw, bh = xc * w, yc * h, bw * w, bh * h
            
            # Expand
            bw_ext, bh_ext = bw * (1 + context_margin), bh * (1 + context_margin)
            x1, y1 = max(0, int(xc - bw_ext/2)), max(0, int(yc - bh_ext/2))
            x2, y2 = min(w, int(xc + bw_ext/2)), min(h, int(yc + bh_ext/2))
            
            crop = full_img[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            tensor = preprocess_crop(crop).to(device)
            with torch.no_grad():
                outputs = model(tensor).cpu().numpy()[0]
            
            for i, conf in enumerate(outputs):
                if conf > threshold:
                    res = {'bbox': [x1, y1, x2, y2], 'class': pos_names[i], 'confidence': float(conf)}
                    results.append(res)
                    if vis_img is not None:
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis_img, f"{pos_names[i]} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"Image {image} results: {results}")
        if save_vis:
            cv2.imwrite(save_vis, vis_img)
            print(f"Visualization saved to {save_vis}")
            
    return results
