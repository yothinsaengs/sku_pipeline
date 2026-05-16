import os
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Optional
from sku_clf.infer import infer

def infer_batch(config: Dict[str, Any], 
                checkpoint_path: str, 
                images_dir: Optional[str] = None, 
                boxes_dir: Optional[str] = None, 
                crops_dir: Optional[str] = None,
                output_dir: str = "results"):
    
    os.makedirs(output_dir, exist_ok=True)
    summary_data = []
    
    if crops_dir:
        # Batch Mode B: Pre-cropped directory
        crop_files = [f for f in os.listdir(crops_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in tqdm(crop_files, desc="Batch Inference (Crops)"):
            path = os.path.join(crops_dir, f)
            res = infer(config, checkpoint_path, crops=path)
            
            # Save individual JSON
            with open(os.path.join(output_dir, f + ".json"), 'w') as jf:
                json.dump(res, jf, indent=4)
                
            # Aggregate for summary
            counts = {}
            for r in res:
                counts[r['class']] = counts.get(r['class'], 0) + 1
            summary_data.append({'filename': f, **counts})
            
    elif images_dir and boxes_dir:
        # Batch Mode A: Images + Boxes directories
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in tqdm(image_files, desc="Batch Inference (Images)"):
            img_path = os.path.join(images_dir, f)
            basename = os.path.splitext(f)[0]
            box_path = os.path.join(boxes_dir, basename + ".txt")
            
            if not os.path.exists(box_path):
                continue
                
            res = infer(config, checkpoint_path, image=img_path, boxes=box_path)
            
            # Save individual JSON
            with open(os.path.join(output_dir, basename + ".json"), 'w') as jf:
                json.dump(res, jf, indent=4)
                
            # Aggregate for summary
            counts = {}
            for r in res:
                counts[r['class']] = counts.get(r['class'], 0) + 1
            summary_data.append({'filename': f, **counts})
            
    # Save batch summary CSV
    df = pd.DataFrame(summary_data).fillna(0)
    df.to_csv(os.path.join(output_dir, 'batch_summary.csv'), index=False)
    print(f"Batch inference complete. Results saved to {output_dir}")
