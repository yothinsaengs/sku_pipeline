import os
import yaml
import glob
from typing import List, Dict, Any, Set
from sku_clf.utils.io import extract_dataset, resolve_data_paths

def get_unique_classes(labels_dir: str) -> Set[int]:
    class_ids = set()
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    for f in label_files:
        with open(f, 'r') as lbl:
            for line in lbl:
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))
    return class_ids

def find_class_names(data_dir: str, class_ids: Set[int]) -> Dict[int, str]:
    # Try common YOLO name file locations
    name_files = [
        os.path.join(data_dir, "classes.txt"),
        os.path.join(data_dir, "obj.names"),
        os.path.join(os.path.dirname(data_dir), "classes.txt")
    ]
    
    names = {}
    for nf in name_files:
        if os.path.exists(nf):
            with open(nf, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                for idx, name in enumerate(lines):
                    if idx in class_ids:
                        names[idx] = name
            break
            
    # Fill missing with generic
    for cid in class_ids:
        if cid not in names:
            names[cid] = f"class_{cid}"
            
    return names

def interactive_config_gen(data_path: str, output_config: str = "config.yaml"):
    print(f"--- SKU Classifier Configuration Generator ---")
    
    # 1. Extract/Resolve
    data_dir = extract_dataset(data_path)
    images_dir, labels_dir = resolve_data_paths(data_dir)
    
    # 2. Scan classes
    class_ids = get_unique_classes(labels_dir)
    if not class_ids:
        print("Error: No class IDs found in labels.")
        return
    
    id_to_name = find_class_names(data_dir, class_ids)
    
    print(f"\nFound {len(class_ids)} unique classes in the dataset:")
    sorted_ids = sorted(list(class_ids))
    for cid in sorted_ids:
        print(f"  [{cid}] {id_to_name[cid]}")
        
    # 3. User Selection
    print("\n--- Positive Class Selection ---")
    print("Enter the IDs of POSITIVE classes (comma separated, e.g. 0,1):")
    pos_input = input("> ").strip()
    pos_ids = [int(x.strip()) for x in pos_input.split(',') if x.strip().isdigit()]
    
    print("\n--- Negative Class Selection ---")
    print("Enter the IDs of NEGATIVE classes (comma separated, or type 'all' for all remaining):")
    neg_input = input("> ").strip().lower()
    
    neg_ids = []
    if neg_input == 'all':
        neg_ids = [cid for cid in sorted_ids if cid not in pos_ids]
    else:
        neg_ids = [int(x.strip()) for x in neg_input.split(',') if x.strip().isdigit()]
        
    # 4. Construct classes list
    classes_list = []
    for cid in pos_ids:
        classes_list.append({'id': cid, 'name': id_to_name[cid], 'role': 'positive'})
    for cid in neg_ids:
        classes_list.append({'id': cid, 'name': id_to_name[cid], 'role': 'negative'})
        
    # 5. Save/Update Config
    config = {}
    if os.path.exists(output_config):
        with open(output_config, 'r') as f:
            config = yaml.safe_load(f)
            
    # Update classes and dataset paths
    config['classes'] = classes_list
    if 'dataset' not in config: config['dataset'] = {}
    config['dataset']['images_dir'] = images_dir
    config['dataset']['labels_dir'] = labels_dir
    config['model']['num_classes'] = len(pos_ids)
    
    with open(output_config, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
        
    print(f"\nSuccess! Updated {output_config} with {len(pos_ids)} positive and {len(neg_ids)} negative classes.")
