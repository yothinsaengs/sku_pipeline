import os
import yaml
import glob
import questionary
from typing import List, Dict, Any, Set
from sku_clf.utils.io import extract_dataset, resolve_data_paths

def get_class_stats(labels_dir: str) -> Dict[int, int]:
    counts = {}
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    for f in label_files:
        with open(f, 'r') as lbl:
            for line in lbl:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    counts[class_id] = counts.get(class_id, 0) + 1
    return counts

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
    print(f"\n--- SKU Classifier Configuration Generator ---")
    
    # 1. Extract/Resolve
    data_dir = extract_dataset(data_path)
    images_dir, labels_dir = resolve_data_paths(data_dir)
    
    # 2. Scan classes
    stats = get_class_stats(labels_dir)
    if not stats:
        print("Error: No class IDs found in labels.")
        return
    
    class_ids = set(stats.keys())
    id_to_name = find_class_names(data_dir, class_ids)
    sorted_ids = sorted(list(class_ids))
    
    # 3. Positive Selection using Questionary
    choices = [
        questionary.Choice(
            title=f"{cid:3}: {id_to_name[cid]:15} (Count: {stats[cid]})",
            value=cid
        ) for cid in sorted_ids
    ]
    
    pos_ids = questionary.checkbox(
        "Select POSITIVE classes:",
        choices=choices
    ).ask()
    
    if pos_ids is None or not pos_ids:
        print("Operation cancelled or no positive classes selected.")
        return

    # 4. Negative Selection from remaining
    remaining_ids = [cid for cid in sorted_ids if cid not in pos_ids]
    if remaining_ids:
        neg_choices = [
            questionary.Choice(
                title=f"{cid:3}: {id_to_name[cid]:15} (Count: {stats[cid]})",
                value=cid
            ) for cid in remaining_ids
        ]
        
        # Add "Select All" logic or just use checkbox
        neg_ids = questionary.checkbox(
            "Select NEGATIVE classes:",
            choices=neg_choices
        ).ask()
        
        if neg_ids is None:
            neg_ids = []
    else:
        neg_ids = []
        
    # 5. Construct classes list
    classes_list = []
    for cid in sorted(pos_ids):
        classes_list.append({'id': cid, 'name': id_to_name[cid], 'role': 'positive'})
    for cid in sorted(neg_ids):
        classes_list.append({'id': cid, 'name': id_to_name[cid], 'role': 'negative'})
        
    # 6. Save/Update Config
    config = {}
    if os.path.exists(output_config):
        with open(output_config, 'r') as f:
            config = yaml.safe_load(f)
            
    if 'model' not in config: config['model'] = {}
    if 'dataset' not in config: config['dataset'] = {}
    
    config['classes'] = classes_list
    config['dataset']['images_dir'] = images_dir
    config['dataset']['labels_dir'] = labels_dir
    config['model']['num_classes'] = len(pos_ids)
    
    with open(output_config, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
        
    print(f"\nSuccess! Updated {output_config}")
    print(f"  Positive: {len(pos_ids)} classes")
    print(f"  Negative: {len(neg_ids)} classes")
