import zipfile
import os
import shutil

def extract_dataset(zip_path: str, extract_to: str = "temp_dataset"):
    if not zipfile.is_zipfile(zip_path):
        return zip_path # Assume it's a directory
    
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    return extract_to

def resolve_data_paths(data_dir: str):
    # Standard YOLO format: images/ and labels/
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        # Maybe they are nested? 
        # Search for them
        found_images = False
        found_labels = False
        for root, dirs, files in os.walk(data_dir):
            if "images" in dirs:
                images_dir = os.path.join(root, "images")
                found_images = True
            if "labels" in dirs:
                labels_dir = os.path.join(root, "labels")
                found_labels = True
            if found_images and found_labels:
                break
    
    return images_dir, labels_dir
