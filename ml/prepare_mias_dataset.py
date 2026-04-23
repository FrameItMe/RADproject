"""
Prepare MIAS Mammography dataset for training.
Converts PGM images to PNG and organizes them into train/val/test folders.
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import random
from collections import defaultdict
import re

def parse_info_file(info_path):
    """Parse Info.txt to get image labels (normal/benign/malignant)."""
    labels = {}
    
    with open(info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('REFNUM'):
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue

            if not re.match(r'^mdb\d{3}$', parts[0]):
                continue

            ref_num = parts[0]  # e.g., mdb001
            abnormality = parts[2]  # 3rd column: NORM/CIRC/SPIC/...
            severity = parts[3] if len(parts) > 3 else ''  # 4th column: B/M or missing

            if abnormality == 'NORM':
                label = 'normal'
            elif severity == 'M':
                label = 'malignant'
            elif severity == 'B':
                label = 'benign'
            else:
                # Fallback for rare malformed rows with non-NORM but no severity.
                label = 'benign'

            # Some refs appear multiple times; keep the most severe label.
            prev = labels.get(ref_num)
            if prev == 'malignant' or label == prev:
                continue
            if prev == 'benign' and label == 'normal':
                continue
            labels[ref_num] = label
    
    return labels

def convert_pgm_to_png(pgm_path, png_path):
    """Convert PGM to PNG."""
    try:
        img = Image.open(pgm_path)
        img.save(png_path, 'PNG')
        return True
    except Exception as e:
        print(f"Error converting {pgm_path}: {e}")
        return False

def prepare_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Prepare MIAS dataset for training.
    
    Args:
        data_dir: Path to directory containing all-mias folder
        output_dir: Output directory for organized train/val/test folders
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set (test = 1 - train - val)
    """
    
    mias_folder = Path(data_dir) / 'all-mias'
    info_file = mias_folder / 'Info.txt'
    
    if not mias_folder.exists():
        print(f"Error: {mias_folder} does not exist")
        return
    
    if not info_file.exists():
        print(f"Error: {info_file} does not exist")
        return
    
    # Parse labels
    print("Parsing Info.txt...")
    labels = parse_info_file(str(info_file))
    
    # Group images by class
    by_class = defaultdict(list)
    for ref_num, label in labels.items():
        by_class[label].append(ref_num)
    
    print("Dataset composition:")
    for label, images in by_class.items():
        print(f"  {label}: {len(images)} images")
    
    # Create output directory structure
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    for split in ['train', 'val', 'test']:
        for class_name in ['normal', 'benign', 'malignant']:
            class_dir = output_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # Split images by class
    test_ratio = 1 - train_ratio - val_ratio
    
    print("\nConverting and organizing images...")
    total_converted = 0
    
    for class_name, image_list in by_class.items():
        # Shuffle to split randomly
        shuffled = image_list.copy()
        random.shuffle(shuffled)
        
        # Calculate split indices
        n = len(shuffled)
        train_idx = int(n * train_ratio)
        val_idx = train_idx + int(n * val_ratio)
        
        train_list = shuffled[:train_idx]
        val_list = shuffled[train_idx:val_idx]
        test_list = shuffled[val_idx:]
        
        print(f"\n{class_name}:")
        print(f"  Train: {len(train_list)}, Val: {len(val_list)}, Test: {len(test_list)}")
        
        # Process each image
        for split_name, split_images in [('train', train_list), ('val', val_list), ('test', test_list)]:
            for ref_num in split_images:
                pgm_file = mias_folder / f'{ref_num}.pgm'
                output_file = output_path / split_name / class_name / f'{ref_num}.png'
                
                if pgm_file.exists():
                    if convert_pgm_to_png(str(pgm_file), str(output_file)):
                        total_converted += 1
                    else:
                        print(f"Failed to convert {pgm_file}")
    
    print(f"\nTotal images converted: {total_converted}")
    print(f"Dataset saved to: {output_path}")

if __name__ == '__main__':
    data_dir = r'c:\CODING\yanoey\data'
    output_dir = r'c:\CODING\yanoey\dataset'
    
    prepare_dataset(data_dir, output_dir)
    print("\nDataset preparation complete!")
