"""
preprocess.py - Road Damage Dataset Preprocessing Pipeline
Matches images with labels, creates train/val/test splits, and generates data.yaml.
"""

import argparse
import os
import shutil
import random
from pathlib import Path
from collections import Counter
import yaml

DEFAULT_CLASSES = {0: "Pothole", 1: "Crack", 2: "Manhole"}
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def find_image_label_pairs(source_path: Path):
    """Finds matching image and YOLO annotation .txt files."""
    image_label_pairs = []
    source_path = Path(source_path)
    
    all_images = [p for p in source_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    
    for img_path in all_images:
        label_name = img_path.stem + ".txt"
        
        possible_label_paths = [
            img_path.parent / label_name,
            img_path.parent.parent / "labels" / label_name,
            img_path.parent.parent / "labels-YOLO" / label_name,
            source_path / "labels" / label_name,
            source_path / "labels-YOLO" / label_name,
        ]
        
        for lbl_path in possible_label_paths:
            if lbl_path.exists():
                image_label_pairs.append((img_path, lbl_path))
                break
                
    image_label_pairs = list(set(image_label_pairs))
    print(f"✅ Found {len(image_label_pairs)} valid image-label pairs.")
    return image_label_pairs


def split_and_copy_dataset(
    pairs: list, 
    dest_path: Path, 
    subset_ratio: float = 1.0, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.2
):
    """Shuffles and splits image-label pairs into train, val, and test folders."""
    random.seed(42)
    random.shuffle(pairs)
    
    dest_path = Path(dest_path)
    subset_size = int(len(pairs) * subset_ratio)
    pairs = pairs[:subset_size]
    print(f"📦 Using {len(pairs)} pairs ({subset_ratio * 100:.0f}% of total).")
    
    n_train = int(len(pairs) * train_ratio)
    n_val = int(len(pairs) * val_ratio)
    
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:]
    }
    
    for split_name, split_pairs in splits.items():
        img_dest_dir = dest_path / split_name / "images"
        lbl_dest_dir = dest_path / split_name / "labels"
        
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, img_dest_dir / img_path.name)
            shutil.copy2(lbl_path, lbl_dest_dir / lbl_path.name)
            
        print(f"   📂 {split_name}: {len(split_pairs)} samples")


def create_data_yaml(dest_path: Path, class_names: dict = DEFAULT_CLASSES):
    """Generates the data.yaml configuration file for YOLO."""
    dest_path = Path(dest_path)
    yaml_data = {
        "path": str(dest_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": class_names
    }
    yaml_file = dest_path / "data.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
        
    print(f"📄 Created data.yaml at {yaml_file}")


def analyze_dataset_distribution(labels_dir: Path, class_names: dict = DEFAULT_CLASSES):
    """Prints the distribution of damage classes (Pothole, Crack, Manhole)."""
    labels_dir = Path(labels_dir)
    class_counts = Counter()
    
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    
    total = sum(class_counts.values())
    print("\n📊 Class Distribution:")
    for class_id, count in sorted(class_counts.items()):
        name = class_names.get(class_id, f"Class {class_id}")
        pct = (count / total * 100) if total > 0 else 0
        print(f"   {name}: {count} ({pct:.1f}%)")
    print(f"   Total Annotations: {total}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Road Damage Dataset for YOLO")
    parser.add_argument("--source", type=str, required=True, help="Path to raw dataset folder")
    parser.add_argument("--dest", type=str, default="data/road_damage_yolo", help="Destination output folder")
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of dataset to use (0.0 to 1.0)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    source_path = Path(args.source)
    dest_path = Path(args.dest)

    if dest_path.exists():
        shutil.rmtree(dest_path)

    pairs = find_image_label_pairs(source_path)
    split_and_copy_dataset(pairs, dest_path, args.subset, args.train_ratio, args.val_ratio)
    create_data_yaml(dest_path, DEFAULT_CLASSES)
    analyze_dataset_distribution(dest_path / "train" / "labels", DEFAULT_CLASSES)


if __name__ == "__main__":
    main()