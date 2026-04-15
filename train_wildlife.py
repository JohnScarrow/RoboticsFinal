# train_wildlife.py — v1.0
#
# Builds a custom YOLOv8s wildlife detection model for forest road driving.
#
# Pipeline:
#   1. Download a road/animal dataset from Roboflow Universe
#   2. Download a wildlife subset from LILA.science (Caltech Camera Traps)
#   3. Convert LILA annotations from COCO format to YOLO format
#   4. Merge both datasets into one unified training set
#   5. Train YOLOv8s — auto-detects GPU and sets batch size accordingly
#   6. Validate and report final metrics
#
# Setup required before running:
#   - Create a free account at roboflow.com
#   - Get your API key from roboflow.com/settings/api
#   - Find a road/wildlife dataset on universe.roboflow.com
#     (search: "deer road", "wildlife vehicle", "animal detection road")
#   - Copy your API key and dataset info into the CONFIG section below
#     OR create a .env file in this folder with:
#       ROBOFLOW_API_KEY=your_key_here
#
# Run on the desktop (RTX 5070 Ti) for fastest training (~30-45 min).
# Also works on the laptop (RTX 3050 Ti) — will use a smaller batch (~2-3 hrs).

import os
import sys
import json
import shutil
import random
import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
import torch

# -----------------------------------------------------------------------
# CONFIG — fill these in before running
# -----------------------------------------------------------------------
load_dotenv()  # loads ROBOFLOW_API_KEY from .env file if present

# Your Roboflow API key — get it at roboflow.com/settings/api
# Leave as None to be prompted at runtime, or set ROBOFLOW_API_KEY in .env
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY', None)

# Roboflow dataset to download — find one at universe.roboflow.com
# Fill in after choosing a dataset (visible in the dataset's download page)
ROBOFLOW_WORKSPACE = 'autowildlife'
ROBOFLOW_PROJECT   = 'deer-hqp4i-zdzgs'
ROBOFLOW_VERSION   = 1

# Where to save everything
DATASET_DIR  = Path('datasets/wildlife')
ROBOFLOW_DIR = DATASET_DIR / 'roboflow'
LILA_DIR     = DATASET_DIR / 'lila'
MERGED_DIR   = DATASET_DIR / 'merged'

# How many images to pull from LILA.science
# More = better accuracy, longer download. 2000 is a good starting point.
LILA_MAX_IMAGES = 2000

# LILA.science animal categories to download.
# These are Caltech Camera Traps categories — common forest road animals.
LILA_CATEGORIES = [
    'deer', 'coyote', 'raccoon', 'rabbit', 'squirrel',
    'skunk', 'fox', 'bobcat', 'bear', 'mountain lion'
]

# Training config
EPOCHS   = 100
IMG_SIZE = 640

# -----------------------------------------------------------------------
# Auto-detect batch size based on available VRAM
# -----------------------------------------------------------------------
def get_batch_size():
    if not torch.cuda.is_available():
        print('No GPU detected — training on CPU (very slow, not recommended)')
        return 4
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    print(f'GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)')
    if vram_gb >= 12:
        return 32   # 5070 Ti or similar
    elif vram_gb >= 6:
        return 16
    else:
        return 8    # 3050 Ti laptop

# -----------------------------------------------------------------------
# Phase 1 — Download Roboflow dataset
# -----------------------------------------------------------------------
def download_roboflow():
    print('\n--- Phase 1: Downloading Roboflow dataset ---')

    api_key = ROBOFLOW_API_KEY
    if not api_key:
        api_key = input('Enter your Roboflow API key: ').strip()
    if not api_key:
        print('ERROR: Roboflow API key required. Get one at roboflow.com/settings/api')
        sys.exit(1)

    if ROBOFLOW_WORKSPACE == 'your-workspace':
        print('ERROR: Fill in ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, and ROBOFLOW_VERSION')
        print('       Find these on your dataset page at universe.roboflow.com')
        sys.exit(1)

    print(f'Downloading {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}...')
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download('yolov8', location=str(ROBOFLOW_DIR))

    print(f'Roboflow dataset saved to: {ROBOFLOW_DIR}')
    return ROBOFLOW_DIR

# -----------------------------------------------------------------------
# Phase 2 — Download LILA.science subset (Caltech Camera Traps)
# -----------------------------------------------------------------------
def download_lila():
    print('\n--- Phase 2: Downloading LILA.science subset ---')

    # Caltech Camera Traps metadata JSON (public, no auth required)
    METADATA_URL = (
        'https://lilablobssc.blob.core.windows.net/'
        'caltech-unzipped/cct_images/eccv_18_all_images_sm.json.zip'
    )
    # Base URL for images
    IMAGE_BASE_URL = 'https://lilablobssc.blob.core.windows.net/caltech-unzipped/cct_images/'

    metadata_zip = LILA_DIR / 'metadata.json.zip'
    metadata_json = LILA_DIR / 'metadata.json'
    images_dir = LILA_DIR / 'images'
    labels_dir = LILA_DIR / 'labels'

    LILA_DIR.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Download metadata if not already cached
    if not metadata_json.exists():
        print('Downloading Caltech Camera Traps metadata...')
        r = requests.get(METADATA_URL, stream=True)
        total = int(r.headers.get('content-length', 0))
        with open(metadata_zip, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        import zipfile
        with zipfile.ZipFile(metadata_zip, 'r') as zf:
            # The zip contains one JSON file — extract and rename it
            name = zf.namelist()[0]
            zf.extract(name, LILA_DIR)
            (LILA_DIR / name).rename(metadata_json)
        metadata_zip.unlink()

    print('Parsing metadata...')
    with open(metadata_json) as f:
        data = json.load(f)

    # Build lookup: category_id -> category_name
    cat_id_to_name = {c['id']: c['name'].lower() for c in data['categories']}

    # Build lookup: image_id -> annotation list
    img_annotations = {}
    for ann in data['annotations']:
        img_annotations.setdefault(ann['image_id'], []).append(ann)

    # Build lookup: image_id -> image info
    img_info = {img['id']: img for img in data['images']}

    # Filter to images that contain at least one of our target categories
    target_ids = {c['id'] for c in data['categories']
                  if c['name'].lower() in LILA_CATEGORIES}

    qualifying = []
    for img_id, anns in img_annotations.items():
        if any(a['category_id'] in target_ids for a in anns):
            qualifying.append(img_id)

    # Sample up to LILA_MAX_IMAGES
    random.seed(42)
    selected = random.sample(qualifying, min(LILA_MAX_IMAGES, len(qualifying)))
    print(f'Found {len(qualifying)} qualifying images, downloading {len(selected)}...')

    # Build a consecutive class map from LILA categories to YOLO class IDs
    # (YOLO needs 0-indexed integers)
    lila_class_map = {cat_id: idx for idx, cat_id in enumerate(sorted(target_ids))}
    lila_names = {idx: cat_id_to_name[cat_id]
                  for cat_id, idx in lila_class_map.items()}

    downloaded = 0
    for img_id in tqdm(selected, desc='Downloading LILA images'):
        info = img_info[img_id]
        img_url = IMAGE_BASE_URL + info['file_name']
        img_path = images_dir / Path(info['file_name']).name

        # Skip if already downloaded
        if not img_path.exists():
            try:
                r = requests.get(img_url, timeout=10)
                if r.status_code == 200:
                    img_path.write_bytes(r.content)
                else:
                    continue
            except Exception:
                continue

        # Convert COCO bbox [x, y, width, height] to YOLO format
        # YOLO: [class_id, cx_norm, cy_norm, w_norm, h_norm]
        img_w = info['width']
        img_h = info['height']
        anns = img_annotations.get(img_id, [])

        yolo_lines = []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in lila_class_map:
                continue
            class_idx = lila_class_map[cat_id]
            x, y, bw, bh = ann['bbox']
            cx = (x + bw / 2) / img_w
            cy = (y + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h
            yolo_lines.append(f'{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')

        if yolo_lines:
            label_path = labels_dir / (img_path.stem + '.txt')
            label_path.write_text('\n'.join(yolo_lines))
            downloaded += 1

    print(f'LILA download complete: {downloaded} images with labels')
    return images_dir, labels_dir, lila_names

# -----------------------------------------------------------------------
# Phase 3 — Merge Roboflow + LILA into one dataset
# -----------------------------------------------------------------------
def merge_datasets(roboflow_dir, lila_images_dir, lila_labels_dir):
    print('\n--- Phase 3: Merging datasets ---')

    for split in ('train', 'val'):
        (MERGED_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Copy Roboflow images and labels (already split into train/val)
    for split in ('train', 'val'):
        rf_imgs = roboflow_dir / split / 'images'
        rf_lbls = roboflow_dir / split / 'labels'
        if rf_imgs.exists():
            for f in rf_imgs.iterdir():
                shutil.copy(f, MERGED_DIR / 'images' / split / f.name)
        if rf_lbls.exists():
            for f in rf_lbls.iterdir():
                shutil.copy(f, MERGED_DIR / 'labels' / split / f.name)

    # Split LILA images 85% train / 15% val and copy
    lila_imgs = sorted(lila_images_dir.iterdir())
    random.seed(42)
    random.shuffle(lila_imgs)
    split_idx = int(len(lila_imgs) * 0.85)

    for i, img_path in enumerate(lila_imgs):
        label_path = lila_labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        split = 'train' if i < split_idx else 'val'
        shutil.copy(img_path,   MERGED_DIR / 'images' / split / img_path.name)
        shutil.copy(label_path, MERGED_DIR / 'labels' / split / label_path.name)

    train_count = len(list((MERGED_DIR / 'images' / 'train').iterdir()))
    val_count   = len(list((MERGED_DIR / 'images' / 'val').iterdir()))
    print(f'Merged dataset: {train_count} train / {val_count} val images')
    return train_count, val_count

# -----------------------------------------------------------------------
# Phase 4 — Write data.yaml
# -----------------------------------------------------------------------
def write_data_yaml(roboflow_dir, lila_names):
    print('\n--- Phase 4: Writing data.yaml ---')

    # Read class names from the Roboflow data.yaml
    rf_yaml = roboflow_dir / 'data.yaml'
    rf_names = []
    if rf_yaml.exists():
        import yaml
        with open(rf_yaml) as f:
            rf_data = yaml.safe_load(f)
        rf_names = rf_data.get('names', [])

    # Combine Roboflow class names + LILA class names (deduplicated)
    all_names = list(rf_names)
    for name in lila_names.values():
        if name not in all_names:
            all_names.append(name)

    yaml_content = f"""# AutoWildLife dataset — merged Roboflow + LILA.science
path: {MERGED_DIR.resolve()}
train: images/train
val:   images/val

nc: {len(all_names)}
names: {all_names}
"""
    yaml_path = MERGED_DIR / 'data.yaml'
    yaml_path.write_text(yaml_content)
    print(f'data.yaml written: {len(all_names)} classes — {all_names}')
    return yaml_path

# -----------------------------------------------------------------------
# Phase 5 — Train YOLOv8s
# -----------------------------------------------------------------------
def train(yaml_path, batch_size):
    print('\n--- Phase 5: Training YOLOv8s ---')
    print(f'Epochs: {EPOCHS}  |  Batch: {batch_size}  |  Image size: {IMG_SIZE}')

    model = YOLO('yolo11s.pt')  # start from pretrained weights (transfer learning)
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/wildlife',
        name='train',
        exist_ok=True,
        patience=20,        # stop early if no improvement for 20 epochs
        save=True,
        plots=True,         # save training curves to runs/wildlife/train/
    )
    return results

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    print('=== AutoWildLife Training Pipeline ===')
    print(f'Dataset dir : {DATASET_DIR.resolve()}')
    print(f'Output dir  : runs/wildlife/train/')

    batch_size = get_batch_size()

    roboflow_dir              = download_roboflow()
    lila_images, lila_labels, lila_names = download_lila()
    merge_datasets(roboflow_dir, lila_images, lila_labels)
    yaml_path                 = write_data_yaml(roboflow_dir, lila_names)
    results                   = train(yaml_path, batch_size)

    best_model = Path('runs/wildlife/train/weights/best.pt')
    print('\n=== Training complete ===')
    print(f'Best model saved to: {best_model}')
    print(f'To use in AutoWildLife.py, set:')
    print(f'  MODEL_PATH = "{best_model}"')
    print(f'  CUSTOM_MODEL_MODE = True')
