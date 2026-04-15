# train_wildlife.py — v2.1
#
# Builds two custom wildlife detection models for northern Idaho forest roads:
#   - yolo11s: full accuracy model for laptop (RTX 3050 Ti / 5070 Ti)
#   - yolo11n: lightweight model exported to ONNX for Raspberry Pi
#
# Pipeline:
#   1. Download merged Roboflow dataset (deer, elk, turkey, moose) via direct URL
#   2. Download wildlife subset from LILA.science (Caltech Camera Traps)
#   3. Convert LILA annotations from COCO to YOLO format
#   4. Merge all datasets into one unified training set
#   5. Train YOLOv11s (laptop/desktop model)
#   6. Train YOLOv11n and export to ONNX (Raspberry Pi model)
#
# Run on the desktop (RTX 5070 Ti) for fastest training (~30-45 min).
# Also works on the laptop (RTX 3050 Ti) — will use a smaller batch (~2-3 hrs).

import os
import sys
import json
import shutil
import random
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import torch

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------

# Roboflow direct download URL — merged dataset (deer, elk, turkey, moose)
# Get this from your Roboflow project: Export → Show download code → Raw URL
ROBOFLOW_DOWNLOAD_URL = 'https://app.roboflow.com/ds/bj2r5cgnUd?key=peifzDxJIS'

# Where to save everything
DATASET_DIR  = Path('datasets/wildlife')
ROBOFLOW_DIR = DATASET_DIR / 'roboflow'
LILA_DIR     = DATASET_DIR / 'lila'
MERGED_DIR   = DATASET_DIR / 'merged'

# LILA.science — northern Idaho relevant animals
LILA_MAX_IMAGES = 2000
LILA_CATEGORIES = [
    'deer', 'elk', 'moose', 'bear', 'turkey',
    'coyote', 'bighorn sheep', 'mountain lion', 'raccoon', 'fox'
]

# Training config
EPOCHS   = 100
IMG_SIZE = 640

# -----------------------------------------------------------------------
# Auto-detect batch size based on available VRAM
# -----------------------------------------------------------------------
def get_batch_size(model_size='s'):
    if not torch.cuda.is_available():
        print('No GPU detected — training on CPU (very slow, not recommended)')
        return 4
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    print(f'GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)')
    if vram_gb >= 12:
        return 32 if model_size == 's' else 64   # 5070 Ti
    elif vram_gb >= 6:
        return 16 if model_size == 's' else 32
    else:
        return 8  if model_size == 's' else 16   # 3050 Ti laptop

# -----------------------------------------------------------------------
# Phase 1 — Download Roboflow dataset via direct URL
# -----------------------------------------------------------------------
def download_roboflow():
    print('\n--- Phase 1: Downloading Roboflow dataset ---')

    out_dir  = ROBOFLOW_DIR / 'wildlife'
    zip_path = ROBOFLOW_DIR / 'wildlife.zip'

    # Skip if already downloaded
    if (out_dir / 'data.yaml').exists():
        print(f'  Already downloaded — skipping ({out_dir})')
        return [out_dir]

    ROBOFLOW_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Downloading from Roboflow...')
    r = requests.get(ROBOFLOW_DOWNLOAD_URL, stream=True)
    if r.status_code != 200:
        print(f'ERROR: Download failed — HTTP {r.status_code}')
        sys.exit(1)

    total = int(r.headers.get('content-length', 0))
    with open(zip_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc='Downloading') as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f'Extracting...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    zip_path.unlink()

    print(f'  Saved to: {out_dir}')
    return [out_dir]

# -----------------------------------------------------------------------
# Phase 2 — Download LILA.science subset (Caltech Camera Traps)
# -----------------------------------------------------------------------
def download_lila():
    print('\n--- Phase 2: Downloading LILA.science subset ---')

    METADATA_URL   = (
        'https://lilablobssc.blob.core.windows.net/'
        'caltech-unzipped/cct_images/eccv_18_all_images_sm.json.zip'
    )
    IMAGE_BASE_URL = 'https://lilablobssc.blob.core.windows.net/caltech-unzipped/cct_images/'

    metadata_zip  = LILA_DIR / 'metadata.json.zip'
    metadata_json = LILA_DIR / 'metadata.json'
    images_dir    = LILA_DIR / 'images'
    labels_dir    = LILA_DIR / 'labels'

    LILA_DIR.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

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
            name = zf.namelist()[0]
            zf.extract(name, LILA_DIR)
            (LILA_DIR / name).rename(metadata_json)
        metadata_zip.unlink()

    print('Parsing metadata...')
    with open(metadata_json) as f:
        data = json.load(f)

    cat_id_to_name  = {c['id']: c['name'].lower() for c in data['categories']}
    img_annotations = {}
    for ann in data['annotations']:
        img_annotations.setdefault(ann['image_id'], []).append(ann)
    img_info = {img['id']: img for img in data['images']}

    target_ids = {c['id'] for c in data['categories']
                  if c['name'].lower() in LILA_CATEGORIES}

    qualifying = [img_id for img_id, anns in img_annotations.items()
                  if any(a['category_id'] in target_ids for a in anns)]

    random.seed(42)
    selected = random.sample(qualifying, min(LILA_MAX_IMAGES, len(qualifying)))
    print(f'Found {len(qualifying)} qualifying images, downloading {len(selected)}...')

    lila_class_map = {cat_id: idx for idx, cat_id in enumerate(sorted(target_ids))}
    lila_names     = {idx: cat_id_to_name[cat_id] for cat_id, idx in lila_class_map.items()}

    downloaded = 0
    for img_id in tqdm(selected, desc='Downloading LILA images'):
        info      = img_info[img_id]
        img_url   = IMAGE_BASE_URL + info['file_name']
        img_path  = images_dir / Path(info['file_name']).name

        if not img_path.exists():
            try:
                r = requests.get(img_url, timeout=10)
                if r.status_code == 200:
                    img_path.write_bytes(r.content)
                else:
                    continue
            except Exception:
                continue

        img_w = info['width']
        img_h = info['height']
        yolo_lines = []
        for ann in img_annotations.get(img_id, []):
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
            (labels_dir / (img_path.stem + '.txt')).write_text('\n'.join(yolo_lines))
            downloaded += 1

    print(f'LILA download complete: {downloaded} images with labels')
    return images_dir, labels_dir, lila_names

# -----------------------------------------------------------------------
# Phase 3 — Merge all datasets
# -----------------------------------------------------------------------
def merge_datasets(roboflow_dirs, lila_images_dir, lila_labels_dir):
    print('\n--- Phase 3: Merging all datasets ---')

    for split in ('train', 'val'):
        (MERGED_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Copy all Roboflow datasets
    # Roboflow zips use 'train' and 'valid' (not 'val') — map both to 'train'/'val'
    split_map = {'train': 'train', 'valid': 'val', 'val': 'val'}
    for rf_dir in roboflow_dirs:
        for rf_split, merged_split in split_map.items():
            for sub in ('images', 'labels'):
                src = rf_dir / rf_split / sub
                if src.exists():
                    for f in src.iterdir():
                        # Prefix filename with dataset name to avoid collisions
                        dest = MERGED_DIR / sub / merged_split / f'{rf_dir.name}_{f.name}'
                        shutil.copy(f, dest)

    # Split LILA 85% train / 15% val
    lila_imgs = sorted(lila_images_dir.iterdir())
    random.seed(42)
    random.shuffle(lila_imgs)
    split_idx = int(len(lila_imgs) * 0.85)

    for i, img_path in enumerate(lila_imgs):
        label_path = lila_labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        split = 'train' if i < split_idx else 'val'
        shutil.copy(img_path,   MERGED_DIR / 'images' / split / f'lila_{img_path.name}')
        shutil.copy(label_path, MERGED_DIR / 'labels' / split / f'lila_{label_path.name}')

    train_count = len(list((MERGED_DIR / 'images' / 'train').iterdir()))
    val_count   = len(list((MERGED_DIR / 'images' / 'val').iterdir()))
    print(f'Merged dataset: {train_count} train / {val_count} val images')

# -----------------------------------------------------------------------
# Phase 4 — Write data.yaml
# -----------------------------------------------------------------------
def write_data_yaml(roboflow_dirs, lila_names):
    print('\n--- Phase 4: Writing data.yaml ---')
    import yaml

    all_names = []
    for rf_dir in roboflow_dirs:
        rf_yaml = rf_dir / 'data.yaml'
        if rf_yaml.exists():
            with open(rf_yaml) as f:
                rf_data = yaml.safe_load(f)
            for name in rf_data.get('names', []):
                if name.lower() not in [n.lower() for n in all_names]:
                    all_names.append(name)

    for name in lila_names.values():
        if name.lower() not in [n.lower() for n in all_names]:
            all_names.append(name)

    yaml_content = f"""# AutoWildLife dataset — merged Roboflow + LILA.science
# Target species: northern Idaho road hazard animals
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
# Phase 5 — Train YOLOv11s (laptop / desktop model)
# -----------------------------------------------------------------------
def train_full(yaml_path, batch_size):
    print('\n--- Phase 5: Training YOLOv11s (laptop/desktop model) ---')
    print(f'Epochs: {EPOCHS}  |  Batch: {batch_size}  |  Image size: {IMG_SIZE}')

    model = YOLO('yolo11s.pt')
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/wildlife',
        name='train_s',
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
    )
    return Path('runs/wildlife/train_s/weights/best.pt')

# -----------------------------------------------------------------------
# Phase 6 — Train YOLOv11n and export to ONNX (Raspberry Pi model)
# -----------------------------------------------------------------------
def train_rpi(yaml_path, batch_size):
    print('\n--- Phase 6: Training YOLOv11n + ONNX export (Raspberry Pi model) ---')
    print(f'Epochs: {EPOCHS}  |  Batch: {batch_size}  |  Image size: {IMG_SIZE}')

    model = YOLO('yolo11n.pt')
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/wildlife',
        name='train_n',
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
    )

    best_n = Path('runs/wildlife/train_n/weights/best.pt')
    print('\nExporting YOLOv11n to ONNX for Raspberry Pi...')
    rpi_model = YOLO(str(best_n))
    rpi_model.export(format='onnx', imgsz=IMG_SIZE, simplify=True)
    onnx_path = best_n.with_suffix('.onnx')
    print(f'ONNX model saved to: {onnx_path}')
    return best_n, onnx_path

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    print('=== AutoWildLife Training Pipeline v2.0 ===')
    print(f'Datasets    : {[p for _, p, _ in ROBOFLOW_DATASETS]}')
    print(f'Dataset dir : {DATASET_DIR.resolve()}')
    print(f'Output dir  : runs/wildlife/')

    roboflow_dirs              = download_roboflow()
    lila_images, lila_labels, lila_names = download_lila()
    merge_datasets(roboflow_dirs, lila_images, lila_labels)
    yaml_path                  = write_data_yaml(roboflow_dirs, lila_names)

    batch_s = get_batch_size('s')
    batch_n = get_batch_size('n')

    best_s            = train_full(yaml_path, batch_s)
    best_n, onnx_path = train_rpi(yaml_path, batch_n)

    print('\n=== Training complete ===')
    print(f'Laptop/desktop model : {best_s}')
    print(f'Raspberry Pi model   : {onnx_path}')
    print()
    print('To use in AutoWildLife.py:')
    print(f'  MODEL_PATH = "{best_s}"')
    print(f'  CUSTOM_MODEL_MODE = True')
    print()
    print('To use on Raspberry Pi:')
    print(f'  MODEL_PATH = "{onnx_path}"')
    print(f'  CUSTOM_MODEL_MODE = True')
