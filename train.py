"""
YOLOv8 Training Script
======================
Expected dataset layout:
    dataset/
        images/
            train/   *.jpg / *.png
            val/     *.jpg / *.png
        labels/
            train/   *.txt  (YOLO format: class cx cy w h, normalized)
            val/     *.txt
        data.yaml

data.yaml example:
    path: /absolute/path/to/dataset
    train: images/train
    val: images/val
    nc: 2
    names: ['cat', 'dog']
"""

from pathlib import Path
from ultralytics import YOLO

# --- Configuration ---
DATA_YAML = "dataset/data.yaml"   # path to your data.yaml
MODEL = "premadePT/yolov8s.pt"    # starting weights (downloads if not present)
EPOCHS = 100
BATCH = 32
IMG_SIZE = 640
DEVICE = 0                        # GPU index (0 = first GPU)
WORKERS = 8                       # dataloader workers
PROJECT = "runs/train"            # output root
NAME = "exp"                      # run sub-folder name (auto-incremented)

# --- Sanity check ---
if not Path(DATA_YAML).exists():
    raise FileNotFoundError(
        f"data.yaml not found at '{DATA_YAML}'.\n"
        "Create your dataset folder and data.yaml before training.\n"
        "See the docstring at the top of this file for the expected layout."
    )

# --- Train ---
model = YOLO(MODEL)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMG_SIZE,
    device=DEVICE,
    workers=WORKERS,
    project=PROJECT,
    name=NAME,
    # Useful defaults — tune as needed:
    patience=20,        # early stopping epochs
    save=True,          # save best.pt and last.pt
    save_period=-1,     # save checkpoint every N epochs (-1 = only best/last)
    cache=False,        # set True to cache images in RAM for speed
    amp=True,           # automatic mixed precision (FP16) — big speedup on RTX
    plots=True,         # save training curves / confusion matrix
    verbose=True,
)

print("\nTraining complete.")
print(f"Best weights : {results.save_dir}/weights/best.pt")
print(f"Last weights : {results.save_dir}/weights/last.pt")
