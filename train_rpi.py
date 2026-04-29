# train_rpi.py
#
# Trains a lightweight YOLOv11n wildlife detection model and exports it to
# INT8-quantized ONNX format for use on a Raspberry Pi.
#
# Resolution: 864×480 (16:9, both dims divisible by 32 as YOLO requires)
# Quantization: INT8 dynamic — ~2× faster on ARM CPU vs FP32, minimal accuracy loss
#
# Run AFTER train_wildlife.py has built the merged dataset.
# The merged dataset at datasets/wildlife/merged/ must already exist.
#
# Output:
#   runs/wildlife/train_n/weights/best.pt        — PyTorch weights (FP32)
#   runs/wildlife/train_n/weights/best.onnx      — ONNX FP32
#   runs/wildlife/train_n/weights/best_int8.onnx — ONNX INT8 (copy this to the Pi)
#
# Usage:
#   python3 train_rpi.py

from pathlib import Path
from ultralytics import YOLO
import torch

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
MERGED_DIR = Path('datasets/wildlife/merged')
DATA_YAML  = MERGED_DIR / 'data.yaml'

EPOCHS = 100

# 16:9 at ~480p — nearest dimensions divisible by 32
# True 480p 16:9 = 854×480, but 854 is not divisible by 32
# 864×480 = 27×32 × 15×32 — ratio 1.800 vs true 16:9 ratio 1.778 (~1.3% off)
IMG_W = 864
IMG_H = 480

# -----------------------------------------------------------------------
# Auto-detect batch size based on available VRAM
# -----------------------------------------------------------------------
def get_batch_size():
    if not torch.cuda.is_available():
        print('No GPU detected — training on CPU (very slow)')
        return 8
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    print(f'GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)')
    if vram_gb >= 12:
        return 32   # RTX 5070 Ti — capped at 32 for WSL2 VRAM stability
    elif vram_gb >= 6:
        return 32
    else:
        return 16   # RTX 3050 Ti laptop

# -----------------------------------------------------------------------
# Train YOLOv11n at 864×480
# -----------------------------------------------------------------------
def train():
    batch = get_batch_size()
    print(f'\nTraining YOLOv11n — Epochs: {EPOCHS}  Batch: {batch}  Size: {IMG_W}×{IMG_H}')

    model = YOLO('yolo11n.pt')
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=[IMG_H, IMG_W],   # [height, width]
        batch=batch,
        device=0 if torch.cuda.is_available() else 'cpu',
        project=str(Path('runs/wildlife').resolve()),
        name='train_n',
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        workers=4,      # reduced from 8 — limits CPU/RAM pressure on WSL2
    )
    return Path('runs/wildlife/train_n/weights/best.pt').resolve()

# -----------------------------------------------------------------------
# Export to FP32 ONNX, then quantize to INT8
# -----------------------------------------------------------------------
def export_onnx_int8(best_pt):
    print(f'\nExporting to ONNX (FP32): {best_pt}')
    model = YOLO(str(best_pt))
    model.export(format='onnx', imgsz=[IMG_H, IMG_W], simplify=True)
    fp32_path = best_pt.with_suffix('.onnx')
    print(f'FP32 ONNX saved: {fp32_path}')

    print('\nQuantizing to INT8 (dynamic)...')
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8_path = best_pt.parent / 'best_int8.onnx'
    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f'INT8 ONNX saved: {int8_path}')
    return fp32_path, int8_path

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    print('=== RPi ONNX Training Pipeline ===')
    print(f'Resolution : {IMG_W}×{IMG_H} (16:9)')
    print(f'Quantization: INT8 dynamic')

    if not DATA_YAML.exists():
        print(f'\nERROR: {DATA_YAML} not found.')
        print('Run train_wildlife.py first to build the merged dataset.')
        raise SystemExit(1)

    # Check onnxruntime quantization is available
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print('\nERROR: onnxruntime quantization tools not found.')
        print('Install with: pip install --user --break-system-packages onnxruntime')
        raise SystemExit(1)

    best_pt             = train()
    fp32_path, int8_path = export_onnx_int8(best_pt)

    print('\n=== Done ===')
    print(f'PyTorch (FP32) : {best_pt}')
    print(f'ONNX FP32      : {fp32_path}')
    print(f'ONNX INT8      : {int8_path}')
    print()
    print('Copy the INT8 model to the Raspberry Pi:')
    print(f'  scp {int8_path} pi@raspberrypi:~/autowildlife/')
    print()
    print('On the Pi, set in AutoWildLife.py:')
    print(f'  MODEL_PATH = "autowildlife/{int8_path.name}"')
    print(f'  CUSTOM_MODEL_MODE = True')
