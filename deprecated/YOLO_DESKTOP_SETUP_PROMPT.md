I need you to set up a complete YOLOv8 training environment on this desktop (Ubuntu 24.04, RTX 5070 Ti).

Please do the following steps, running commands directly using your tools:

1. **Check hardware** — confirm the RTX 5070 Ti is detected and that the NVIDIA driver and nvidia-smi are working.

2. **Install CUDA Toolkit** via `sudo apt install nvidia-cuda-toolkit` if nvcc is not already present.

3. **Install Python dependencies** using `pip install --user`:
   - `ultralytics` (the official YOLOv8 package — includes training, inference, and the YOLO CLI)
   - `torch torchvision torchaudio` with CUDA support (use the correct pip install command from pytorch.org for CUDA 12.x)
   - `opencv-python`

4. **Verify the setup** by running a quick Python check:
   ```python
   import torch
   print(torch.cuda.is_available())         # should be True
   print(torch.cuda.get_device_name(0))     # should show RTX 5070 Ti
   from ultralytics import YOLO
   model = YOLO('yolov8s.pt')               # downloads the small model
   print(model.info())
   ```

5. **Run a test inference** on a sample image using the downloaded model to confirm everything works end to end.

6. **Create a `train.py` starter script** in the current directory that:
   - Loads YOLOv8s
   - Trains on a dataset folder structure (images/train, images/val, labels/train, labels/val)
   - Uses GPU (device=0)
   - Trains for 100 epochs with batch size 32
   - Saves results to a `runs/` folder

**Notes:**
- This machine is running Ubuntu 24.04 with an RTX 5070 Ti (16GB VRAM)
- Use `pip install --user` for all Python packages (no virtual environments)
- If the apt CUDA toolkit version is too old for PyTorch, install the PyTorch CUDA build directly via pip instead of building from source
- If nvidia-smi shows the driver is already installed, skip driver installation and go straight to CUDA toolkit
