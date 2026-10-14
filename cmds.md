# Project Commands & GPU Setup Guide

## 1. Activate Environment (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Verify GPU Detection
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

## 3. Install Dependencies with CUDA
```powershell
python -m uv pip install --reinstall-package torch torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m uv pip install -r requirements.txt
```


## 4. Run Training (with Anti-Overfitting Features)
Default configuration now includes:
- **Spatial Dropout (`--dropout 0.2`)**: Applied across fusion and decoder blocks.
- **Label Smoothing (`--label-smoothing 0.1`)**: Prevents overconfident logit predictions.
- **Early Stopping (`--patience 10`)**: Monitors validation mIoU with patience 10 to halt before overfitting.
- **Increased Weight Decay (`--weight-decay 0.01`)**: Regularizes model weights with AdamW.
- **Enhanced Data Augmentation**: Flips, random contrast/brightness, affine scaling/rotation, and Gaussian blur.

```powershell
python train.py --epochs 50 --batch-size 8
```

### Custom Anti-Overfitting Flags:
```powershell
python train.py --epochs 50 --batch-size 8 --patience 10 --label-smoothing 0.1 --weight-decay 0.01 --dropout 0.2
```

To resume training from an existing checkpoint with early stopping:
```powershell
python train.py --resume checkpoints/last_checkpoint.pt --epochs 60 --patience 10
```
