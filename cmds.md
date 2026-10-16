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


## 4. Run Training (Accelerated Single-GPU & Anti-Overfitting Pipeline)
Default configuration now includes:
- **Spatial Dropout (`--dropout 0.3`)**: Applied across encoder stages (conv4 & conv5) as well as fusion and decoder blocks.
- **Advanced Augmentation**: CoarseDropout (Cutout with `fill_mask=255` so masked regions are ignored in loss), expanded Affine rotations/scales, and brightness/contrast jitter.
- **Decoupled Weight Decay (`--weight-decay 0.02`)**: Regularizes 2D convolution weights while leaving BatchNorm and bias parameters unpenalized.
- **Label Smoothing (`--label-smoothing 0.1`)**: Prevents overconfident logit predictions.
- **6-Column Overlays for Train & Val**:
  Renders `[image, thermal_image, predicted_mask, pred_overlay, true_label, true_overlay]` for both training batches and validation batches after every epoch (`epoch_XXX_train_predictions.png`, `epoch_XXX_val_predictions.png`, and combined `epoch_XXX_predictions.png`).
- **Single-GPU Acceleration (`--device cuda:0`)**:
  - `torch.compile` kernel compilation enabled by default (`--compile`)
  - FP16 Automatic Mixed Precision enabled by default (`--amp`)
  - cuDNN auto-tuner benchmark enabled
  - Multi-worker DataLoader with `persistent_workers=True` and page-locked pinned memory.
- **Early Stopping (`--patience 10`)**: Halts training if validation mIoU fails to improve after 10 epochs.

```bash
python train.py --epochs 50 --batch-size 8
```

### Custom Anti-Overfitting & Acceleration Flags:
```bash
python train.py --epochs 50 --batch-size 8 --patience 10 --label-smoothing 0.1 --weight-decay 0.02 --dropout 0.3 --compile --amp --device cuda:0
```

To resume training from an existing checkpoint with fine-tuning / anti-overfitting settings:
```bash
python train.py --resume checkpoints/best_checkpoint.pt --epochs 60 --patience 10 --lr 1e-4 --lr-mode restart
```
