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
uv pip install -r requirements.txt
# or
pip install -r requirements.txt
```


## 4. Run Training
```powershell
python train.py --epochs 50 --batch-size 8
```
