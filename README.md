# KLA Semiconductor Image Restoration

AI-based restoration of degraded grayscale semiconductor inspection images.

## Final Run 4 model

- Architecture: Residual U-Net
- Input: 128x128 grayscale `.npy`
- Output: 256x256 restored `.npy`
- Best checkpoint: epoch 11
- Validation PSNR: **28.3825 dB**
- Validation SSIM: **0.7570**

## Problem

Restore paired degraded low-resolution semiconductor inspection images toward clean high-resolution ground truth. The project investigates Gaussian noise, speckle noise, spatial downsampling, and combinations of these degradations.

## Architecture

```text
128x128 degraded image
        |
        v
2x bicubic baseline
        |
        v
Residual U-Net
  encoder / bottleneck / decoder
        |
        v
learned residual
        |
        v
baseline + residual
        |
        v
256x256 restored image
```

## Data format

The evaluator expects `.npy` grayscale arrays:

- Input shape: `(128, 128)`
- Output shape: `(256, 256)`
- Input values are kept as provided; the evaluator does not blindly normalize or clip the input before restoration.

Do **not** commit the competition dataset to this repository.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Evaluation / inference

```bash
python evaluation.py --input_dir /path/to/test_npy --output_dir /path/to/restored_outputs --weights weights/final_model.pth
```

Equivalent entry point:

```bash
python inference.py --input_dir /path/to/test_npy --output_dir /path/to/restored_outputs --weights weights/final_model.pth
```

No source-code editing is required.

## Verification completed

The verified Run 4 pipeline was executed on **400 real 128x128 test inputs** with a Tesla T4. All 400 outputs were generated successfully as 256x256 `.npy` arrays.

Additional single-sample robustness experiments were performed for Gaussian noise, speckle noise, combined degradations, and all six permutations of Gaussian, speckle, and downsampling. These are robustness experiments, not official test-set scores.

## Training

The original training work used paired `.npy` data with 2560 training samples and 640 validation samples. The competition dataset should not be committed to the repository.

## Weights

Place the verified Run 4 checkpoint at:

```text
weights/final_model.pth
```

The checkpoint used during development was verified to load into the exact `model.py` architecture and produce `(1, 1, 256, 256)` from `(1, 1, 128, 128)`.
