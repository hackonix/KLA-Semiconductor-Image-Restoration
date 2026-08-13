# KLA Semiconductor Image Restoration

AI-based restoration of degraded grayscale semiconductor inspection images.

## Problem

Restore paired degraded low-resolution images to clean full-resolution ground truth images. The target degradation space includes speckle noise, Gaussian noise/blur-like degradation, spatial downsampling, and combinations in varying order.

## Current experimental result

Strongest measured validation result so far: **Edge-aware Residual U-Net — 27.946 dB PSNR, 0.74879 SSIM**. These are development/validation results, not final KLA test-set scores.

## Architecture

```text
Degraded LR image
       |
       v
2x bicubic reconstruction
       |
       v
Residual U-Net encoder/decoder
       |
       v
Learned residual
       |
       +----> add to bicubic reconstruction
                    |
                    v
              Restored HR image
```

Restoration formulation: `output = bicubic(input) + learned_residual`.

## Data

Images are grayscale. Ground truth images are 256x256 or 512x512; degraded inputs can be 128x128 or 256x256. Degraded intensities can lie outside `[0,1]`, so the implementation must not blindly clip the input before restoration.

Do **not** commit the competition dataset to this repository.

## Repository structure

```text
├── README.md
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
├── weights/
│   └── final_model.pth        # add after final training
└── restored_outputs/
    └── README.md              # add actual outputs after inference
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inference

```bash
python inference.py --input_dir /path/to/test_images --output_dir /path/to/restored_outputs --weights weights/final_model.pth
```

No source-code editing should be required by the evaluator.

## Evaluation

```bash
python evaluate.py --input_dir /path/to/test_images --output_dir /path/to/restored_outputs --weights weights/final_model.pth
```

When paired ground truth is available, the evaluator reports PSNR/SSIM. LPIPS and final benchmark timing must be added before submission.

## Training

```bash
python train.py --data_dir /path/to/paired_dataset --output_dir checkpoints
```

The intended final training uses paired degraded/ground-truth data, residual reconstruction, and edge-aware loss.

## Submission status

The final trained Edge-aware Residual U-Net weights are **not yet persisted in this repository**. The previous Colab runtime held the trained weights only in RAM and was disconnected. The Drive file `best_model.pth` was verified to contain the earlier Simple CNN (`conv1`, `conv2`, `conv3`), not the U-Net.

Therefore `weights/final_model.pth` must only be added after the final U-Net is retrained and saved. This prevents submitting an incorrect checkpoint.

## Final checklist

- [ ] Train final U-Net and save `weights/final_model.pth`
- [ ] Verify checkpoint loads in a fresh process
- [ ] Run inference without source edits
- [ ] Add actual restored test outputs
- [ ] Measure PSNR, SSIM and LPIPS
- [ ] Measure inference time per image
- [ ] Test on a clean machine
- [ ] Freeze exact dependencies in `requirements.txt`
