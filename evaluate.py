import argparse
from pathlib import Path
import time
import torch
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from train import ResidualUNet


def read_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", default="weights/final_model.pth")
    ap.add_argument("--gt_dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet().to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted([p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    psnr_values, ssim_values, times = [], [], []

    with torch.inference_mode():
        for p in inputs:
            arr = read_gray(p)
            x = torch.from_numpy(arr)[None, None].to(device)
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            pred_np = pred[0, 0].clamp(0, 1).cpu().numpy()
            Image.fromarray((pred_np * 255).round().astype(np.uint8)).save(out_dir / f"{p.stem}.png")

            if args.gt_dir:
                gt_path = Path(args.gt_dir) / p.name
                if gt_path.exists():
                    gt = read_gray(gt_path)
                    psnr_values.append(peak_signal_noise_ratio(gt, pred_np, data_range=1.0))
                    ssim_values.append(structural_similarity(gt, pred_np, data_range=1.0))

    print(f"Images: {len(inputs)}")
    if times:
        print(f"Mean inference time: {np.mean(times):.6f} s/image")
    if psnr_values:
        print(f"PSNR: {np.mean(psnr_values):.6f} dB")
        print(f"SSIM: {np.mean(ssim_values):.6f}")


if __name__ == "__main__":
    main()
