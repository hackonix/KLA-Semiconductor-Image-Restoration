import sys
from pathlib import Path
import numpy as np
import torch

from model import ResidualUNet


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet().to(device)

    weights = Path(__file__).parent / "models" / "final_model.pth"
    checkpoint = torch.load(weights, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    files = sorted(input_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    print(f"Device: {device}")
    print(f"Images: {len(files)}")

    with torch.inference_mode():
        for i, path in enumerate(files, 1):
            arr = np.load(path, allow_pickle=False).astype(np.float32)
            arr = np.squeeze(arr)

            if arr.shape != (128, 128):
                raise ValueError(f"{path.name}: expected (128,128), got {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{path.name}: input contains NaN or Inf")

            x = torch.from_numpy(arr)[None, None].to(device)
            pred = model(x)
            pred = pred[0, 0].detach().cpu().numpy().astype(np.float32)
            pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
            pred = np.clip(pred, 0.0, 1.0)
            np.save(output_dir / path.name, pred)

            if i % 25 == 0 or i == len(files):
                print(f"Processed {i}/{len(files)}")

    print("DONE")


if __name__ == "__main__":
    main()
