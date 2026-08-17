import argparse
from pathlib import Path
import numpy as np
import torch
from model import ResidualUNet


def main():
    ap = argparse.ArgumentParser(description="Run Run-4 restoration on 128x128 .npy images.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", default="weights/final_model.pth")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet().to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            x = torch.from_numpy(arr)[None, None].to(device)
            pred = model(x)[0, 0].cpu().numpy().astype(np.float32)
            np.save(output_dir / path.name, pred)
            if i % 25 == 0 or i == len(files):
                print(f"Processed {i}/{len(files)}")

    print("DONE")


if __name__ == "__main__":
    main()
