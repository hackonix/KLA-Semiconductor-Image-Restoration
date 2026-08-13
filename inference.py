import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from train import ResidualUNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", default="weights/final_model.pth")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet().to(device)
    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    with torch.inference_mode():
        for p in files:
            arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            x = torch.from_numpy(arr)[None, None].to(device)
            pred = model(x).clamp(0, 1)[0, 0].cpu().numpy()
            Image.fromarray((pred * 255.0).round().astype(np.uint8)).save(out_dir / f"{p.stem}.png")
            print(f"{p.name} -> {p.stem}.png")


if __name__ == "__main__":
    main()
