"""Train the KLA image-restoration model.

Expected dataset layout:
    data_dir/
        degraded/
        gt/

This is a compact reproducible scaffold. Replace dataset-specific assumptions
only if the supplied competition dataset uses different filenames/layout.
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class PairedImageDataset(Dataset):
    def __init__(self, degraded_dir, gt_dir):
        self.degraded = sorted(Path(degraded_dir).glob("*"))
        self.gt_dir = Path(gt_dir)
        if not self.degraded:
            raise RuntimeError(f"No degraded images found in {degraded_dir}")

    def __len__(self):
        return len(self.degraded)

    def __getitem__(self, i):
        p = self.degraded[i]
        gt = self.gt_dir / p.name
        if not gt.exists():
            raise FileNotFoundError(f"Missing ground truth for {p.name}")
        x = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        y = np.asarray(Image.open(gt).convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(x)[None], torch.from_numpy(y)[None]


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1)
        )
    def forward(self, x):
        return x + self.net(x)


class ResidualUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.e1 = nn.Sequential(ResidualBlock(base), ResidualBlock(base))
        self.d1 = nn.Conv2d(base, base*2, 3, stride=2, padding=1)
        self.e2 = nn.Sequential(ResidualBlock(base*2), ResidualBlock(base*2))
        self.d2 = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1)
        self.b = nn.Sequential(ResidualBlock(base*4), ResidualBlock(base*4))
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.c2 = nn.Conv2d(base*4, base*2, 3, padding=1)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.c1 = nn.Conv2d(base*2, base, 3, padding=1)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x):
        base = nn.functional.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        z = self.in_conv(base)
        e1 = self.e1(z)
        e2 = self.e2(self.d1(e1))
        b = self.b(self.d2(e2))
        z = self.u2(b)
        z = self.c2(torch.cat([z, e2], 1))
        z = self.u1(z)
        z = self.c1(torch.cat([z, e1], 1))
        return base + self.out(z)


def edge_loss(pred, target):
    px = pred[..., :, 1:] - pred[..., :, :-1]
    py = pred[..., 1:, :] - pred[..., :-1, :]
    tx = target[..., :, 1:] - target[..., :, :-1]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return (px - tx).abs().mean() + (py - ty).abs().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PairedImageDataset(Path(args.data_dir)/"degraded", Path(args.data_dir)/"gt")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = ResidualUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = mse(pred, y) + 0.05 * edge_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * x.size(0)
        avg = total / len(ds)
        print(f"Epoch [{epoch+1}/{args.epochs}] Train: {avg:.6f}")
        state = {"epoch": epoch+1, "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "train_loss": avg}
        torch.save(state, out / f"epoch_{epoch+1:02d}.pth")
        if avg < best:
            best = avg
            torch.save(state, out / "final_model.pth")


if __name__ == "__main__":
    main()
