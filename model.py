import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return residual + out


class ResidualUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.input_conv = nn.Conv2d(1, base, 3, padding=1)
        self.enc1 = nn.Sequential(
            ResidualBlock(base),
            ResidualBlock(base),
        )
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            ResidualBlock(base * 2),
            ResidualBlock(base * 2),
        )
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base * 4),
            ResidualBlock(base * 4),
            ResidualBlock(base * 4),
        )
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(base * 2),
        )
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(base),
        )
        self.output_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x):
        baseline = F.interpolate(
            x, size=(256, 256), mode="bicubic", align_corners=False
        )
        x1 = self.enc1(self.input_conv(baseline))
        x2 = self.enc2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))
        y2 = self.dec2(torch.cat([self.up2(x3), x2], dim=1))
        y1 = self.dec1(torch.cat([self.up1(y2), x1], dim=1))
        residual = self.output_conv(y1)
        return torch.clamp(baseline + residual, 0.0, 1.0)
