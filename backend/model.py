"""
STCrackNet — the model from your Colab notebook.
2.27M params · F1: 0.7199 · IoU: 0.5624
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class STCrackNet(nn.Module):
    """Dual-branch RGB + Sobel-edge encoder with channel attention fusion."""

    def __init__(self):
        super().__init__()
        self.rgb_e1 = ConvBlock(3, 64)
        self.rgb_e2 = ConvBlock(64, 128)
        self.rgb_e3 = ConvBlock(128, 256)

        self.edge_e1 = ConvBlock(1, 32)
        self.edge_e2 = ConvBlock(32, 64)
        self.edge_e3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.ca_rgb = ChannelAttention(256)
        self.ca_edge = ChannelAttention(128)

        self.fusion = nn.Sequential(
            nn.Conv2d(384, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.out_conv = nn.Conv2d(64, 1, 1)

    def _sobel(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                          dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        ky = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                          dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        return torch.sqrt(
            F.conv2d(gray, kx, padding=1) ** 2 +
            F.conv2d(gray, ky, padding=1) ** 2 + 1e-6
        )

    def forward(self, x):
        edge = self._sobel(x)

        r1 = self.rgb_e1(x);              e1 = self.edge_e1(edge)
        r2 = self.rgb_e2(self.pool(r1));  e2 = self.edge_e2(self.pool(e1))
        r3 = self.rgb_e3(self.pool(r2));  e3 = self.edge_e3(self.pool(e2))

        fused = self.fusion(torch.cat([self.ca_rgb(r3), self.ca_edge(e3)], dim=1))

        d3 = self.dec3(torch.cat([self.up3(fused), r2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), r1], dim=1))

        return self.out_conv(d2)


def load_stcracknet(weights_path: str, device: str = "cpu") -> STCrackNet:
    """Load a trained STCrackNet checkpoint."""
    model = STCrackNet().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
