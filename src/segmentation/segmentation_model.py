import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.middle = DoubleConv(32, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = DoubleConv(32, 16)

        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        m = self.middle(p2)

        u1 = self.up1(m)
        if u1.shape[-2:] != d2.shape[-2:]:
            u1 = nn.functional.interpolate(
                u1,
                size=d2.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.dec1(u1)

        u2 = self.up2(u1)
        if u2.shape[-2:] != d1.shape[-2:]:
            u2 = nn.functional.interpolate(
                u2,
                size=d1.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.dec2(u2)

        return torch.sigmoid(self.out(u2))