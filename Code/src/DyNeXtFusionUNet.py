import torch
import torch.nn as nn
import torch.nn.functional as F
from .Dynamic_upsample import DySample
from .CGAF import CGAFusion
from .CMUNeXtBlock import CMUNeXtBlock
# ----------------------
# DoubleConv
# ----------------------
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# ----------------------
# Down
# ----------------------
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class DownCMU(nn.Sequential):
    def __init__(self, in_channels, out_channels, k=7, depth=1, expand_ratio=4, act="gelu", norm="bn"):
        super().__init__(
            nn.MaxPool2d(2, stride=2),
            CMUNeXtBlock(in_channels, out_channels, k=k, depth=depth,
                         expand_ratio=expand_ratio, act=act, norm=norm)
        )

# ----------------------
# Up
# ----------------------
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        up_in_c = in_channels // 2

        def pick_groups(c):
            for g in (8, 4, 2, 1):
                if c % g == 0:
                    return g
            return 1
        g = pick_groups(up_in_c)

        self.up = DySample(in_channels=up_in_c, scale=2, style='lp', groups=g)

        self.align = None

        self.fusion = CGAFusion(up_in_c)

        self.conv = DoubleConv(up_in_c, out_channels, up_in_c)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_x or diff_y:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        if x2.size(1) != x1.size(1):
            if (self.align is None) or (self.align.in_channels != x2.size(1)) or (self.align.out_channels != x1.size(1)):
                self.align = nn.Conv2d(x2.size(1), x1.size(1), kernel_size=1).to(x2.device)
            x2 = self.align(x2)

        return self.conv(self.fusion(x1, x2))

# ----------------------
# OutConv
# ----------------------
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

# ----------------------
# DyNeXtFusion-UNet
# ----------------------
class DyNeXtFusionUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 base_c: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        factor = 2


        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = DownCMU(base_c, base_c * 2, k=7, depth=1)
        self.down2 = DownCMU(base_c * 2, base_c * 4, k=7, depth=1)
        self.down3 = DownCMU(base_c * 4, base_c * 8, k=7, depth=1)
        self.down4 = DownCMU(base_c * 8, base_c * 16 // factor, k=7, depth=1)


        self.up1 = Up(base_c * 16, base_c * 8 // factor)
        self.up2 = Up(base_c * 8, base_c * 4 // factor)
        self.up3 = Up(base_c * 4, base_c * 2 // factor)
        self.up4 = Up(base_c * 2, base_c)


        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.out_conv(x)
        return {"out": logits}
