import torch
from torch import nn
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):

        x_avg = torch.mean(x, dim=1, keepdim=True)  # torch.Size([3, 1, 64, 64])

        x_max, _ = torch.max(x, dim=1, keepdim=True)  # torch.Size([3, 1, 64, 64])

        x2 = torch.cat([x_avg, x_max], dim=1)  # torch.Size([3, 2, 64, 64])

        sattn = self.sa(x2)  # torch.Size([3, 1, 64, 64])

        return sattn

class ChannelAttention(nn.Module):

    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.ca = nn.Sequential(

            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),

            nn.ReLU(inplace=True),

            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        # x.shape = torch.Size([3, 32, 64, 64])

        x_gap = self.gap(x)  # torch.Size([3, 32, 1, 1])

        cattn = self.ca(x_gap)  # torch.Size([3, 32, 1, 1])

        return cattn

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()

        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape

        x = x.unsqueeze(dim=2)  # B, C, 1, H, W torch.Size([3, 32, 1, 64, 64])

        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W torch.Size([3, 32, 1, 64, 64])
        # x2.shape = torch.Size([3, 32, 2, 64, 64])
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        # x2.shape = torch.Size([3, 64, 64, 64])
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)

        pattn2 = self.pa2(x2)  # pattn2 torch.Size([3, 32, 64, 64])

        pattn2 = self.sigmoid(pattn2)  # pattn2 torch.Size([3, 32, 64, 64])

        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()

        self.sa = SpatialAttention()

        self.ca = ChannelAttention(dim, reduction)

        self.pa = PixelAttention(dim)

        self.conv = nn.Conv2d(dim, dim, 1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # x.shape = torch.Size([3, 32, 64, 64])
        initial = x + y  # initial.shape torch.Size([3, 32, 64, 64])。

        cattn = self.ca(initial)  # cattn torch.Size([3, 32, 1, 1])

        sattn = self.sa(initial)  # sattn torch.Size([3, 1, 64, 64])

        pattn1 = sattn + cattn  # torch.Size([3, 32, 64, 64])

        # pattn2 torch.Size([3, 32, 64, 64]) self.pa(initial, pattn1) torch.Size([3, 32, 64, 64])
        pattn2 = self.sigmoid(self.pa(initial, pattn1))

        result = initial + pattn2 * x + (1 - pattn2) * y
        # result torch.Size([3, 32, 64, 64])

        result = self.conv(result)
        # result torch.Size([3, 32, 64, 64])

        return result


if __name__ == '__main__':
    block = CGAFusion(32)
    input1 = torch.rand(3, 32, 64, 64)
    input2 = torch.rand(3, 32, 64, 64)
    output = block(input1, input2)  # torch.Size([3, 32, 64, 64])
    print(output.size())