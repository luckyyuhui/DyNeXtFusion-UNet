import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn, alpha=1.0):
        super().__init__()
        self.fn = fn
        self.alpha = alpha

    def forward(self, x):
        return x + self.alpha * self.fn(x)

def _get_act(name: str):
    name = name.lower()
    return {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True)
    }[name]

def _norm_2d(norm: str, num_channels: int):
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        for g in (8, 4, 2, 1):
            if num_channels % g == 0:
                return nn.GroupNorm(g, num_channels)
        return nn.GroupNorm(1, num_channels)
    elif norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True, track_running_stats=False)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

class CMUNeXtBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 7,
        depth: int = 1,
        expand_ratio: int = 4,
        act: str = "gelu",
        norm: str = "bn",
        res_scale: float = 1.0
    ):
        super().__init__()
        assert in_channels > 0 and out_channels > 0
        assert k % 2 == 1

        Act = lambda c: _get_act(act)
        Norm = lambda c: _norm_2d(norm, c)

        body = []
        for _ in range(depth):
            body.append(
                Residual(
                    nn.Sequential(

                        nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k // 2,
                                  groups=in_channels, bias=False),
                        Norm(in_channels),
                        Act(in_channels),

                        nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1, bias=False),
                        Norm(in_channels * expand_ratio),
                        Act(in_channels * expand_ratio),
                        nn.Conv2d(in_channels * expand_ratio, in_channels, kernel_size=1, bias=False),
                        Norm(in_channels),
                        Act(in_channels),
                    ),
                    alpha=res_scale
                )
            )
        self.body = nn.Sequential(*body)


        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Norm(out_channels),
            Act(out_channels)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.groups > 1 and m.kernel_size[0] >= 5:
                    m.weight.data.mul_(0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.body(x)
        x = self.proj(x)
        return x
