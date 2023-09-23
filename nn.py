import math
from typing import Optional
import einops
import torch
import torch.nn


def get_norm(num_channels: int) -> torch.nn.Module:
    #return torch.nn.BatchNorm2d(num_channels, momentum=0.01)
    return torch.nn.GroupNorm(num_channels//16, num_channels)
    # return torch.nn.GroupNorm(num_groups=32, num_channels=num_channels)


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0237
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0],
                                   s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1],
                                   s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return torch.nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Downsample(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> torch.Tensor:
        super().__init__()
        self.norm = get_norm(in_channels)
        self.act = torch.nn.SiLU()
        # self.conv = torch.nn.Conv2d(channels, channels, kernel_size=2, stride=2)
        self.conv = Conv2dSame(in_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor, _1: Optional[torch.Tensor] = None,
                _2: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.act(self.norm(x))
        return self.conv(x)


class Upsample(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> torch.Tensor:
        super().__init__()
        # self.up_sample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.norm = get_norm(in_channels)
        self.act = torch.nn.SiLU()
        self.conv = torch.nn.Conv2d(in_channels, 4 * out_channels, kernel_size=1, padding="same")

    def forward(self, x: torch.Tensor, _1: Optional[torch.Tensor] = None,
                _2: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.act(self.norm(x))
        x = self.conv(x)
        x = einops.rearrange(x, 'b (h1 w1 c) h w -> b c (h h1) (w w1)', h1=2, w1=2)
        # x = self.up_sample(x)
        return x


class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: Optional[float] = 0.0):
        super().__init__()

        modules = [
            get_norm(in_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        ]

        self.nn = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 temb: bool = True, cond: bool = True) -> None:
        super().__init__()

        self.out_channels = out_channels

        self.temb_dense = torch.nn.Linear(256, hidden_channels) if temb else None
        self.cond_dense = torch.nn.Linear(256, hidden_channels) if cond else None

        if in_channels != out_channels:
            self.short_cut = torch.nn.Conv2d(in_channels, out_channels,
                                             kernel_size=1, padding="same")
        else:
            self.short_cut = torch.nn.Identity()

        self.out_nn_1 = BasicBlock(in_channels, out_channels)
        self.out_nn_2 = BasicBlock(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor,
                temb: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:

        x0 = self.short_cut(x)

        x = self.out_nn_1(x)

        if self.temb_dense:
            tt = self.temb_dense(temb)
            x += tt.unsqueeze(dim=2).unsqueeze(dim=3)

        if self.cond_dense:
            cc = self.cond_dense(cond)
            x += cc.unsqueeze(dim=2).unsqueeze(dim=3)

        x = self.out_nn_2(x)

        return x0 + x
