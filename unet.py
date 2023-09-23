from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn

from attention import SelfAttention
from nn import ResBlock, Downsample, Upsample, get_norm


@dataclass
class UNetConfig:
    input_channels: int
    input_size: int
    channels: int
    num_blocks: List[int]
    channel_mults: List[int]
    attn_resolutions: List[int]
    conditional: bool = False


class UNet(torch.nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()

        self.cfg = cfg

        self.temb_nn = torch.nn.Sequential(
            torch.nn.Linear(self.cfg.channels, 4 * self.cfg.channels),
            get_norm(4*self.cfg.channels),
            torch.nn.SiLU(),
            torch.nn.Linear(4 * self.cfg.channels, 256),
            get_norm(256),
            torch.nn.SiLU()
        )

        self.cond_nn = torch.nn.Sequential(
            torch.nn.Linear(10, 4 * self.cfg.channels),
            get_norm(4*self.cfg.channels),
            torch.nn.SiLU(),
            torch.nn.Linear(4 * self.cfg.channels, 256),
            get_norm(256),
            torch.nn.SiLU()
        ) if self.cfg.conditional else None

        self.conv0 = torch.nn.Conv2d(
            self.cfg.input_channels, self.cfg.channels, kernel_size=3, padding="same")

        current_size = self.cfg.input_size
        self.downs = []
        self.ups = []
        for i, (nb, cm) in enumerate(zip(self.cfg.num_blocks, self.cfg.channel_mults)):
            down_layers = []

            if current_size < self.cfg.input_size:
                down_layers.append(Downsample(
                    self.cfg.channel_mults[i - 1] * self.cfg.channels, cm * self.cfg.channels))

            for _ in range(nb):
                down_layers.append(ResBlock(
                    cm * self.cfg.channels, cm * self.cfg.channels, cm * self.cfg.channels,
                    cond=self.cfg.conditional))
                if current_size in self.cfg.attn_resolutions:
                    down_layers.append(SelfAttention((current_size, current_size),
                                                     cm * self.cfg.channels, 32, True))

            self.downs.append(torch.nn.ModuleList(down_layers))

            up_layers = []
            for j in range(nb):
                if i < len(self.cfg.num_blocks) - 1 and j == 0:
                    up_layers.append(ResBlock(
                        2 * cm * self.cfg.channels, cm * self.cfg.channels, cm * self.cfg.channels,
                        cond=self.cfg.conditional))
                else:
                    up_layers.append(ResBlock(
                        cm * self.cfg.channels, cm * self.cfg.channels, cm * self.cfg.channels,
                        cond=self.cfg.conditional))

                if current_size in self.cfg.attn_resolutions:
                    up_layers.append(SelfAttention((current_size, current_size),
                                                   cm * self.cfg.channels, 32, True))

            if current_size < self.cfg.input_size:
                up_layers.append(Upsample(
                    cm * self.cfg.channels, self.cfg.channel_mults[i - 1] * self.cfg.channels))

            self.ups.append(torch.nn.ModuleList(up_layers))

            current_size //= 2

        self.downs = torch.nn.ModuleList(self.downs)
        self.ups = torch.nn.ModuleList(self.ups[::-1])

        self.out_nn = torch.nn.Sequential(
            torch.nn.SiLU(),
            get_norm(self.cfg.channels),
            torch.nn.Conv2d(
                self.cfg.channels, self.cfg.input_channels, kernel_size=3, padding="same")
        )

    def forward(self, x: torch.Tensor, time_embs: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        temb = self.temb_nn(time_embs)
        cond = self.cond_nn(cond) if self.cfg.conditional else None

        x = self.conv0(x)

        down_outputs = [x]

        for d in self.downs:
            for l in d:
                x = l(x, temb, cond)
            down_outputs.append(x)

        for i, u in enumerate(self.ups):
            if i > 0:
                x = torch.concat([down_outputs[len(self.ups) - i], x], dim=1)
            for l in u:
                x = l(x, temb, cond)

        x = self.out_nn(x)

        return x

    def separate_parameters(self):
        parameters_high_decay = set()
        for m_name, m in self.named_modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)) and \
                m_name.endswith(("fc_q", "fc_k", "fc_v")):
                for p_name, param in m.named_parameters():
                    if p_name.endswith("weight"):
                        parameters_high_decay.add(param)

        parameters_low_decay = set(
            self.parameters()) - parameters_high_decay

        # sanity check
        assert len(parameters_high_decay & parameters_low_decay) == 0
        assert len(parameters_high_decay) + len(parameters_low_decay) == \
            len(list(self.parameters()))

        return list(parameters_high_decay), list(parameters_low_decay)
