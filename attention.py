import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from nn import get_norm


class RelativePositionEncoding(nn.Module):
    def __init__(self, num_heads: int, height: int, width: int):
        super().__init__()

        self.num_heads = num_heads
        self.height = height
        self.width = width

        self.pos_enc = nn.Parameter(
            torch.empty(num_heads, (2 * height - 1) * (2 * width - 1)).normal_(std=0.02))


        self.register_buffer("relative_indices", self.get_indices(), persistent=False)

    def forward(self, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_encoding(self):
        rel_pos_enc = self.pos_enc.gather(-1, self.relative_indices)
        rel_pos_enc = rel_pos_enc.unflatten(-1, (self.height*self.width, self.height*self.width))
        return rel_pos_enc

    def get_indices(self):
        h, w = self.height, self.width

        y = torch.arange(h, dtype=torch.long)
        x = torch.arange(w, dtype=torch.long)

        y1, x1, y2, x2 = torch.meshgrid(y, x, y, x, indexing='ij')
        indices = (y1 - y2 + h - 1) * (2 * w - 1) + x1 - x2 + w - 1
        indices = indices.flatten()

        return indices.expand(self.num_heads, -1)


class SelfAttention(nn.Module):
    def __init__(self, shape2d: Tuple[int, int], embedding_dims: int, head_channels: int,
                 relative_encoding: bool):
        super().__init__()
        assert 0 == (embedding_dims % head_channels)

        self.norm0 = get_norm(embedding_dims) #nn.LayerNorm((embedding_dims,)+shape2d)

        self.shape2d = shape2d
        self.embedding_dims = embedding_dims
        self.num_heads = embedding_dims // head_channels
        self.relative_encoding = relative_encoding

        self.fc_q = nn.Conv2d(self.embedding_dims, self.embedding_dims, kernel_size=1)
        self.fc_k = nn.Conv2d(self.embedding_dims, self.embedding_dims, kernel_size=1)
        self.fc_v = nn.Conv2d(self.embedding_dims, self.embedding_dims, kernel_size=1)
        self.attention_dropout = nn.Dropout(0.0)

        self.fc_proj = nn.Conv2d(self.embedding_dims, self.embedding_dims, kernel_size=1)
        self.output_dropout = nn.Dropout(0.0)

        if self.relative_encoding:
            height, width = self.shape2d
            self.rel_pos_enc = RelativePositionEncoding(self.num_heads, height, width)

    def forward(self, x: torch.Tensor, _1: torch.Tensor, _2: torch.Tensor) -> torch.Tensor:
        x0 = x
        x = self.norm0(x)

        batch_dim, embedding_channels, height, width = x.size()
        num_embeddings = height * width

        q, k, v = self.fc_q(x), self.fc_k(x), self.fc_v(x)

        # (B, H, L, HC)
        q = q.view(q.shape[0], self.num_heads, self.embedding_dims // self.num_heads,
                   num_embeddings).permute(0, 1, 3, 2)
        k = k.view(q.shape[0], self.num_heads, self.embedding_dims // self.num_heads,
                   num_embeddings).permute(0, 1, 3, 2)
        v = v.view(q.shape[0], self.num_heads, self.embedding_dims // self.num_heads,
                   num_embeddings).permute(0, 1, 3, 2)

        # (B, H, L, HC) x (B, H, HC, L) -> (B, H, L, L)
        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.relative_encoding:
            attention += self.rel_pos_enc.get_encoding()

        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)

        output = attention @ v  # (B, H, L, L) x (B, H, L, HC) -> (B, H, L, HC)
        output = output.transpose(2, 3).contiguous().view(batch_dim, embedding_channels,
                                                          height, width)  # (B, C, H, W)

        output = self.output_dropout(self.fc_proj(output))

        return output + x0
