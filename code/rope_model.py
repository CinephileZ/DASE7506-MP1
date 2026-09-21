"""GPT variant using rotary position information in causal attention."""
import torch
from torch import nn
from torch.nn import functional as F


class RoPEBlock(nn.Module):
    def __init__(self, width, heads, cos, sin):
        super().__init__()
        if width % heads != 0:
            raise ValueError('width must be divisible by heads.')
        head_dim = width // heads
        if head_dim % 2 != 0:
            raise ValueError('head dimension must be even for RoPE.')
        self.heads = heads
        self.head_dim = head_dim
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def rotate(self, x):
        length = x.shape[-2]
        cos = self.rope_cos[:length].view(1, 1, length, self.head_dim // 2)
        sin = self.rope_sin[:length].view(1, 1, length, self.head_dim // 2)
        first, second = x[..., :self.head_dim // 2], x[..., self.head_dim // 2:]
        return torch.cat((first * cos - second * sin,
                          first * sin + second * cos), dim=-1)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k = self.rotate(q), self.rotate(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class RoPEGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        heads = config['heads']
        head_dim = width // heads
        if width % heads != 0 or head_dim % 2 != 0:
            raise ValueError('width / heads must be a positive even integer.')
        positions = torch.arange(self.context, dtype=torch.float32)
        frequencies = 10000 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        angles = positions[:, None] * frequencies[None, :]
        cos, sin = angles.cos(), angles.sin()
        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([
            RoPEBlock(width, heads, cos, sin)
            for _ in range(config['depth'])
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return RoPEGPT(config)
