# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch
from einops import rearrange

from model.base import BaseModule


class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = 1000.0 * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RefBlock(BaseModule):
    def __init__(self, out_dim, time_emb_dim):
        super(RefBlock, self).__init__()
        base_dim = out_dim // 4
        self.mlp1 = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                                base_dim))
        self.mlp2 = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                                2 * base_dim))
        self.block11 = torch.nn.Sequential(torch.nn.Conv2d(1, 2 * base_dim, 
                      3, 1, 1), torch.nn.InstanceNorm2d(2 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block12 = torch.nn.Sequential(torch.nn.Conv2d(base_dim, 2 * base_dim, 
                      3, 1, 1), torch.nn.InstanceNorm2d(2 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block21 = torch.nn.Sequential(torch.nn.Conv2d(base_dim, 4 * base_dim,
                      3, 1, 1), torch.nn.InstanceNorm2d(4 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block22 = torch.nn.Sequential(torch.nn.Conv2d(2 * base_dim, 4 * base_dim,
                      3, 1, 1), torch.nn.InstanceNorm2d(4 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block31 = torch.nn.Sequential(torch.nn.Conv2d(2 * base_dim, 8 * base_dim,
                      3, 1, 1), torch.nn.InstanceNorm2d(8 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block32 = torch.nn.Sequential(torch.nn.Conv2d(4 * base_dim, 8 * base_dim,
                      3, 1, 1), torch.nn.InstanceNorm2d(8 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.final_conv = torch.nn.Conv2d(4 * base_dim, out_dim, 1)

    def forward(self, x, mask, time_emb):
        y = self.block11(x * mask)
        y = self.block12(y * mask)
        y += self.mlp1(time_emb).unsqueeze(-1).unsqueeze(-1)
        y = self.block21(y * mask)
        y = self.block22(y * mask)
        y += self.mlp2(time_emb).unsqueeze(-1).unsqueeze(-1)
        y = self.block31(y * mask)
        y = self.block32(y * mask)
        y = self.final_conv(y * mask)
        return (y * mask).sum((2, 3)) / (mask.sum((2, 3)) * x.shape[2])
