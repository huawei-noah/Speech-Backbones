# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import torch

from model.base import BaseModule
from model.modules import Mish


class Block(BaseModule):
    def __init__(self, dim, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim, 7, 
                     padding=3), torch.nn.GroupNorm(groups, dim), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.block1 = Block(dim, groups=groups)
        self.block2 = Block(dim, groups=groups)
        self.res = torch.nn.Conv2d(dim, dim, 1)

    def forward(self, x, mask):
        h = self.block1(x, mask)
        h = self.block2(h, mask)
        output = self.res(x * mask) + h
        return output


class PostNet(BaseModule):
    def __init__(self, dim, groups=8):
        super(PostNet, self).__init__()
        self.init_conv = torch.nn.Conv2d(1, dim, 1)
        self.res_block = ResnetBlock(dim, groups=groups)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask):
        x = x.unsqueeze(1)
        mask = mask.unsqueeze(1)
        x = self.init_conv(x * mask)
        x = self.res_block(x, mask)
        output = self.final_conv(x * mask)
        return output.squeeze(1)
