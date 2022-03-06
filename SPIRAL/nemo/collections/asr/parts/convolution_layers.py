# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from torch import nn as nn
import torch.nn.functional as F


conv_dic = {'1d': torch.nn.Conv1d, '2d': torch.nn.Conv2d}

act_dic = {"hardtanh": nn.Hardtanh, "relu": nn.ReLU}


class ProjUpsampling(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, *, rate, norm_type, act_func,
                 dropout=0.0,
                 padding='same',
                 use_tf_pad=True,
                 ln_eps=1e-5,
                 bias=True):
        super(ProjUpsampling, self).__init__()

        self.upsample_rate = rate
        self.filters = filters
        self.proj = ConvNormAct(in_channels=in_channels, filters=self.filters * self.upsample_rate, kernel_size=kernel_size,
                                stride=(1,), dilation=(1,), norm_type=None, act_func=None,
                                conv_type='1d', dropout=0.0,
                                padding=padding, use_tf_pad=use_tf_pad, ln_eps=ln_eps, gn_groups=None, bias=bias)

        assert norm_type is None or norm_type == 'ln'
        self.norm = get_norm(norm_type, '1d', self.filters, ln_eps=ln_eps, gn_groups=None)
        self.norm_type = norm_type
        self.act = identity if act_func is None else act_dic[act_func]()
        self.drop = identity if dropout == 0 else nn.Dropout(p=dropout)

    def forward(self, x, lens):
        pad_mask = create_pad_mask(lens, max_len=x.size(2))
        output, lens, _ = self.proj(x, lens, pad_mask=pad_mask)
        output = output.transpose(1, 2)
        B, T, C = output.size()
        output = output.reshape(B, T * self.upsample_rate, self.filters)
        lens = lens * self.upsample_rate
        output = self.norm(output)
        output = self.act(output)
        output = self.drop(output)
        output = output.transpose(1, 2)
        return output, lens


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride, dilation, norm_type, act_func,
                 conv_type,
                 dropout=0.0,
                 padding='same',
                 use_tf_pad=True,
                 ln_eps=1e-5,
                 gn_groups=None,
                 bias=None):
        super(ConvNormAct, self).__init__()

        if bias is None:
            bias = norm_type is None

        self.conv = Conv(in_channels, filters, tuple(kernel_size),
                         stride=tuple(stride),
                         padding=padding,
                         dilation=tuple(dilation),
                         bias=bias,
                         conv_type=conv_type,
                         use_tf_pad=use_tf_pad)
        self.proj_conv = None
        assert conv_type in ['1d', '2d']
        self.norm = get_norm(norm_type, conv_type, filters, ln_eps=ln_eps, gn_groups=gn_groups)
        self.norm_type = norm_type
        self.act = identity if act_func is None else act_dic[act_func]()
        self.drop = identity if dropout == 0 else nn.Dropout(p=dropout)

    def forward(self, x, lens, pad_mask=None):
        # x: [B, C, T] or [B, C, T, F]

        output, lens, pad_mask = self.conv(x, lens, pad_mask)
        if self.norm_type == 'ln':
            output = torch.transpose(output, -1, -2)
        output = self.norm(output)
        if self.norm_type == 'ln':
            output = torch.transpose(output, -1, -2)
        output = self.act(output)
        output = self.drop(output)

        return output, lens, pad_mask

    def update_out_seq_lens(self, lens):
        return self.conv.update_out_seq_lens(lens)


def get_norm(norm_type, conv_type, filters, ln_eps=1e-5, gn_groups=None):
    if norm_type == 'bn':
        if conv_type == '2d':
            norm = nn.BatchNorm2d(filters, momentum=0.01, eps=1e-3)
        else:
            norm = nn.BatchNorm1d(filters, momentum=0.01, eps=1e-3)
    elif norm_type == 'ln':
        assert conv_type != '2d'
        norm = nn.LayerNorm(filters, eps=ln_eps)
    elif norm_type == 'gn':
        assert gn_groups is not None
        norm = nn.GroupNorm(gn_groups, filters)
    else:
        assert norm_type is None, norm_type
        norm = identity
    return norm


# conv wrapper supports same padding, tf style padding, track length change during subsampling
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(1,),
                 padding='same',
                 dilation=(1,),
                 bias=True,
                 conv_type='1d',
                 use_tf_pad=False):
        super(Conv, self).__init__()

        self.conv_type = conv_type
        self.is_2d_conv = self.conv_type == '2d'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
            if self.use_conv2d:
                kernel_size = kernel_size * 2

        if isinstance(stride, int):
            stride = (stride,)
            if self.use_conv2d:
                stride = stride * 2

        self.padding = padding

        assert dilation == (1,) or dilation == (1, 1)

        assert use_tf_pad
        self.use_tf_pad = use_tf_pad
        if self.use_tf_pad:
            self.pad_num, self.even_pad_num = get_tf_pad(kernel_size, stride)

        self.conv = conv_dic[self.conv_type](in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=self.get_padding_num(kernel_size, stride, dilation),
                                             bias=bias)
        self.need_pad = kernel_size[0] > 1 or (len(kernel_size) == 2 and kernel_size[1] > 1)
        self.need_pad_mask = kernel_size[0] > 1
        assert stride[0] >= 1
        self.subsample_factor = stride[0]

    def forward(self, x, lens, pad_mask=None):
        # x: [B, C, T] or [B, C, T, F]
        if pad_mask is not None and self.need_pad_mask:
            if self.is_2d_conv:
                x = x.masked_fill(pad_mask.unsqueeze(1).unsqueeze(-1), 0.0)
            else:
                x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        if self.use_tf_pad and self.need_pad:
            x = self.pad_like_tf(x)

        output = self.conv(x)

        if self.subsample_factor > 1:
            lens = self.update_out_seq_lens(lens)
            pad_mask = create_pad_mask(lens, max_len=output.size(2))

        return output, lens, pad_mask

    def get_padding_num(self, kernel_size, stride, dilation):
        if self.padding == 'same':
            if self.use_tf_pad:
                padding_val = 0
            else:
                assert not self.use_tf_pad
                padding_val = get_same_padding(kernel_size, stride, dilation)
        else:
            raise ValueError("currently only 'same' padding is supported")
        return padding_val

    def update_out_seq_lens(self, lens):
        t = 0  # axis of time dimension
        if self.padding == 'same':
            if self.use_tf_pad:
                lens = (lens + self.conv.stride[t] - 1) // self.conv.stride[t]
            else:
                # todo: verify this in pytorch
                lens = (lens + 2 * self.conv.padding[t] - self.conv.dilation[t] * (self.conv.kernel_size[t] - 1) - 1) // self.conv.stride[t] + 1
        else:
            assert self.padding == 'valid' and self.use_tf_pad
            lens = (lens - self.conv.kernel_size[t] + self.conv.stride[t]) // self.conv.stride[t]
        return lens

    def pad_like_tf(self, x):
        if self.is_2d_conv:
            if x.size(-1) % 2 == 0:
                w_pad_num = self.even_pad_num[1]
            else:
                w_pad_num = self.pad_num[1]
            if x.size(-2) % 2 == 0:
                h_pad_num = self.even_pad_num[0]
            else:
                h_pad_num = self.pad_num[0]
            pad_num = w_pad_num + h_pad_num
        else:
            if x.size(-2) % 2 == 0:
                pad_num = self.even_pad_num[0]
            else:
                pad_num = self.pad_num[0]

        return F.pad(x, pad_num)


def get_same_padding(kernel_size, stride, dilation):
    # todo: support 2d conv
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2


def get_tf_pad(kernel_size, stride):
    pad_config = []
    even_pad_config = []
    for i in range(len(kernel_size)):
        assert kernel_size[i] % 2 == 1
        pad_num_i = kernel_size[i] // 2
        pad_config.append([pad_num_i, pad_num_i])
        if stride[i] == 2:
            even_pad_config.append([pad_num_i - 1, pad_num_i])
        else:
            assert stride[i] == 1
            even_pad_config.append([pad_num_i, pad_num_i])
    return pad_config, even_pad_config


def create_pad_mask(lens, max_len=None):
    mask = torch.arange(max_len).to(lens.device) >= lens.unsqueeze(-1)
    return mask


def identity(x):
    return x
