# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import torch
import torchaudio
import numpy as np
from librosa.filters import mel as librosa_mel_fn

from model.base import BaseModule


def mse_loss(x, y, mask, n_feats):
    loss = torch.sum(((x - y)**2) * mask)
    return loss / (torch.sum(mask) * n_feats)


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


class PseudoInversion(BaseModule):
    def __init__(self, n_mels, sampling_rate, n_fft):
        super(PseudoInversion, self).__init__()
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        mel_basis = librosa_mel_fn(sampling_rate, n_fft, n_mels, 0, 8000)
        mel_basis_inverse = np.linalg.pinv(mel_basis)
        mel_basis_inverse = torch.from_numpy(mel_basis_inverse).float()
        self.register_buffer("mel_basis_inverse", mel_basis_inverse)

    def forward(self, log_mel_spectrogram):
        mel_spectrogram = torch.exp(log_mel_spectrogram)
        stftm = torch.matmul(self.mel_basis_inverse, mel_spectrogram)
        return stftm


class InitialReconstruction(BaseModule):
    def __init__(self, n_fft, hop_size):
        super(InitialReconstruction, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        window = torch.hann_window(n_fft).float()
        self.register_buffer("window", window)

    def forward(self, stftm):
        real_part = torch.ones_like(stftm, device=stftm.device)
        imag_part = torch.zeros_like(stftm, device=stftm.device)
        stft = torch.stack([real_part, imag_part], -1)*stftm.unsqueeze(-1)
        istft = torchaudio.functional.istft(stft, n_fft=self.n_fft, 
                           hop_length=self.hop_size, win_length=self.n_fft, 
                           window=self.window, center=True)
        return istft.unsqueeze(1)


# Fast Griffin-Lim algorithm as a PyTorch module
class FastGL(BaseModule):
    def __init__(self, n_mels, sampling_rate, n_fft, hop_size, momentum=0.99):
        super(FastGL, self).__init__()
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.momentum = momentum
        self.pi = PseudoInversion(n_mels, sampling_rate, n_fft)
        self.ir = InitialReconstruction(n_fft, hop_size)
        window = torch.hann_window(n_fft).float()
        self.register_buffer("window", window)

    @torch.no_grad()
    def forward(self, s, n_iters=32):
        c = self.pi(s)
        x = self.ir(c)
        x = x.squeeze(1)
        c = c.unsqueeze(-1)
        prev_angles = torch.zeros_like(c, device=c.device)
        for _ in range(n_iters):        
            s = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_size, 
                           win_length=self.n_fft, window=self.window, 
                           center=True)
            real_part, imag_part = s.unbind(-1)
            stftm = torch.sqrt(torch.clamp(real_part**2 + imag_part**2, min=1e-8))
            angles = s / stftm.unsqueeze(-1)
            s = c * (angles + self.momentum * (angles - prev_angles))
            x = torchaudio.functional.istft(s, n_fft=self.n_fft, hop_length=self.hop_size, 
                                            win_length=self.n_fft, window=self.window, 
                                            center=True)
            prev_angles = angles
        return x.unsqueeze(1)
