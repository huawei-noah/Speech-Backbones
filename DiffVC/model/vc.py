# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import torch

from model.base import BaseModule
from model.encoder import MelEncoder
from model.postnet import PostNet
from model.diffusion import Diffusion
from model.utils import sequence_mask, fix_len_compatibility, mse_loss


# "average voice" encoder as the module parameterizing the diffusion prior
class FwdDiffusion(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, dim):
        super(FwdDiffusion, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.dim = dim
        self.encoder = MelEncoder(n_feats, channels, filters, heads, layers, 
                                  kernel, dropout, window_size)
        self.postnet = PostNet(dim)

    @torch.no_grad()
    def forward(self, x, mask):
        x, mask = self.relocate_input([x, mask])
        z = self.encoder(x, mask)
        z_output = self.postnet(z, mask)
        return z_output

    def compute_loss(self, x, y, mask):
        x, y, mask = self.relocate_input([x, y, mask])
        z = self.encoder(x, mask)
        z_output = self.postnet(z, mask)
        loss = mse_loss(z_output, y, mask, self.n_feats)
        return loss


# the whole voice conversion model consisting of the "average voice" encoder 
# and the diffusion-based speaker-conditional decoder
class DiffVC(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, enc_dim, spk_dim, use_ref_t, dec_dim, 
                 beta_min, beta_max):
        super(DiffVC, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.enc_dim = enc_dim
        self.spk_dim = spk_dim
        self.use_ref_t = use_ref_t
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.encoder = FwdDiffusion(n_feats, channels, filters, heads, layers,
                                    kernel, dropout, window_size, enc_dim)
        self.decoder = Diffusion(n_feats, dec_dim, spk_dim, use_ref_t, 
                                 beta_min, beta_max)

    def load_encoder(self, enc_path):
        enc_dict = torch.load(enc_path, map_location=lambda loc, storage: loc)
        self.encoder.load_state_dict(enc_dict, strict=False)

    @torch.no_grad()
    def forward(self, x, x_lengths, x_ref, x_ref_lengths, c, n_timesteps, 
                mode='ml'):
        """
        Generates mel-spectrogram from source mel-spectrogram conditioned on
        target speaker embedding. Returns:
            1. 'average voice' encoder outputs
            2. decoder outputs
        
        Args:
            x (torch.Tensor): batch of source mel-spectrograms.
            x_lengths (torch.Tensor): numbers of frames in source mel-spectrograms.
            x_ref (torch.Tensor): batch of reference mel-spectrograms.
            x_ref_lengths (torch.Tensor): numbers of frames in reference mel-spectrograms.
            c (torch.Tensor): batch of reference speaker embeddings
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            mode (string, optional): sampling method. Can be one of:
              'pf' - probability flow sampling (Euler scheme for ODE)
              'em' - Euler-Maruyama SDE solver
              'ml' - Maximum Likelihood SDE solver
        """
        x, x_lengths = self.relocate_input([x, x_lengths])
        x_ref, x_ref_lengths, c = self.relocate_input([x_ref, x_ref_lengths, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        x_ref_mask = sequence_mask(x_ref_lengths).unsqueeze(1).to(x_ref.dtype)
        mean = self.encoder(x, x_mask)
        mean_x = self.decoder.compute_diffused_mean(x, x_mask, mean, 1.0)
        mean_ref = self.encoder(x_ref, x_ref_mask)

        b = x.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(x.dtype)
        mean_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                device=x.device)
        mean_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                  device=x.device)
        for i in range(b):
            mean_new[i, :, :x_lengths[i]] = mean[i, :, :x_lengths[i]]
            mean_x_new[i, :, :x_lengths[i]] = mean_x[i, :, :x_lengths[i]]

        z = mean_x_new
        z += torch.randn_like(mean_x_new, device=mean_x_new.device)

        y = self.decoder(z, x_mask_new, mean_new, x_ref, x_ref_mask, mean_ref, c, 
                         n_timesteps, mode)
        return mean_x, y[:, :, :max_length]

    def compute_loss(self, x, x_lengths, x_ref, c):
        """
        Computes diffusion (score matching) loss.
            
        Args:
            x (torch.Tensor): batch of source mel-spectrograms.
            x_lengths (torch.Tensor): numbers of frames in source mel-spectrograms.
            x_ref (torch.Tensor): batch of reference mel-spectrograms.
            c (torch.Tensor): batch of reference speaker embeddings
        """
        x, x_lengths, x_ref, c = self.relocate_input([x, x_lengths, x_ref, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        mean = self.encoder(x, x_mask).detach()
        mean_ref = self.encoder(x_ref, x_mask).detach()
        diff_loss = self.decoder.compute_loss(x, x_mask, mean, x_ref, mean_ref, c)
        return diff_loss
