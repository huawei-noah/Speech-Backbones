# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import params
from data import VCDecDataset, VCDecBatchCollate
from model.vc import DiffVC
from model.utils import FastGL
from utils import save_plot, save_audio

n_mels = params.n_mels
sampling_rate = params.sampling_rate
n_fft = params.n_fft
hop_size = params.hop_size

channels = params.channels
filters = params.filters
layers = params.layers
kernel = params.kernel
dropout = params.dropout
heads = params.heads
window_size = params.window_size
enc_dim = params.enc_dim

dec_dim = params.dec_dim
spk_dim = params.spk_dim
use_ref_t = params.use_ref_t
beta_min = params.beta_min
beta_max = params.beta_max

random_seed = params.seed
test_size = params.test_size

data_dir = '../data/LibriTTS'
val_file = 'filelists/valid.txt'
exc_file = 'filelists/exceptions_libritts.txt'

log_dir = 'logs_dec'
enc_dir = 'logs_enc'
epochs = 110
batch_size = 32
learning_rate = 1e-4
save_every = 1


if __name__ == "__main__":

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    os.makedirs(log_dir, exist_ok=True)

    print('Initializing data loaders...')
    train_set = VCDecDataset(data_dir, val_file, exc_file)
    collate_fn = VCDecBatchCollate()
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                              collate_fn=collate_fn, num_workers=4, drop_last=True)

    print('Initializing and loading models...')
    fgl = FastGL(n_mels, sampling_rate, n_fft, hop_size).cuda()
    model = DiffVC(n_mels, channels, filters, heads, layers, kernel, 
                   dropout, window_size, enc_dim, spk_dim, use_ref_t, 
                   dec_dim, beta_min, beta_max).cuda()
    model.load_encoder(os.path.join(enc_dir, 'enc.pt'))

    print('Encoder:')
    print(model.encoder)
    print('Number of parameters = %.2fm\n' % (model.encoder.nparams/1e6))
    print('Decoder:')
    print(model.decoder)
    print('Number of parameters = %.2fm\n' % (model.decoder.nparams/1e6))

    print('Initializing optimizers...')
    optimizer = torch.optim.Adam(params=model.decoder.parameters(), lr=learning_rate)

    print('Start training.')
    torch.backends.cudnn.benchmark = True
    iteration = 0
    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        model.train()
        losses = []
        for batch in tqdm(train_loader, total=len(train_set)//batch_size):
            mel, mel_ref = batch['mel1'].cuda(), batch['mel2'].cuda()
            c, mel_lengths = batch['c'].cuda(), batch['mel_lengths'].cuda()
            model.zero_grad()
            loss = model.compute_loss(mel, mel_lengths, mel_ref, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1)
            optimizer.step()
            losses.append(loss.item())
            iteration += 1

        losses = np.asarray(losses)
        msg = 'Epoch %d: loss = %.4f\n' % (epoch, np.mean(losses))
        print(msg)
        with open(f'{log_dir}/train_dec.log', 'a') as f:
            f.write(msg)
        losses = []

        if epoch % save_every > 0:
            continue

        model.eval()
        print('Inference...\n')
        with torch.no_grad():
            mels = train_set.get_valid_dataset()
            for i, (mel, c) in enumerate(mels):
                if i >= test_size:
                    break
                mel = mel.unsqueeze(0).float().cuda()
                c = c.unsqueeze(0).float().cuda()
                mel_lengths = torch.LongTensor([mel.shape[-1]]).cuda()
                mel_avg, mel_rec = model(mel, mel_lengths, mel, mel_lengths, c, 
                                         n_timesteps=100)
                if epoch == save_every:
                    save_plot(mel.squeeze().cpu(), f'{log_dir}/original_{i}.png')
                    audio = fgl(mel)
                    save_audio(f'{log_dir}/original_{i}.wav', sampling_rate, audio)
                save_plot(mel_avg.squeeze().cpu(), f'{log_dir}/average_{i}.png')
                audio = fgl(mel_avg)
                save_audio(f'{log_dir}/average_{i}.wav', sampling_rate, audio)
                save_plot(mel_rec.squeeze().cpu(), f'{log_dir}/reconstructed_{i}.png')
                audio = fgl(mel_rec)
                save_audio(f'{log_dir}/reconstructed_{i}.wav', sampling_rate, audio)

        print('Saving model...\n')
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/vc_{epoch}.pt")
