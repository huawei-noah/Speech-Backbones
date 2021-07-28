import argparse
import json
import datetime as dt
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write

import torch

import os
import sys
from subprocess import call

def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)
  
current = os.getcwd()
print(current)
full = current + "/Grad-TTS/model/monotonic_align"
print(full)
os.chdir(full)
print(os.getcwd())
run_cmd("python3 setup.py build_ext --inplace")
os.chdir("../../..")
print(os.getcwd())

# For Grad-TTS
import sys
sys.path.append('Grad-TTS/')
import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

# For HiFi-GAN
sys.path.append('Grad-TTS/hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
import scipy
import gradio as gr



torch.hub.download_url_to_file('https://github.com/AK391/Speech-Backbones/releases/download/v1/grad-tts-old.pt', './Grad-TTS/checkpts/grad-tts-old.pt')

torch.hub.download_url_to_file('https://github.com/AK391/Speech-Backbones/releases/download/v1/grad-tts.pt', './Grad-TTS/checkpts/grad-tts.pt')

torch.hub.download_url_to_file('https://github.com/AK391/Speech-Backbones/releases/download/v1/hifigan.pt', './Grad-TTS/checkpts/hifigan.pt')


generator = GradTTS(len(symbols)+1, params.n_enc_channels, params.filter_channels,
                    params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                    params.enc_kernel, params.enc_dropout, params.window_size,
                    params.n_feats, params.dec_dim, params.beta_min, params.beta_max,
                    pe_scale=1000)  # pe_scale=1 for `grad-tts-old.pt`
generator.load_state_dict(torch.load('./Grad-TTS/checkpts/grad-tts.pt', map_location=lambda loc, storage: loc))
_ = generator

cmu = cmudict.CMUDict('./Grad-TTS/resources/cmu_dictionary')
with open('./Grad-TTS/checkpts/hifigan-config.json') as f:
    h = AttrDict(json.load(f))
hifigan = HiFiGAN(h)
hifigan.load_state_dict(torch.load('./Grad-TTS/checkpts/hifigan.pt',
                                   map_location=lambda loc, storage: loc)['generator'])
_ = hifigan.eval()
hifigan.remove_weight_norm()
def inference(text):
    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols)))[None]
    x_lengths = torch.LongTensor([x.shape[-1]])
    x.shape, x_lengths
    y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=3, temperature=5,
                                        stoc=False, length_scale=0.91)

    with torch.no_grad():
        audio = hifigan.forward(y_dec).cpu().squeeze().clamp(-1, 1).detach().numpy()

    scipy.io.wavfile.write("out.wav", 22050, audio)
    return "./out.wav"

inputs = gr.inputs.Textbox(lines=5, label="Input Text")
outputs =  gr.outputs.Audio(label="Output Audio", type="file")


title = "Grad-TTS"
description = "Gradio demo for Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2105.06337'>Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech</a> | <a href='https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS'>Github Repo</a></p>"

examples = [
 ["Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes"],
 ["In this paper we introduce Grad-TTS, a novel text-to-speech model with score-based decoder producing mel-spectrograms by gradually transforming noise predicted by encoder and aligned with text input by means of Monotonic Alignment Search."]   
]

gr.Interface(inference, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()
