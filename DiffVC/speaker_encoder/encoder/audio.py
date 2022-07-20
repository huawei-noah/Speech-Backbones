""" from https://github.com/CorentinJ/Real-Time-Voice-Cloning """

from scipy.ndimage.morphology import binary_dilation
from encoder.params_data import *
from pathlib import Path
from typing import Optional, Union
import numpy as np
import webrtcvad
import librosa
import struct

import torch
from torchaudio.transforms import Resample
from librosa.filters import mel as librosa_mel_fn


int16_max = (2 ** 15) - 1


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)

    # Apply the preprocessing: normalize volume and shorten long silences 
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)
    
    return wav


def preprocess_wav_batch(wavs, source_sr=22050):
    # This torch version is designed to cope with a batch of same lengths wavs
    if sampling_rate != source_sr:
        resample = Resample(source_sr, sampling_rate)
        wavs = resample(wavs)
    wavs_preprocessed = normalize_volume_batch(wavs, audio_norm_target_dBFS, 
                                               increase_only=True)
    # Trimming silence is not implemented in this version yet!
    return wavs_preprocessed


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


def wav_to_mel_spectrogram_batch(wavs):
    # This torch version is designed to cope with a batch of same lengths wavs
    n_fft = int(sampling_rate * mel_window_length / 1000)
    hop_length = int(sampling_rate * mel_window_step / 1000)
    win_length = int(sampling_rate * mel_window_length / 1000)
    window = torch.hann_window(n_fft).to(wavs)
    mel_basis = torch.from_numpy(librosa_mel_fn(sampling_rate, n_fft, 
                                                mel_n_channels)).to(wavs)
    s = torch.stft(wavs, n_fft=n_fft, hop_length=hop_length, 
                   win_length=win_length, window=window, center=True)
    real_part, imag_part = s.unbind(-1)
    stftm = real_part**2 + imag_part**2
    mels = torch.matmul(mel_basis, stftm)
    return torch.transpose(mels, 1, 2)


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


def normalize_volume_batch(wavs, target_dBFS, increase_only=False, decrease_only=False):
    # This torch version is designed to cope with a batch of same lengths wavs
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * torch.log10(torch.mean(wavs ** 2, axis=-1))
    scales = torch.ones(wavs.shape[0], device=wavs.device, dtype=wavs.dtype)
    if increase_only:
        mask = (dBFS_change > 0).to(scales)
    elif decrease_only:
        mask = (dBFS_change < 0).to(scales)
    else:
        mask = torch.zeros_like(scales)
    scales = scales + mask * (10 ** (dBFS_change / 20) - 1.0)
    return wavs * scales.unsqueeze(-1)


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]
