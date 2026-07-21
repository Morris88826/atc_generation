#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mix_speech_and_noise_single.py

Mix one speech file and one noise file.

Steps:
  1. Load both (mono, resampled to 16 kHz by default).
  2. Tile/trim noise to match speech length.
  3. Optionally scale noise to a target SNR (speech:noise, in dB).
  4. Add and peak-normalize.
  5. Save to <out_dir>/<speech_basename>_mix[_{SNR}dB].wav
"""

import os
import argparse

import numpy as np
import librosa
import scipy
from scipy.io import wavfile

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def ensure_length(noise: np.ndarray, target_len: int) -> np.ndarray:
    """Tile or trim noise to match target length."""
    cur_len = len(noise)
    if cur_len == target_len:
        return noise
    if cur_len < target_len:
        reps = int(np.ceil(target_len / cur_len))
        noise = np.tile(noise, reps)[:target_len]
    else:
        noise = noise[:target_len]
    return noise


def mix_speech_and_noise(
    speech_path: str,
    noise_path: str,
    out_path: str,
    sr: int = 16000,
    snr_db: float = None,
):
    speech, _ = librosa.load(speech_path, sr=8000, mono=True)
    speech = librosa.resample(speech, orig_sr=8000, target_sr=sr)
    sos = scipy.signal.butter(4, 200, 'high', fs=sr, output='sos')
    speech = scipy.signal.sosfilt(sos, speech)

    noise, _ = librosa.load(noise_path, sr=sr, mono=True)

    speech = speech.astype("float32")
    noise = noise.astype("float32")

    # Match length
    noise = ensure_length(noise, len(speech))

    # Optional SNR control
    if snr_db is not None:
        speech_rms = rms(speech)
        noise_rms = rms(noise)
        desired_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
        scale = desired_noise_rms / (noise_rms + 1e-12)
        noise = noise * scale

    mix = speech + noise

    # Peak normalization to avoid clipping
    peak = float(np.max(np.abs(mix)))
    if peak > 0.99:
        mix = mix / peak * 0.99
    wavfile.write(out_path, sr, np.round(mix * 32767).astype(np.int16))
