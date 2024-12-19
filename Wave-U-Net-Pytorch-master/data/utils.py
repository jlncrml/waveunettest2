import librosa
import numpy as np
import soundfile
import torch


def random_amplify(mix, targets, shapes, min, max):
    # Since there's only one source, 'targets' is a single tensor, not a dict
    # The logic now:
    # residual = mix - targets
    # Apply random gain to residual and targets separately
    residual = mix[0] - targets
    residual *= np.random.uniform(min, max)
    targets = targets * np.random.uniform(min, max)
    mix[0] = residual + targets
    mix[0] = np.clip(mix, -1.0, 1.0)
    return mix, targets


def load(path, sr=48000, mono=True, mode="numpy", offset=0.0, duration=None):
    y, curr_sr = librosa.load(path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)
    return y, curr_sr


def write_wav(path, audio, sr):
    soundfile.write(path, audio.T, sr, "PCM_16")
