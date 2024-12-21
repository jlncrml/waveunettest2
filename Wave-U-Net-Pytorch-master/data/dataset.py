import torch
from scipy.signal import butter, lfilter
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import torch.nn.functional as F
import soundfile as sf


def butter_lowpass_filter(data, cutoff_freq=11000, sr=48000, order=6):
    nyquist = 0.5 * sr
    b, a = butter(order, cutoff_freq / nyquist, btype='low', analog=False)
    filtered = lfilter(b, a, data, axis=-1)
    filtered = filtered.astype(np.float32)
    return filtered


def get_waveform(path):
    audio, _ = sf.read(path)
    waveform = np.squeeze(audio).astype("float32")
    return torch.tensor(waveform, dtype=torch.float32)


class SeparationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        input_length,
        output_length,
        validate=False
    ):
        super(SeparationDataset, self).__init__()
        self.validate = validate
        self.input_length = input_length
        self.output_length = output_length
        self.output_start = (input_length - output_length) // 2
        self.output_end = self.output_start + output_length
        self.data = []

        for folder in glob.glob(os.path.join(path, '*')):
            voice_path = os.path.join(folder, "voice.wav")
            piano_speaker_bleed_path = os.path.join(folder, "piano_speaker_bleed.wav")
            piano_source_path = os.path.join(folder, "piano_source.wav")
            mix_path = os.path.join(folder, "mix.wav")

            voice_waveform = get_waveform(voice_path)
            piano_speaker_bleed_waveform = get_waveform(piano_speaker_bleed_path)
            piano_source_waveform = get_waveform(piano_source_path)
            mix_waveform = get_waveform(mix_path)

            filtered_voice_waveform = butter_lowpass_filter(voice_waveform)
            filtered_piano_speaker_bleed_waveform = butter_lowpass_filter(piano_speaker_bleed_waveform)
            filtered_piano_source_waveform = butter_lowpass_filter(piano_source_waveform)
            filtered_mix_waveform = butter_lowpass_filter(mix_waveform)

            downsampled_voice_waveform = filtered_voice_waveform[::4]
            downsampled_piano_speaker_bleed_waveform = filtered_piano_speaker_bleed_waveform[::4]
            downsampled_piano_source_waveform = filtered_piano_source_waveform[::4]
            downsampled_mix_waveform = filtered_mix_waveform[::4]

            min_length = min(
                downsampled_voice_waveform.shape[0],
                downsampled_piano_speaker_bleed_waveform.shape[0],
                downsampled_piano_source_waveform.shape[0],
                downsampled_mix_waveform.shape[0],
            )

            downsampled_voice_waveform = downsampled_voice_waveform[:min_length]
            downsampled_piano_speaker_bleed_waveform = downsampled_piano_speaker_bleed_waveform[:min_length]
            downsampled_piano_source_waveform = downsampled_piano_source_waveform[:min_length]
            downsampled_mix_waveform = downsampled_mix_waveform[:min_length]

            self.data.append({
                "voice_waveform": downsampled_voice_waveform,
                "piano_speaker_bleed_waveform": downsampled_piano_speaker_bleed_waveform,
                "piano_source_waveform": downsampled_piano_source_waveform,
                "mix_waveform": downsampled_mix_waveform,
                "length": min_length
            })

        lengths = [((d["length"] // self.output_length) + 1) for d in self.data]

        self.snippet_mapping = [
            (file_idx, snippet_idx)
            for file_idx, file in enumerate(self.data)
            for snippet_idx in range(lengths[file_idx])
        ]

        self.length = len(self.snippet_mapping)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        file_idx, snippet_idx = self.snippet_mapping[index]

        item = self.data[file_idx]

        if self.validate:
            start_target_pos = np.random.randint(0, max(item["length"] - self.output_length + 1, 1))
        else:
            start_target_pos = snippet_idx * self.output_length

        start_pos = start_target_pos - self.output_start
        end_pos = start_target_pos - self.output_start + self.input_length

        pad_front = max(-start_pos, 0)
        start_pos = max(start_pos, 0)

        pad_back = max(end_pos - item["length"], 0)
        end_pos = min(end_pos, item["length"])

        mix_audio = torch.tensor(item["mix_waveform"][start_pos:end_pos].astype(np.float32))
        mix_audio = F.pad(mix_audio.unsqueeze(0), (pad_front, pad_back), 'constant', 0.0).squeeze(0)

        piano_source_audio = torch.tensor(item["piano_source_waveform"][start_pos:end_pos].astype(np.float32))
        piano_source_audio = F.pad(piano_source_audio.unsqueeze(0), (pad_front, pad_back), 'constant', 0.0).squeeze(0)

        mix_audio[self.output_end:] = 0

        targets_data = torch.tensor(item["voice_waveform"][start_pos:end_pos].astype(np.float32))
        targets_data = F.pad(targets_data.unsqueeze(0), (pad_front, pad_back), 'constant', 0.0).squeeze(0)
        targets = targets_data[self.output_start:self.output_end]

        return mix_audio, piano_source_audio, targets

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)