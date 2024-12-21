import torch
from scipy.signal import butter, lfilter
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import torch.nn.functional as F
import soundfile as sf


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

        voice_filename = "voice.wav"
        piano_speaker_bleed_filename = "piano_speaker_bleed.wav"
        piano_source_filename = "piano_source.wav"

        for folder in glob.glob(os.path.join(path, '*')):
            voice_path, piano_speaker_bleed_path, piano_source_path = (
                os.path.join(folder, filename)
                for filename in [voice_filename, piano_speaker_bleed_filename, piano_source_filename]
            )

            voice_waveform, piano_speaker_bleed_waveform, piano_source_waveform = (
                self.__class__.downsampled_waveform(path) for path in [voice_path, piano_speaker_bleed_path, piano_source_path]
            )

            min_length = min(
                voice_waveform.shape[0],
                piano_speaker_bleed_waveform.shape[0],
                piano_source_waveform.shape[0],
            )

            voice_waveform, piano_speaker_bleed_waveform, piano_source_waveform = (
                waveform[:min_length]
                for waveform in [voice_waveform, piano_speaker_bleed_waveform, piano_source_waveform]
            )

            self.data.append({
                "voice_waveform": voice_waveform,
                "piano_speaker_bleed_waveform": piano_speaker_bleed_waveform,
                "piano_source_waveform": piano_source_waveform,
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

        voice_waveform, piano_bleed_waveform, piano_source_audio = (
            F.pad(
                torch.tensor(item[key][start_pos:end_pos].astype(np.float32)),
                (pad_front, pad_back),
                'constant',
                0.0
            )
            for key in ["voice_waveform", "piano_speaker_bleed_waveform", "piano_source_waveform"]
        )

        mix_audio = voice_waveform + piano_bleed_waveform
        peak = torch.max(torch.abs(mix_audio))

        if peak > 0:
            scale = 1.0 / peak
            mix_audio = mix_audio * scale
            voice_waveform = voice_waveform * scale
            piano_bleed_waveform = piano_bleed_waveform * scale
            piano_source_audio = piano_source_audio * scale
        else:
            scale = 1.0

        mix_audio[self.output_end:] = 0
        targets = voice_waveform[self.output_start: self.output_end]

        return mix_audio, piano_source_audio, targets

    @staticmethod
    def downsampled_waveform(path):
        return SeparationDataset.lowpass(SeparationDataset.get_waveform(path))[::4]

    @staticmethod
    def lowpass(waveform, cutoff_freq=11000, sr=48000, order=6):
        nyquist = 0.5 * sr
        b, a = butter(order, cutoff_freq / nyquist, btype='low', analog=False)
        filtered = lfilter(b, a, waveform, axis=-1)
        filtered = filtered.astype(np.float32)
        return filtered

    @staticmethod
    def get_waveform(path):
        audio, _ = sf.read(path)
        waveform = np.squeeze(audio).astype("float32")
        return torch.tensor(waveform, dtype=torch.float32)

    @staticmethod
    def normalize(waveform):
        peak = torch.max(torch.abs(waveform))
        return waveform / peak if peak > 0 else waveform