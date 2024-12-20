import torch
from scipy.signal import butter, lfilter
from sortedcontainers import SortedList
from tensorboard.summary.v1 import audio
from torch.utils.data import Dataset
import os
import numpy as np
import glob
from data.utils import load, write_wav
import torch.nn.functional as F


def butter_lowpass_filter(data, cutoff_freq, sr, order=6):
    nyquist = 0.5 * sr
    b, a = butter(order, cutoff_freq / nyquist, btype='low', analog=False)
    filtered = lfilter(b, a, data, axis=-1)
    filtered = filtered.astype(np.float32)
    return filtered


from torch.utils.data import Dataset
from sortedcontainers import SortedList  # Ensure this is imported
import torch
import torch.nn.functional as F
import numpy as np

class SeparationDataset(Dataset):
    def __init__(self, dataset, partition, instruments, sr, channels, input_frames, output_frames, random_hops):
        super(SeparationDataset, self).__init__()
        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.output_frames_start = (input_frames - self.output_frames) // 2
        self.output_frames_end = self.output_frames_start + self.output_frames
        self.instruments = instruments
        self.cutoff_freq = sr // 2 - 1000

        self.data = []
        num_examples = len(dataset.get(partition, []))
        if num_examples == 0:
            raise ValueError(f"No data found for partition '{partition}'.")

        excluded_samples = 0  # Counter for excluded samples

        for example in dataset[partition]:
            mix_audio, _ = load(example["mix"])
            piano_source_audio, _ = load(example["piano_source"])
            source_audios = []

            for source in instruments:
                source_audio, _ = load(example[source])
                source_audios.append(source_audio)

            source_audios = np.concatenate(source_audios, axis=0)

            # Apply filters
            mix_audio = butter_lowpass_filter(mix_audio, self.cutoff_freq, self.sr)
            piano_source_audio = butter_lowpass_filter(piano_source_audio, self.cutoff_freq, self.sr)
            source_audios = butter_lowpass_filter(source_audios, self.cutoff_freq, self.sr)

            # Downsample
            mix_audio = mix_audio[::4]
            piano_source_audio = piano_source_audio[::4]
            source_audios = source_audios[::4]

            # Trim to minimum length
            min_length = min(mix_audio.shape[0], piano_source_audio.shape[0], source_audios.shape[0])

            # Only include samples with sufficient length
            if min_length >= self.output_frames:
                mix_audio = mix_audio[:min_length]
                piano_source_audio = piano_source_audio[:min_length]
                source_audios = source_audios[:min_length]

                self.data.append({
                    "mix": mix_audio,
                    "piano_source": piano_source_audio,
                    "targets": source_audios,
                    "length": min_length,
                })
            else:
                excluded_samples += 1
                print(f"Excluded sample with min_length={min_length} < output_frames={self.output_frames}")

        # Calculate lengths only for valid samples
        lengths = [d["length"] // self.output_frames for d in self.data]

        if lengths:
            self.start_pos = SortedList(np.cumsum(lengths))
            self.length = self.start_pos[-1]
        else:
            self.start_pos = SortedList()
            self.length = 0

        print(f"Total samples: {num_examples}, Included: {len(self.data)}, Excluded: {excluded_samples}")

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def get_dataset(database_path):
    subsets = []

    for subset in ["train", "test"]:
        print("Loading " + subset + " set...")
        tracks = glob.glob(os.path.join(database_path, subset, "*"))
        samples = []

        # Go through tracks
        for track_folder in sorted(tracks):
            example = {}
            voice_path = os.path.join(track_folder, "voice.wav")
            piano_bleed_path = os.path.join(track_folder, "piano_speaker_bleed.wav")
            piano_source_path = os.path.join(track_folder, "piano_source.wav")
            mix_path = os.path.join(track_folder, "mix.wav")

            # Ensure the stem files exist
            if not os.path.exists(voice_path):
                print(f"Voice file not found: {voice_path}")
                continue
            if not os.path.exists(piano_bleed_path):
                print(f"Piano speaker bleed file not found: {piano_bleed_path}")
                continue
            if not os.path.exists(piano_source_path):  # New check
                print(f"Piano source file not found: {piano_source_path}")
                continue

            example["mix"] = mix_path
            example["voice"] = voice_path
            example["piano_speaker_bleed"] = piano_bleed_path
            example["piano_source"] = piano_source_path

            samples.append(example)

        subsets.append(samples)

    return subsets


def get_dataset_folds(root_path):
    dataset = get_dataset(root_path)

    train_val_list = dataset[0]
    test_list = dataset[1]

    np.random.seed(1337)

    train_size = int(len(train_val_list) * 0.8)

    if train_size == 0:
        train_size = 1

    train_list = np.random.choice(train_val_list, train_size, replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]

    return {"train": train_list, "val": val_list, "test": test_list}