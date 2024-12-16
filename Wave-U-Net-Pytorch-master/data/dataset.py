from scipy.signal import butter, lfilter
from sortedcontainers import SortedList
from torch.utils.data import Dataset
import os
import numpy as np
import glob
from data.utils import load, write_wav


def butter_lowpass_filter(data, cutoff_freq, sr, order=6):
    nyquist = 0.5 * sr
    b, a = butter(order, cutoff_freq / nyquist, btype='low', analog=False)
    filtered = lfilter(b, a, data, axis=-1)
    filtered = filtered.astype(np.float32)
    return filtered


class SeparationDataset(Dataset):
    def __init__(self, dataset, partition, instruments, sr, channels, shapes, random_hops, audio_transform=None):
        super(SeparationDataset, self).__init__()
        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.instruments = instruments
        self.overlap_factor = 0.5
        self.variation_factor = 0.25

        self.data = []
        self.snippet_indices = []

        for example in dataset.get(partition, []):
            mix_audio, _ = load(example["mix"], mono=True)
            piano_source_audio, _ = load(example["piano_source"], mono=True)
            source_audios = [load(example[src], mono=True)[0] for src in instruments]
            source_audios = np.concatenate(source_audios, axis=0)

            # Downsample and align lengths
            mix_audio = butter_lowpass_filter(mix_audio[:, ::4], sr // 2 - 1000, sr)
            piano_source_audio = butter_lowpass_filter(piano_source_audio[:, ::4], sr // 2 - 1000, sr)
            source_audios = butter_lowpass_filter(source_audios[:, ::4], sr // 2 - 1000, sr)

            min_length = min(mix_audio.shape[1], piano_source_audio.shape[1], source_audios.shape[1])
            mix_audio = mix_audio[:, :min_length]
            piano_source_audio = piano_source_audio[:, :min_length]
            source_audios = source_audios[:, :min_length]

            self.data.append({
                "mix": mix_audio,
                "piano_source": piano_source_audio,
                "targets": source_audios,
                "length": min_length,
            })

            # Calculate base indices with overlap
            input_frames = shapes["input_frames"]
            output_frames = shapes["output_frames"]
            step_size = int(output_frames * (1 - self.overlap_factor))  # Step size for 50% overlap
            num_snippets = (min_length - input_frames) // step_size + 1
            base_indices = np.arange(num_snippets) * step_size

            # Add variation to the base indices
            max_shift = int(output_frames * self.variation_factor)  # 25% variation
            random_shifts = np.random.randint(-max_shift, max_shift + 1, size=base_indices.shape)
            adjusted_indices = base_indices + random_shifts

            # Ensure indices are valid
            adjusted_indices = np.clip(adjusted_indices, 0, min_length - input_frames)
            self.snippet_indices.append(adjusted_indices)

        self.cumulative_snippets = np.cumsum([len(indices) for indices in self.snippet_indices])

    def __len__(self):
        return self.cumulative_snippets[-1]

    def __getitem__(self, index):
        # Find the track corresponding to this global index
        track_idx = np.searchsorted(self.cumulative_snippets, index, side="right")
        if track_idx > 0:
            index -= self.cumulative_snippets[track_idx - 1]

        item = self.data[track_idx]
        snippet_index = self.snippet_indices[track_idx][index]
        input_frames = self.shapes["input_frames"]

        start_pos = snippet_index
        end_pos = start_pos + input_frames

        mix_audio = item["mix"][:, start_pos:end_pos]
        piano_source_audio = item["piano_source"][:, start_pos:end_pos]
        targets = item["targets"][:, start_pos:end_pos]

        audio = np.concatenate((mix_audio, piano_source_audio), axis=0)

        if self.audio_transform is not None:
            audio, targets = self.audio_transform(audio, targets)

        return audio, targets

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