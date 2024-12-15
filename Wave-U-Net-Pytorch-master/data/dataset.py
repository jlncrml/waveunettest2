from scipy.signal import butter, lfilter
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import numpy as np
import glob
from data.utils import load, write_wav
from random import sample


def butter_lowpass_filter(data, cutoff_freq, sr, order=6):
    nyquist = 0.5 * sr
    b, a = butter(order, cutoff_freq / nyquist, btype='low', analog=False)
    filtered = lfilter(b, a, data, axis=-1)
    filtered = filtered.astype(np.float32)
    return filtered


class SeparationDataset(Dataset):
    def __init__(self, dataset, partition, sr, shapes, random_hops, audio_transform=None, cutoff_freq=5500,
                 filter_order=6):
        super().__init__()
        self.sr = sr
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        self.random_hops = random_hops

        print("Loading and preprocessing audio files...")
        self.songs = []  # Will store tuples of (mix, piano_source, voice, length)
        num_examples = len(dataset.get(partition, []))
        if num_examples == 0:
            raise ValueError(f"No data found for partition '{partition}'. Please check your dataset.")

        for example in tqdm(dataset[partition]):
            mix_path, piano_source_path, voice_path = example

            mix_audio, piano_source_audio, voice_audio = map(
                lambda path: butter_lowpass_filter(load(path, mono=True)[0], self.cutoff_freq, self.sr,
                                                   order=self.filter_order),
                (mix_path, piano_source_path, voice_path)
            )

            if self.sr == 24000:
                mix_audio, piano_source_audio, voice_audio = map(
                    lambda audio: audio[:, ::2],
                    (mix_audio, piano_source_audio, voice_audio)
                )
            elif self.sr == 12000:
                mix_audio, piano_source_audio, voice_audio = map(
                    lambda audio: audio[:, ::4],
                    (mix_audio, piano_source_audio, voice_audio)
                )

            min_length = min(mix_audio.shape[1], piano_source_audio.shape[1], voice_audio.shape[1])
            mix_audio, piano_source_audio, voice_audio = (
                audio[:, :min_length] for audio in (mix_audio, piano_source_audio, voice_audio)
            )

            self.songs.append((mix_audio, piano_source_audio, voice_audio, min_length))

        self.snippet_length = self.shapes["input_frames"]
        self.step_size = self.snippet_length // 2

        self.per_song_snippet_count = [
            ((length - self.snippet_length) // self.step_size) + 1 if length >= self.snippet_length else 0
            for _, _, _, length in self.songs
        ]

        self.length = sum(self.per_song_snippet_count)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.length == 0:
            raise IndexError("Cannot get item from an empty dataset.")

        snippet_count_so_far = 0
        for song_idx, count in enumerate(self.per_song_snippet_count):
            if snippet_count_so_far + count > index:
                local_index = index - snippet_count_so_far
                break
            snippet_count_so_far += count
        else:
            raise IndexError("Index out of range")

        mix_audio, piano_source_audio, targets_data, length = self.songs[song_idx]

        base_start = local_index * self.step_size

        if self.random_hops:
            max_offset = self.snippet_length // 4
            offset = np.random.randint(-max_offset, max_offset + 1)
            start_pos = max(0, min(base_start + offset, length - self.snippet_length))
        else:
            start_pos = base_start

        mix_snippet = mix_audio[:, start_pos:start_pos + self.snippet_length].astype(np.float32)
        piano_snippet = piano_source_audio[:, start_pos:start_pos + self.snippet_length].astype(np.float32)
        targets_snippet = targets_data[:, start_pos:start_pos + self.snippet_length].astype(np.float32)

        audio = np.concatenate((mix_snippet, piano_snippet), axis=0)
        targets = targets_snippet

        if self.audio_transform is not None:
            audio, targets = self.audio_transform(audio, targets)

        return audio, targets

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        pass


def get_dataset(database_path):
    print("Loading train set...")
    track_folders = sorted(glob.glob(os.path.join(database_path, "train", "*")))
    samples = [
        (os.path.join(folder, "mix.wav"),
         os.path.join(folder, "piano_source.wav"),
         os.path.join(folder, "voice.wav"))
        for folder in track_folders
        if all(os.path.exists(os.path.join(folder, filename)) for filename in ["mix.wav", "piano_source.wav", "voice.wav"])
    ]
    return samples


def get_dataset_folds(root_path):
    dataset = get_dataset(root_path)  # Returns a flat list of tuples

    np.random.seed(1337)

    train_size = int(len(dataset) * 0.8)
    if train_size == 0:
        train_size = 1

    train_list = sample(dataset, train_size)

    val_list = [elem for elem in dataset if elem not in train_list]

    return {"train": train_list, "val": val_list}