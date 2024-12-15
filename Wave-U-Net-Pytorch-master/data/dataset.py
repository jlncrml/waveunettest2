import os

import h5py
from scipy.signal import butter, lfilter
from sortedcontainers import SortedList
from torch.utils.data import Dataset
from tqdm import tqdm
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
    def __init__(self, dataset, partition, instruments, sr, channels, shapes,
                 random_hops, hdf_dir, audio_transform=None, in_memory=False,
                 cutoff_freq=10000, filter_order=6):
        super(SeparationDataset, self).__init__()
        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")
        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.in_memory = in_memory
        self.instruments = instruments
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order

        # PREPARE HDF FILE

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_dir):
            # Create HDF file
            with h5py.File(self.hdf_dir, "w") as f:
                f.attrs["sr"] = sr
                f.attrs["channels"] = channels
                f.attrs["instruments"] = instruments

                print("Adding audio files to dataset (preprocessing)...")
                num_examples = len(dataset.get(partition, []))
                if num_examples == 0:
                    raise ValueError(f"No data found for partition '{partition}'. Please check your dataset.")

                for idx, example in enumerate(tqdm(dataset[partition])):
                    # Load mix
                    mix_audio, _ = load(example["mix"], mono=(self.channels == 1))

                    # Load piano source (the hint)
                    piano_source_audio, _ = load(example["piano_source"], mono=(self.channels == 1))

                    # Load source audios (targets)
                    source_audios = []
                    for source in instruments:
                        source_audio, _ = load(example[source], mono=(self.channels == 1))
                        source_audios.append(source_audio)
                    source_audios = np.concatenate(source_audios, axis=0)

                    # Apply low-pass filter here in the initializer
                    mix_audio = butter_lowpass_filter(mix_audio, self.cutoff_freq, self.sr, order=self.filter_order)
                    piano_source_audio = butter_lowpass_filter(piano_source_audio, self.cutoff_freq, self.sr,
                                                               order=self.filter_order)
                    source_audios = butter_lowpass_filter(source_audios, self.cutoff_freq, self.sr,
                                                          order=self.filter_order)

                    if self.sr == 24000:
                        mix_audio = mix_audio[:, ::2]
                        piano_source_audio = piano_source_audio[:, ::2]
                        source_audios = source_audios[:, ::2]

                    min_length = min(mix_audio.shape[1], piano_source_audio.shape[1], source_audios.shape[1])
                    mix_audio = mix_audio[:, :min_length]
                    piano_source_audio = piano_source_audio[:, :min_length]
                    source_audios = source_audios[:, :min_length]

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    # Store inputs: mix and piano_source
                    inputs_grp = grp.create_group("inputs")
                    inputs_grp.create_dataset("mix", data=mix_audio)
                    inputs_grp.create_dataset("piano_source", data=piano_source_audio)
                    # Store targets
                    grp.create_dataset("targets", data=source_audios)
                    grp.attrs["length"] = min_length
                    grp.attrs["target_length"] = min_length

        # Check HDF5 file attributes
        with h5py.File(self.hdf_dir, "r") as f:
            if f.attrs["sr"] != sr or \
                    f.attrs["channels"] != channels or \
                    list(f.attrs["instruments"]) != instruments:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate, channel count, or instruments are not as expected. Did you load an outdated HDF file?")

            # HDF FILE READY
            # SET SAMPLING POSITIONS

            # Go through HDF and collect lengths of all audio files
            num_songs = len(f)
            if num_songs == 0:
                raise ValueError(f"The HDF5 file '{self.hdf_dir}' is empty. Please check your dataset preparation.")

            lengths = []
            for song_idx in range(num_songs):
                song_key = str(song_idx)
                if song_key in f:
                    target_length = f[song_key].attrs["target_length"]
                    lengths.append(target_length)
                else:
                    print(f"Warning: Song index {song_idx} not found in HDF5 file.")

            if not lengths:
                raise ValueError("No valid song lengths found in HDF5 file.")

            print(shapes)

            # Calculate the number of starting positions
            lengths = [(l // self.shapes["output_frames"]) + 1 for l in lengths]

        if lengths:
            self.start_pos = SortedList(np.cumsum(lengths))
            self.length = self.start_pos[-1]
        else:
            self.start_pos = SortedList()
            self.length = 0
            print("Warning: No data found in HDF5 file. Dataset length set to 0.")

    def __len__(self):
        # Limit the dataset size to 10 samples
        return min(self.length if hasattr(self, 'length') else 0, 10000)

    def __getitem__(self, index):
        if self.length == 0:
            raise IndexError("Cannot get item from an empty dataset.")

        if self.hdf_dataset is None:
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r')

        audio_idx = self.start_pos.bisect_right(index)
        if audio_idx > 0:
            index = index - self.start_pos[audio_idx - 1]
        song_key = str(audio_idx)

        if song_key not in self.hdf_dataset:
            raise KeyError(f"Song index {audio_idx} not found in HDF5 file.")

        audio_length = self.hdf_dataset[song_key].attrs["length"]
        target_length = self.hdf_dataset[song_key].attrs["target_length"]

        if self.random_hops:
            start_target_pos = np.random.randint(0, max(target_length - self.shapes["output_frames"] + 1, 1))
        else:
            start_target_pos = index * self.shapes["output_frames"]

        start_pos = start_target_pos - self.shapes["output_start_frame"]
        if start_pos < 0:
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio_length:
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        mix_audio = self.hdf_dataset[song_key]["inputs"]["mix"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            mix_audio = np.pad(mix_audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        piano_source_audio = self.hdf_dataset[song_key]["inputs"]["piano_source"][:, start_pos:end_pos].astype(
            np.float32)
        if pad_front > 0 or pad_back > 0:
            piano_source_audio = np.pad(piano_source_audio, [(0, 0), (pad_front, pad_back)], mode="constant",
                                        constant_values=0.0)

        audio = np.concatenate((mix_audio, piano_source_audio), axis=0)

        targets_data = self.hdf_dataset[song_key]["targets"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            targets_data = np.pad(targets_data, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        targets = targets_data

        if self.audio_transform is not None:
            audio, targets = self.audio_transform(audio, targets)

        return audio, targets

    def __getstate__(self):
        state = self.__dict__.copy()
        # Do not pickle the hdf_dataset
        state['hdf_dataset'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize hdf_dataset in the worker
        self.hdf_dataset = None

    def __del__(self):
        # Ensure the HDF5 file is closed when the dataset is destroyed
        if self.hdf_dataset is not None:
            self.hdf_dataset.close()


def get_dataset(database_path):
    '''
    Retrieve audio file paths for your custom dataset
    :param database_path: Root directory of your dataset
    :return: list containing train and test samples, each sample containing all audio paths
    '''
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
            piano_source_path = os.path.join(track_folder, "piano_source.wav")  # New line
            mix_path = os.path.join(track_folder, "mix.wav")
            acc_path = piano_bleed_path

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
            example["piano_source"] = piano_source_path  # Add this line
            example["accompaniment"] = acc_path  # Accompaniment is piano_speaker_bleed

            samples.append(example)

        subsets.append(samples)

    # Return the dataset after processing all subsets
    return subsets


def get_dataset_folds(root_path, version="HQ"):
    dataset = get_dataset(root_path)

    train_val_list = dataset[0]
    test_list = dataset[1]

    # Warning
    # Limit to 10 samples in training/validation and test sets
    # train_val_list = train_val_list[:10]
    # test_list = test_list[:10]

    np.random.seed(1337)

    train_size = int(len(train_val_list) * 0.8)

    if train_size == 0:
        train_size = 1

    train_list = np.random.choice(train_val_list, train_size, replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]

    return {"train": train_list, "val": val_list, "test": test_list}