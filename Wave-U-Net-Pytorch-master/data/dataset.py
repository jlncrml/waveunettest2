import os

import h5py
from sortedcontainers import SortedList
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import numpy as np
import glob

from data.utils import load, write_wav


class SeparationDataset(Dataset):
    def __init__(self, dataset, partition, instruments, sr, channels, shapes,
                 random_hops, hdf_dir, audio_transform=None, in_memory=False):
        '''
        Initializes a source separation dataset
        '''
        super(SeparationDataset, self).__init__()

        self.hdf_dataset = None  # Do not open the HDF5 file here
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")
        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.in_memory = in_memory
        self.instruments = instruments

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
                    mix_audio, _ = load(example["mix"], sr=self.sr, mono=(self.channels == 1))

                    # Load piano source (the hint)
                    piano_source_audio, _ = load(example["piano_source"], sr=self.sr, mono=(self.channels == 1))

                    # Load source audios (targets)
                    source_audios = []
                    for source in instruments:
                        source_audio, _ = load(example[source], sr=self.sr, mono=(self.channels == 1))
                        source_audios.append(source_audio)
                    source_audios = np.concatenate(source_audios, axis=0)

                    # Ensure all audio arrays have the same length
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
        return self.length if hasattr(self, 'length') else 0

    def __getitem__(self, index):
        if self.length == 0:
            raise IndexError("Cannot get item from an empty dataset.")

        # Open HDF5 file in the worker process
        if self.hdf_dataset is None:
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r')

        # Determine song key and index within the song
        audio_idx = self.start_pos.bisect_right(index)
        if audio_idx > 0:
            index = index - self.start_pos[audio_idx - 1]
        song_key = str(audio_idx)

        # Check length of audio signal
        if song_key not in self.hdf_dataset:
            raise KeyError(f"Song index {audio_idx} not found in HDF5 file.")

        audio_length = self.hdf_dataset[song_key].attrs["length"]
        target_length = self.hdf_dataset[song_key].attrs["target_length"]

        # Determine position where to start targets
        if self.random_hops:
            start_target_pos = np.random.randint(0, max(target_length - self.shapes["output_frames"] + 1, 1))
        else:
            # Map item index to sample position within song
            start_target_pos = index * self.shapes["output_frames"]

        # READ INPUTS
        # Check front padding
        start_pos = start_target_pos - self.shapes["output_start_frame"]
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # Read mix_audio
        mix_audio = self.hdf_dataset[song_key]["inputs"]["mix"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            mix_audio = np.pad(mix_audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        # Read piano_source_audio
        piano_source_audio = self.hdf_dataset[song_key]["inputs"]["piano_source"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            piano_source_audio = np.pad(piano_source_audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        # Stack mix_audio and piano_source_audio along the channel dimension
        audio = np.concatenate((mix_audio, piano_source_audio), axis=0)  # Shape: [channels * 2, samples]

        # Read targets (the true outputs you want the model to predict)
        targets_data = self.hdf_dataset[song_key]["targets"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            targets_data = np.pad(targets_data, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        # Create a dictionary of targets for each instrument
        targets = {inst: targets_data[idx * self.channels:(idx + 1) * self.channels]
                   for idx, inst in enumerate(self.instruments)}

        # Apply audio transformations (if any)
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

    np.random.seed(1337)  # Ensure that partitioning is always the same on each run
    train_size = int(len(train_val_list) * 0.8)  # Adjust the percentage as needed
    if train_size == 0:
        train_size = 1  # Ensure at least one training sample
    train_list = np.random.choice(train_val_list, train_size, replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]

    # Uncomment the line below to debug whether partitioning is deterministic
    # print("First training song: " + str(train_list[0]))
    return {"train": train_list, "val": val_list, "test": test_list}