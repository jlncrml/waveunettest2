import musdb
import os
import numpy as np
import glob

from data.utils import load, write_wav


def get_musdbhq(database_path):
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
            mix_path = os.path.join(track_folder, "mix.wav")
            acc_path = piano_bleed_path  # Accompaniment is piano_speaker_bleed

            # Ensure the stem files exist
            if not os.path.exists(voice_path):
                print(f"Voice file not found: {voice_path}")
                continue
            if not os.path.exists(piano_bleed_path):
                print(f"Piano speaker bleed file not found: {piano_bleed_path}")
                continue

            # Create mix.wav if it doesn't exist
            if not os.path.exists(mix_path):
                print(f"Creating mix for track: {track_folder}")
                # Load audio files
                voice_audio, sr = load(voice_path, sr=None, mono=False)
                piano_audio, sr_piano = load(piano_bleed_path, sr=None, mono=False)

                # Check if sampling rates are the same
                if sr != sr_piano:
                    print(f"Sampling rates do not match for track {track_folder}.")
                    continue

                # Ensure both audio arrays have the same shape
                if voice_audio.shape != piano_audio.shape:
                    print(f"Audio shape mismatch in {track_folder}.")
                    continue

                # Sum the audio files
                mix_audio = voice_audio + piano_audio

                # Clip to avoid clipping issues
                mix_audio = np.clip(mix_audio, -1.0, 1.0)

                # Write the mixed audio to mix.wav
                write_wav(mix_path, mix_audio.T, sr)  # Transpose if necessary

            else:
                print(f"Mix file already exists: {mix_path}")

            # Store paths in the example dictionary
            example["mix"] = mix_path
            example["voice"] = voice_path
            example["piano_speaker_bleed"] = piano_bleed_path
            example["accompaniment"] = acc_path  # Accompaniment is piano_speaker_bleed

            samples.append(example)

        subsets.append(samples)

    return subsets

def get_musdb(database_path):
    '''
    Retrieve audio file paths for your custom dataset
    :param database_path: Root directory of your dataset
    :return: list containing train and test samples, each sample containing all audio paths
    '''
    # Since musdb.DB is specific to the MUSDB dataset, we'll replicate the functionality
    # for your custom dataset in this function as well.

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
            mix_path = os.path.join(track_folder, "mix.wav")
            acc_path = piano_bleed_path  # Accompaniment is piano_speaker_bleed

            # Ensure the stem files exist
            if not os.path.exists(voice_path):
                print(f"Voice file not found: {voice_path}")
                continue
            if not os.path.exists(piano_bleed_path):
                print(f"Piano speaker bleed file not found: {piano_bleed_path}")
                continue

            # Create mix.wav if it doesn't exist
            if not os.path.exists(mix_path):
                print(f"Creating mix for track: {track_folder}")
                # Load audio files
                voice_audio, sr = load(voice_path, sr=None, mono=False)
                piano_audio, sr_piano = load(piano_bleed_path, sr=None, mono=False)

                # Check if sampling rates are the same
                if sr != sr_piano:
                    print(f"Sampling rates do not match for track {track_folder}.")
                    continue

                # Ensure both audio arrays have the same shape
                if voice_audio.shape != piano_audio.shape:
                    print(f"Audio shape mismatch in {track_folder}.")
                    continue

                # Sum the audio files
                mix_audio = voice_audio + piano_audio

                # Clip to avoid clipping issues
                mix_audio = np.clip(mix_audio, -1.0, 1.0)

                # Write the mixed audio to mix.wav
                write_wav(mix_path, mix_audio.T, sr)  # Transpose if necessary

            else:
                print(f"Mix file already exists: {mix_path}")

            # Store paths in the example dictionary
            example["mix"] = mix_path
            example["voice"] = voice_path
            example["piano_speaker_bleed"] = piano_bleed_path
            example["accompaniment"] = acc_path  # Accompaniment is piano_speaker_bleed

            samples.append(example)

        subsets.append(samples)

    return subsets

def get_musdb_folds(root_path, version="HQ"):
    if version == "HQ":
        dataset = get_musdbhq(root_path)
    else:
        dataset = get_musdb(root_path)
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