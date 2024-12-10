import argparse
import os
import data.utils
import model.utils as model_utils
from model.waveunet import Waveunet
import numpy as np
import torch


def main(args):
    # MODEL
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels * 2, num_features, args.channels, kernel_size=args.kernel_size, target_output_size=target_outputs, depth=args.depth, strides=args.strides)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print("Loading model from checkpoint " + str(args.load_model))
    state = model_utils.load_model(model, None, args.load_model, args.cuda)
    print('Step', state['step'])

    preds = predict_song(args, args.input, args.hint, model)

    output_folder = os.path.dirname(args.input) if args.output is None else args.output

    data.utils.write_wav(os.path.join(output_folder, os.path.basename(args.input) + "_" + "inst" + ".wav"), preds, args.sr)


def predict_song(args, audio_path, hint_path, model):
    model.eval()

    mix_audio, mix_sr = data.utils.load(audio_path, sr=None, mono=True) # Load mixture in original sampling rate (mono)
    mix_channels, mix_len = mix_audio.shape

    hint_audio, hint_sr = data.utils.load(hint_path, sr=None, mono=True) # Load hint in original sampling rate (mono)

    max_len = max(mix_audio.shape[1], hint_audio.shape[1])

    if mix_audio.shape[1] < max_len:
        mix_audio = np.pad(mix_audio, ((0, 0), (0, max_len - mix_audio.shape[1])), mode='constant')

    if hint_audio.shape[1] < max_len:
        hint_audio = np.pad(hint_audio, ((0, 0), (0, max_len - hint_audio.shape[1])), mode='constant')

    if mix_audio.ndim != 2 or mix_audio.shape[0] != args.channels: # Ensure mixture shape [channels, frames] matches args.channels
        raise ValueError(f"Expected mixture audio shape [{args.channels}, frames], but got {mix_audio.shape}")

    if hint_audio.ndim != 2 or hint_audio.shape[0] != 1: # Ensure hint is also single-channel and same length as mixture after resampling
        raise ValueError(f"Expected hint audio shape [1, frames], but got {hint_audio.shape}")
    if hint_audio.shape[1] != mix_audio.shape[1]:
        raise ValueError("Hint and mixture must have the same length after resampling")

    output_shift = model.shapes["output_frames"] # Pad input if not divisible by frame shift
    pad_back = mix_audio.shape[1] % output_shift
    pad_back = 0 if pad_back == 0 else output_shift - pad_back

    if pad_back > 0:
        mix_audio = np.pad(mix_audio, [(0, 0), (0, pad_back)], mode="constant", constant_values=0.0)
        hint_audio = np.pad(hint_audio, [(0, 0), (0, pad_back)], mode="constant", constant_values=0.0)

    target_outputs = mix_audio.shape[1]

    outputs = np.zeros([1, mix_audio.shape[1]], dtype=np.float32) # Initialize output array (model outputs 1 channel)

    pad_front_context = model.shapes["output_start_frame"] # Pad mixture and hint for boundary predictions
    pad_back_context = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    mix_audio_padded = np.pad(mix_audio, [(0, 0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)
    hint_audio_padded = np.pad(hint_audio, [(0, 0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)

    with torch.no_grad():
        for target_start_pos in range(0, target_outputs, model.shapes["output_frames"]):
            input_start = target_start_pos
            input_end = target_start_pos + model.shapes["input_frames"]
            curr_mix = mix_audio_padded[:, input_start:input_end]  # [1, frames] # Extract mixture chunk
            curr_hint = hint_audio_padded[:, input_start:input_end] # [1, frames] # Extract hint chunk
            curr_input = np.concatenate([curr_mix, curr_hint], axis=0)  # [2, frames] # Concatenate mixture and hint along the channel dimension
            curr_input_tensor = torch.from_numpy(curr_input).unsqueeze(0).float()  # [1, 2, frames]
            out_tensor = model(curr_input_tensor)  # [1, 1, frames]
            out_np = out_tensor.squeeze(0).cpu().numpy()  # [1, frames]

            if out_np.shape[0] != 1:
                raise ValueError(f"Expected model to output 1 channel, but got {out_np.shape[0]} channels.")

            outputs[:, target_start_pos:target_start_pos + model.shapes["output_frames"]] = out_np

    outputs = outputs[:, :mix_len] # Crop output to original length
    outputs = np.asfortranarray(outputs) # [1, frames]
    outputs = outputs.squeeze(0)  # [frames] # Convert to 1D array since it's single-channel

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, nargs='+', default=["voice"],
                        help="List of instruments to separate (default: \"bass drums other vocals\")")
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--load_model', type=str, default='/Users/julian/Downloads/checkpoint_70518',
                        help='Reload a previously trained model')
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=48000,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=1,
                        help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=2.0,
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=1,
                        help="Train separate model for each source (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")
    parser.add_argument('--input', type=str, default="/Volumes/SANDISK/SampleMix.wav",
                        help="Path to input mixture to be separated")
    parser.add_argument('--hint', type=str,
                        default="/Volumes/SANDISK/Piano Bleed Remover Training Data Piano Source/YOURS - STEMS - VOX LEADS.20_1.wav",
                        help="Path to input hint to be separated")
    parser.add_argument('--output', type=str, default=None, help="Output path (same folder as input path if not set)")

    args = parser.parse_args()

    main(args)
