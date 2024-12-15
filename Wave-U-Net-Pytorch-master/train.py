import argparse
import os
import time
from functools import partial
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm
import model.utils as model_utils
import utils
from data.dataset import SeparationDataset, get_dataset
from data.dataset import get_dataset_folds
from data.utils import crop_targets, random_amplify
from model.waveunet import Waveunet
import logging
from predict import predict_song
logging.basicConfig(level=logging.DEBUG)
from scipy.signal import butter, lfilter
import torch
import torch.nn as nn

class LowPassMAELoss(nn.Module):
    def __init__(self, cutoff_freq, sample_rate, filter_order=6):
        super(LowPassMAELoss, self).__init__()
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.filter_order = filter_order
        self.mae_loss = nn.L1Loss()

        nyquist = 0.5 * sample_rate # Design the low-pass Butterworth filter
        self.b, self.a = butter(filter_order, cutoff_freq / nyquist, btype='low', analog=False)

    def forward(self, output, target):
        assert output.shape == target.shape, "Output and target must have the same shape."
        filtered_output = self.low_pass_filter(output) # Apply low-pass filter to both output and target
        filtered_target = self.low_pass_filter(target)
        loss = self.mae_loss(filtered_output, filtered_target) # Compute MAE loss on filtered signals
        return loss

    def low_pass_filter(self, signal):
        signal_np = signal.cpu().detach().numpy() # Convert to NumPy array for filtering
        filtered_signal = lfilter(self.b, self.a, signal_np, axis=-1) # Apply the filter along the time axis
        return torch.from_numpy(filtered_signal).to(signal.device) # Convert back to torch.Tensor


class LastSamplesMAELoss(nn.Module):
    def __init__(self, n_samples=2048):
        super(LastSamplesMAELoss, self).__init__()
        self.n_samples = n_samples
        self.mae_loss = nn.L1Loss()

    def forward(self, output, target):
        assert output.shape == target.shape, "Output and target must have the same shape."

        # Extract the last n_samples from each signal
        last_out = output[..., -self.n_samples:]
        last_tgt = target[..., -self.n_samples:]

        # Compute MAE on the last samples
        loss = self.mae_loss(last_out, last_tgt)
        return loss


def main(args):
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)

    instrument = "voice"

    model = Waveunet(num_features, kernel_size=args.kernel_size, target_output_size=target_outputs, depth=args.depth, strides=args.strides)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    writer = SummaryWriter(args.log_dir)

    dataset_data = get_dataset_folds(args.dataset_dir) # DATASET
    crop_func = partial(crop_targets, shapes=model.shapes)
    augment_func = partial(random_amplify, shapes=model.shapes, min=0.7, max=1.0)

    print(model.shapes)

    train_data = SeparationDataset(dataset_data, "train", args.sr, model.shapes, True, audio_transform=augment_func)
    val_data = SeparationDataset(dataset_data, "val", args.sr, model.shapes, False, audio_transform=crop_func)

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)

    criterion = nn.L1Loss()
    filtered_criterion = LastSamplesMAELoss()

    optimizer = Adam(params=model.parameters(), lr=args.lr)

    state = {"step": 0, "worse_epochs": 0, "epochs": 0, "best_loss": np.Inf}

    if args.load_model is not None:
        print("Continuing training model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)

    print('TRAINING START')

    while state["worse_epochs"] < args.patience:
        print("Training one epoch from iteration " + str(state["step"]))

        avg_time = 0.

        model.train()

        with tqdm(total=len(train_data) // args.batch_size) as pbar:
            np.random.seed()
            for example_num, (x, targets) in enumerate(dataloader):
                if args.cuda:
                    x = x.cuda()
                    targets = targets.cuda()

                t = time.time()

                utils.set_cyclic_lr(
                    optimizer,
                    example_num,
                    len(train_data) // args.batch_size,
                    args.cycles,
                    args.min_lr,
                    args.lr
                )

                writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])

                optimizer.zero_grad() # Zero gradients

                out = model(x) # Forward pass

                loss = criterion(out, targets) # Compute loss

                # print(loss)

                loss.backward()  # Backward pass and optimization
                optimizer.step()

                avg_loss = loss.item() # Record loss
                state["step"] += 1

                t = time.time() - t # Timing
                avg_time += (1.0 / (example_num + 1)) * (t - avg_time)

                writer.add_scalar("train_loss", avg_loss, state["step"])

                if example_num % args.example_freq == 0:
                    input_centre = torch.mean(
                        x[0, :, model.shapes["output_start_frame"]:model.shapes["output_end_frame"]],
                        dim=0
                    )
                    writer.add_audio("input", input_centre.cpu(), state["step"], sample_rate=args.sr)
                    writer.add_audio("pred", torch.mean(out[0].cpu(), dim=0), state["step"], sample_rate=args.sr)
                    writer.add_audio("target", torch.mean(targets[0].cpu(), dim=0), state["step"], sample_rate=args.sr)

                pbar.update(1)


        val_loss, last_val_loss = validate(args, model, criterion, filtered_criterion, val_data) # VALIDATE
        print("VALIDATION FINISHED: LOSS: " + str(val_loss) + "LAST 2048 LOSS: " + str(last_val_loss))
        writer.add_scalar("val_loss", val_loss, state["step"])

        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path

        state["epochs"] += 1
        print("Saving model...")
        model_utils.save_model(model, optimizer, state, checkpoint_path)

    writer.close()


def evaluate(args, dataset, model, instruments):
    perfs = list()
    model.eval()
    with torch.no_grad():
        for example in dataset:
            print("Evaluating " + example["mix"])

            # Load source references in their original sr and channel number
            target_sources = np.stack([data.utils.load(example[instrument], sr=None, mono=False)[0].T for instrument in instruments])

            # Predict using mixture
            pred_sources = predict_song(args, example["mix"], model)
            pred_sources = np.stack([pred_sources[key].T for key in instruments])

            # Evaluate
            SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(target_sources, pred_sources)
            song = {}
            for idx, name in enumerate(instruments):
                song[name] = {"SDR" : SDR[idx], "ISR" : ISR[idx], "SIR" : SIR[idx], "SAR" : SAR[idx]}
            perfs.append(song)

    return perfs


def validate(args, model, criterion1, criterion2, test_data):
    dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    model.eval()
    total_loss1 = 0.0
    total_loss2 = 0.0
    with torch.no_grad(), tqdm(total=len(test_data) // args.batch_size) as pbar:
        for example_num, (x, target) in enumerate(dataloader):
            if args.cuda:
                x = x.cuda()
                target = target.cuda()

            out = model(x)

            loss1 = criterion1(out, target)
            loss1_val = loss1.item()

            loss2 = criterion2(out, target)
            loss2_val = loss2.item()

            count = example_num + 1
            total_loss1 += (1.0 / count) * (loss1_val - total_loss1)
            total_loss2 += (1.0 / count) * (loss2_val - total_loss2)

            pbar.set_description(
                f"Avg MAE: {total_loss1:.5f}, Avg Last MAE: {total_loss2:.5f}"
            )
            pbar.update(1)

    return total_loss1, total_loss2 # Return both averaged losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--features', type=int, default=32)
    parser.add_argument('--log_dir', type=str, default='logs/waveunet')
    parser.add_argument('--dataset_dir', type=str, default="/Volumes/SANDISK/WaveUNetTrainingData")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/waveunet')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--levels', type=int, default=6)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--sr', type=int, default=48000)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--output_size', type=float, default=2.0)
    parser.add_argument('--strides', type=int, default=4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--example_freq', type=int, default=200)
    parser.add_argument('--res', type=str, default="learned")
    parser.add_argument('--feature_growth', type=str, default="double")
    args = parser.parse_args()
    main(args)
