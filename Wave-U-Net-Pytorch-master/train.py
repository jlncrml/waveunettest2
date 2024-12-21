import argparse
import os
import time
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
import model.utils as model_utils
import utils
from data.dataset import SeparationDataset
from model.waveunet import Waveunet
import torch
import torch.nn as nn

INPUT_LENGTH = 29693
OUTPUT_LENGTH = 24237
STRIDES = 4
SAMPLE_RATE = 12000
N_LEVELS = 6
N_FEATURES = 32
N_WORKERS = 1
KERNEL_SIZE = 5
BATCH_SIZE = 16
N_CYCLES = 2
PATIENCE = 20
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 5e-5


class LastSamplesMAELoss(nn.Module):
    def __init__(self, n_samples=512):
        super(LastSamplesMAELoss, self).__init__()
        self.n_samples = n_samples
        self.mae_loss = nn.L1Loss()

    def forward(self, output, target):
        assert output.shape == target.shape, "Output and target must have the same shape."

        last_out = output[..., -self.n_samples:]
        last_tgt = target[..., -self.n_samples:]

        loss = self.mae_loss(last_out, last_tgt)
        return loss


def main(args):
    num_features = [N_FEATURES*2**idx for idx in range(0, N_LEVELS)]

    model = Waveunet(
        num_features,
        kernel_size=KERNEL_SIZE,
        input_length=INPUT_LENGTH,
        output_length=OUTPUT_LENGTH,
        strides=STRIDES
    )

    if torch.cuda.is_available():
        model = model_utils.DataParallel(model)
        model.cuda()

    train_data = SeparationDataset(
        os.path.join(args.dataset_dir, "train"),
        model.input_length,
        model.output_length,
        True
    )

    val_data = SeparationDataset(
        os.path.join(args.dataset_dir, "test"),
        model.input_length,
        model.output_length,
        False
    )

    dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_WORKERS,
        worker_init_fn=utils.worker_init_fn
    )

    training_criterion = nn.L1Loss()
    validation_criterion = LastSamplesMAELoss()

    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)

    cycle_length = (len(train_data) // BATCH_SIZE) // N_CYCLES

    def lr_lambda(step):
        curr_cycle = min(step // cycle_length, N_CYCLES - 1)
        curr_it = step - cycle_length * curr_cycle

        return (
                MIN_LEARNING_RATE
                + 0.5 * (LEARNING_RATE - MIN_LEARNING_RATE)
                * (1 + np.cos(float(curr_it) / float(cycle_length) * np.pi))
        ) / LEARNING_RATE

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

    state = {"step": 0, "n_worse_epochs": 0, "n_epochs": 0, "best_loss": np.Inf}

    while state["n_worse_epochs"] < PATIENCE:
        print("Epoch " + str(state["n_epochs"] + 1))

        avg_time = 0.

        model.train()

        with tqdm(total=len(train_data) // BATCH_SIZE) as progress_bar:
            np.random.seed()
            for example_num, (mix_audio, piano_source_audio, target) in enumerate(dataloader):
                if torch.cuda.is_available():
                    mix_audio, piano_source_audio, target = (t.cuda() for t in [mix_audio, piano_source_audio, target])

                optimizer.zero_grad()

                out = model(mix_audio, piano_source_audio)

                loss = training_criterion(out, target)

                loss.backward()
                optimizer.step()

                scheduler.step()

                state["step"] += 1

                progress_bar.update(1)

        validation_loss = validate(args, model, validation_criterion, val_data)

        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))

        if validation_loss >= state["best_loss"]:
            state["n_worse_epochs"] += 1
        else:
            state["n_worse_epochs"] = 0
            state["best_loss"] = validation_loss
            state["best_checkpoint"] = checkpoint_path

        state["n_epochs"] += 1

        model_utils.save_model(model, optimizer, state, checkpoint_path)


def validate(args, model, criterion, test_data):
    dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=N_WORKERS
    )

    model.eval()

    total_loss = 0.0

    with torch.no_grad(), tqdm(total=len(test_data) // BATCH_SIZE) as pbar:
        for example_num, (mix_audio, piano_source_audio, target) in enumerate(dataloader):
            if torch.cuda.is_available():
                mix_audio, piano_source_audio, target = (t.cuda() for t in [mix_audio, piano_source_audio, target])

            out = model(mix_audio, piano_source_audio)

            loss = criterion(out, target)
            loss_val = loss.item()

            count = example_num + 1
            total_loss += (1.0 / count) * (loss_val - total_loss)

            pbar.set_description(
                f"Validation Loss: {total_loss:.5f}"
            )

            pbar.update(1)

    return total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="/Volumes/SANDISK/WaveUNetTrainingData/")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/waveunet')
    args = parser.parse_args()
    main(args)