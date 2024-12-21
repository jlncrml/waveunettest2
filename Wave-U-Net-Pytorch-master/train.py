import argparse
import os
import time
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import model.utils as model_utils
import utils
from data.dataset import SeparationDataset, get_dataset
from data.dataset import get_dataset_folds
from model.waveunet import Waveunet
import torch
import torch.nn as nn


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

def main(args):
    num_features = [N_FEATURES*2**idx for idx in range(0, N_LEVELS)]

    model = Waveunet(
        num_features,
        kernel_size=KERNEL_SIZE,
        input_length=INPUT_LENGTH,
        output_length=OUTPUT_LENGTH,
        strides=STRIDES
    )

    if args.cuda:
        model = model_utils.DataParallel(model)
        model.cuda()

    dataset_data = get_dataset_folds(args.dataset_dir)

    train_data = SeparationDataset(
        dataset_data,
        "train",
        ["voice"],
        SAMPLE_RATE,
        1,
        model.input_length,
        model.output_length,
        True
    )

    val_data = SeparationDataset(
        dataset_data,
        "val",
        ["voice"],
        SAMPLE_RATE,
        1,
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

    criterion = nn.L1Loss()
    filtered_criterion = LastSamplesMAELoss()

    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)

    cycle_length = len(train_data) // (BATCH_SIZE * N_CYCLES)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda step: (
    #             MIN_LEARNING_RATE +
    #             0.5 * (LEARNING_RATE - MIN_LEARNING_RATE) *
    #             (1 + np.cos((step % cycle_length) / cycle_length * np.pi))
    #     )
    # )

    state = {"step": 0, "worse_epochs": 0, "epochs": 0, "best_loss": np.Inf}

    print('TRAINING START')

    while state["worse_epochs"] < PATIENCE:
        print("Training one epoch from iteration " + str(state["step"]))

        avg_time = 0.

        model.train()

        with tqdm(total=len(train_data) // BATCH_SIZE) as pbar:
            np.random.seed()
            for example_num, (mix_audio, piano_source_audio, targets) in enumerate(dataloader):
                if args.cuda:
                    mix_audio = mix_audio.cuda()
                    piano_source_audio = piano_source_audio.cuda()
                    targets = targets.cuda()

                t = time.time()

                utils.set_cyclic_lr(
                    optimizer,
                    example_num,
                    len(train_data) // BATCH_SIZE,
                    N_CYCLES,
                    MIN_LEARNING_RATE,
                    LEARNING_RATE
                )

                optimizer.zero_grad()

                out = model(mix_audio, piano_source_audio)

                loss = criterion(out, targets)

                loss.backward()
                optimizer.step()

                # current_lr = scheduler.get_last_lr()
                # print(f"Current LR: {current_lr}")

                # scheduler.step()

                state["step"] += 1

                t = time.time() - t  # Timing
                avg_time += (1.0 / (example_num + 1)) * (t - avg_time)

                pbar.update(1)


        val_loss, last_val_loss = validate(args, model, criterion, filtered_criterion, val_data) # VALIDATE

        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path

        state["epochs"] += 1

        model_utils.save_model(model, optimizer, state, checkpoint_path)


def validate(args, model, criterion1, criterion2, test_data):
    dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=N_WORKERS
    )

    model.eval()

    total_loss1 = 0.0
    total_loss2 = 0.0

    with torch.no_grad(), tqdm(total=len(test_data) // BATCH_SIZE) as pbar:
        for example_num, (mix_audio, piano_source_audio, target) in enumerate(dataloader):
            if args.cuda:
                mix_audio = mix_audio.cuda()
                piano_source_audio = piano_source_audio.cuda()
                target = target.cuda()

            out = model(mix_audio, piano_source_audio)

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
    parser.add_argument('--dataset_dir', type=str, default="/Volumes/SANDISK/WaveUNetTrainingData")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/waveunet')
    args = parser.parse_args()
    main(args)