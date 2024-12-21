import torch
import torch.nn as nn
from torch.nn import functional as F


def center_crop(x, target):
    if x is None or target is None: return x
    difference = (x.shape[-1] - target.shape[-1]) // 2
    if difference < 0: raise ArithmeticError
    return x if difference == 0 else x[..., difference:-difference].contiguous()


class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, transpose=False):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size

        NORM_CHANNELS = 8

        if self.transpose: self.filter = nn.ConvTranspose1d(n_inputs, n_outputs, self.kernel_size, stride)
        else: self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)

        assert(n_outputs % NORM_CHANNELS == 0)

        self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)

    def forward(self, x):
        return F.relu(self.norm((self.filter(x))))


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)
        self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, transpose=True)
        self.pre_shortcut_conv = ConvLayer(n_inputs, n_outputs, kernel_size, 1)
        self.post_shortcut_conv = ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1)

    def forward(self, x, shortcut):
        upsampled = self.upconv(x)
        upsampled = self.pre_shortcut_conv(upsampled)
        combined = center_crop(shortcut, upsampled)
        combined = self.post_shortcut_conv(torch.cat([combined, center_crop(upsampled, combined)], dim=1))
        return combined


class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pre_shortcut_conv = ConvLayer(n_inputs, n_shortcut, kernel_size, 1)
        self.post_shortcut_conv = ConvLayer(n_shortcut, n_outputs, kernel_size, 1)
        self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride)

    def forward(self, x):
        shortcut = x # PREPARING SHORTCUT FEATURES
        shortcut = self.pre_shortcut_conv(shortcut)
        out = shortcut # PREPARING FOR DOWNSAMPLING
        out = self.post_shortcut_conv(out)
        out = self.downconv(out) # DOWNSAMPLING
        return out, shortcut


class Waveunet(nn.Module):
    def __init__(self, num_channels, kernel_size, input_length, output_length, strides=2):
        super(Waveunet, self).__init__()
        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size = kernel_size
        self.input_length = input_length
        self.output_length = output_length

        assert (kernel_size % 2 == 1)

        self.downsampling_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            in_ch = 2 if i == 0 else num_channels[i]

            self.downsampling_blocks.append(
                DownsamplingBlock(in_ch, num_channels[i], num_channels[i + 1],kernel_size, strides)
            )

        self.bottleneck = ConvLayer(num_channels[-1], num_channels[-1], 1, 1)

        self.upsampling_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            self.upsampling_blocks.append(
                UpsamplingBlock(num_channels[-1 - i], num_channels[-2 - i], num_channels[-2 - i], kernel_size, strides)
            )

        self.output_conv = nn.Conv1d(num_channels[0], 1, 1)

    def forward(self, mix_audio, piano_source_audio):
        out = torch.cat((mix_audio.unsqueeze(1), piano_source_audio.unsqueeze(1)), dim=1)

        shortcuts = []

        for block in self.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)

        out = self.bottleneck(out)

        for idx, block in enumerate(self.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])

        out = self.output_conv(out) # Output

        if not self.training:
            out = out.clamp(min=-1.0, max=1.0)

        return torch.squeeze(out)