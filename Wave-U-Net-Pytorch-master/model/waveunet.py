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

        if self.transpose:
            self.filter = nn.ConvTranspose1d(n_inputs, n_outputs, self.kernel_size, stride)
        else:
            self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)

        assert(n_outputs % NORM_CHANNELS == 0)

        self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)

    def forward(self, x):
        return F.relu(self.norm((self.filter(x))))

    def get_input_size(self, output_size):
        if not self.transpose:
            curr_size = (output_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        curr_size = curr_size + self.kernel_size - 1 # o = i + p - k + 1

        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0) # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1

        assert(curr_size > 0)

        return curr_size

    def get_output_size(self, input_size):
        if self.transpose:
            assert input_size > 1
            curr_size = (input_size - 1) * self.stride + self.kernel_size
        else:
            curr_size = input_size
            curr_size = curr_size - self.kernel_size + 1
            assert ((curr_size - 1) % self.stride == 0)
            curr_size = ((curr_size - 1) // self.stride) + 1

        assert curr_size > 0

        return curr_size


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

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)
        curr_size = self.pre_shortcut_conv.get_output_size(curr_size)
        curr_size = self.post_shortcut_conv.get_output_size(curr_size)
        return curr_size


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

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)
        curr_size = self.post_shortcut_conv.get_input_size(curr_size)
        curr_size = self.pre_shortcut_conv.get_input_size(curr_size)
        return curr_size


class Waveunet(nn.Module):
    def __init__(self, num_channels, kernel_size, target_output_size, strides=2):
        super(Waveunet, self).__init__()
        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size = kernel_size

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

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size
        self.input_size, self.output_size = self.check_padding(target_output_size)
        assert ((self.input_size - self.output_size) % 2 == 0)
        self.input_frames = self.input_size
        self.output_frames = self.output_size


    def check_padding(self, target_output_size):
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)

            if out is not False:
                return out

            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        try:
            curr_size = bottleneck

            for block in self.upsampling_blocks: # Compute output size going forward through upsampling
                curr_size = block.get_output_size(curr_size)

            output_size = curr_size

            curr_size = bottleneck # Compute input size going backward through bottleneck and downsampling
            curr_size = self.bottleneck.get_input_size(curr_size)

            for block in reversed(self.downsampling_blocks):
                curr_size = block.get_input_size(curr_size)

            assert output_size >= target_output_size

            return curr_size, output_size

        except AssertionError:
            return False

    def forward(self, mix_audio, piano_source_audio):
        # mix_pad = (self.input_frames - self.output_frames) // 2
        # padded_mix_waveform = F.pad(mix_waveform, (mix_pad, mix_pad), 'constant', 0.0).squeeze(0)
        #
        # x = torch.cat((padded_mix_waveform.unsqueeze(1), piano_source_waveform.unsqueeze(1)), dim=1)

        x = torch.cat((mix_audio.unsqueeze(1), piano_source_audio.unsqueeze(1)), dim=1)

        curr_input_size = x.shape[-1]

        assert (curr_input_size == self.input_size)

        shortcuts = []
        out = x

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
