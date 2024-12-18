import torch
import torch.nn as nn
from torch.nn import functional as F

def center_crop(x, target):
    if x is None:
        return None
    if target is None:
        return x

    target_shape = target.shape
    diff = x.shape[-1] - target_shape[-1]

    assert (diff % 2 == 0)
    crop = diff // 2

    if crop == 0:
        return x
    if crop < 0:
        raise ArithmeticError

    return x[:, :, crop:-crop].contiguous()


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
        out = F.relu(self.norm((self.filter(x))))
        return out

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
            # Transposed convolution: calculate output size without padding
            curr_size = (input_size - 1) * self.stride + self.kernel_size
        else:
            # Standard convolution
            curr_size = input_size
            curr_size = curr_size - self.kernel_size + 1

            # Stride adjustment for standard convolution
            assert ((curr_size - 1) % self.stride == 0)
            curr_size = ((curr_size - 1) // self.stride) + 1

        assert curr_size > 0
        return curr_size

class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)

        self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, transpose=True)

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_outputs, kernel_size, 1)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1) for _ in range(depth - 1)])

    def forward(self, x, shortcut):
        upsampled = self.upconv(x) # UPSAMPLE HIGH-LEVEL FEATURES

        for i, conv in enumerate(self.pre_shortcut_convs):
            upsampled = conv(upsampled)

        combined = center_crop(shortcut, upsampled)

        for i, conv in enumerate(self.post_shortcut_convs):
            combined = conv(torch.cat([combined, center_crop(upsampled, combined)], dim=1))

        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        for conv in self.pre_shortcut_convs: # Upsampling convs
            curr_size = conv.get_output_size(curr_size)

        for conv in self.post_shortcut_convs: # Combine convolutions
            curr_size = conv.get_output_size(curr_size)

        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1) for _ in
                                                  range(depth - 1)])

        self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride)

    def forward(self, x):
        shortcut = x # PREPARING SHORTCUT FEATURES
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        out = shortcut # PREPARING FOR DOWNSAMPLING
        for conv in self.post_shortcut_convs:
            out = conv(out)

        out = self.downconv(out) # DOWNSAMPLING

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size


class Waveunet(nn.Module):
    def __init__(self, num_channels, kernel_size, target_output_size, depth=1, strides=2):
        super(Waveunet, self).__init__()
        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size = kernel_size
        self.depth = depth

        assert (kernel_size % 2 == 1)

        # Downsampling blocks
        self.downsampling_blocks = nn.ModuleList()
        for i in range(self.num_levels - 1):
            in_ch = 2 if i == 0 else num_channels[i]
            self.downsampling_blocks.append(
                DownsamplingBlock(in_ch, num_channels[i], num_channels[i + 1],
                                  kernel_size, strides, depth)
            )

        # Bottleneck
        self.bottlenecks = nn.ModuleList(
            [ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1) for _ in range(depth)]
        )

        # Upsampling blocks
        self.upsampling_blocks = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.upsampling_blocks.append(
                UpsamplingBlock(num_channels[-1 - i], num_channels[-2 - i], num_channels[-2 - i],
                                kernel_size, strides, depth)
            )

        # Output convolution
        self.output_conv = nn.Conv1d(num_channels[0], 1, 1)

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size
        self.input_size, self.output_size = self.check_padding(target_output_size)
        assert ((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {
            "output_start_frame": (self.input_size - self.output_size) // 2,
            "output_end_frame": (self.input_size - self.output_size) // 2 + self.output_size,
            "output_frames": self.output_size,
            "input_frames": self.input_size
        }

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

            for block in reversed(self.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for block in reversed(self.downsampling_blocks):
                curr_size = block.get_input_size(curr_size)

            assert output_size >= target_output_size
            return curr_size, output_size
        except AssertionError:
            return False

    def forward(self, x):
        curr_input_size = x.shape[-1]
        assert (curr_input_size == self.input_size)

        shortcuts = []
        out = x

        for block in self.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)

        for conv in self.bottlenecks:
            out = conv(out)

        for idx, block in enumerate(self.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])

        out = self.output_conv(out) # Output

        if not self.training:
            out = out.clamp(min=-1.0, max=1.0)

        return out