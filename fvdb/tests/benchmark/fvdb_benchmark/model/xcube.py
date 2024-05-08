import torch.nn as nn

from fvdb_benchmark.wrapper import Wrapper


class ConvBlock(nn.Sequential):
    def __init__(self, name: str, in_channels: int, out_channels: int, order: str, backend: str):
        super().__init__()

        self.wrapper = Wrapper(backend=backend)

        for i, char in enumerate(order):
            if char == 'r':
                self.add_module('ReLU', self.wrapper.relu(inplace=True))
            elif char == 'c':
                self.add_module('Conv', self.wrapper.conv3d(
                    f"{name}_conv_{i}",
                    in_channels,
                    out_channels, 3, 1,
                    bias='g' not in order)
                )
            else:
                raise NotImplementedError


class SparseHead(nn.Sequential):
    def __init__(self, name: str, in_channels: int, out_channels: int, order: str, backend: str):
        super().__init__()

        self.wrapper = Wrapper(backend=backend)

        self.add_module('SingleConv', ConvBlock(
            f"{name}_conv_block", in_channels, in_channels, order, backend
        ))
        mid_channels = in_channels
        if out_channels > mid_channels:
            mid_channels = out_channels

        self.add_module('OutConv-1', self.wrapper.conv3d(
            f"{name}_oconv1", in_channels, mid_channels, 1, bias=True))
        self.add_module('ReLU-1', self.wrapper.relu(inplace=True))
        self.add_module('OutConv', self.wrapper.conv3d(
            f"{name}_oconv2", mid_channels, out_channels, 1, bias=True))


class SparseDoubleConv(nn.Sequential):
    def __init__(self,
                 name: str,
                 in_channels: int,
                 out_channels: int,
                 order: str,
                 encoder: bool,
                 pooling = None,
                 backend: str = 'ts'):
        super().__init__()

        self.wrapper = Wrapper(backend=backend)
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
            if pooling == 'max':
                self.add_module('MaxPool', self.wrapper.maxpool(f"{name}_maxpool", 2))
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.add_module('SingleConv1', ConvBlock(f"{name}_cb1", conv1_in_channels, conv1_out_channels, order, backend))
        self.add_module('SingleConv3', ConvBlock(f"{name}_cb2", conv1_out_channels, conv1_in_channels, order, backend))
        self.add_module('SingleConv4', ConvBlock(f"{name}_cb3", conv1_in_channels, conv1_out_channels, order, backend))
        self.add_module('SingleConv2', ConvBlock(f"{name}_cb4", conv2_in_channels, conv2_out_channels, order, backend))


class XCubeVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_blocks: int = 4,
                 f_maps: int = 64,
                 order: str = 'cr',
                 backend='ts', **kwargs):

        super().__init__()
        n_features = [in_channels] + [f_maps * 2 ** k for k in range(num_blocks)]

        self.encoders = nn.ModuleList()
        self.pre_kl_bottleneck = nn.ModuleList()
        self.post_kl_bottleneck = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.struct_convs = nn.ModuleList()
        self.num_blocks = num_blocks

        self.wrapper = Wrapper(backend=backend)

        # Encoder
        self.pre_conv = self.wrapper.conv3d("pre_conv", in_channels, in_channels, 1, 1)
        for layer_idx in range(num_blocks):
            self.encoders.add_module(f'Enc{layer_idx}', SparseDoubleConv(
                f"Enc{layer_idx}",
                n_features[layer_idx],
                n_features[layer_idx + 1],
                order,
                True,
                'max' if layer_idx > 0 else None,
                backend=backend
            ))

        # Bottleneck
        self.pre_kl_bottleneck.add_module(f'pre_kl_bottleneck_0', SparseDoubleConv(
            "pre_kl_bottleneck_0",
            n_features[-1], n_features[-1], order, False, backend=backend))
        self.post_kl_bottleneck.add_module(f'post_kl_bottleneck_1', SparseDoubleConv(
            "post_kl_bottleneck_1",
            n_features[-1], n_features[-1], order, False, backend=backend))

        # Decoder
        for layer_idx in range(-1, -num_blocks - 1, -1):
            self.struct_convs.add_module(f'Struct{layer_idx}', SparseHead(
                f'Struct{layer_idx}',
                n_features[layer_idx], 2, order, backend))
            if layer_idx < -1:
                self.decoders.add_module(f'Dec{layer_idx}', SparseDoubleConv(
                    f'Dec{layer_idx}',
                    n_features[layer_idx + 1],
                    n_features[layer_idx],
                    order, False, None, backend
                ))
                self.upsamplers.add_module(f'Up{layer_idx}', self.wrapper.upsample_generative(f"up-{layer_idx}", 2))

    def forward(self, x, gt_coords: dict):
        x = self.pre_conv(x)

        for module in self.encoders:
            x = module(x)

        for module in self.pre_kl_bottleneck:
            x = module(x)

        for module in self.post_kl_bottleneck:
            x = module(x)

        struct_decision = None
        feat_depth = self.num_blocks - 1

        for module, upsampler, struct_conv in zip(
                [None] + list(self.decoders), [None] + list(self.upsamplers), self.struct_convs):
            if module is not None:
                x = upsampler(x, struct_decision)
                x = module(x)

            struct_decision = struct_conv(x)

            if feat_depth >= 0:
                struct_decision = self.wrapper.get_mask(x, gt_coords[feat_depth])
                feat_depth -= 1

        return x
