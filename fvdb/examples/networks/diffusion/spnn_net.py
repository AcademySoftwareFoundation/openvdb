import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchsparse import SparseTensor
import torchsparse.nn as spnn
import torchsparse.operators as spop
from torchsparse.nn import functional as spF

from torchsparse.nn.utils import fapply, get_kernel_offsets


def zero_module(module: nn.Module):
    """Zero out the parameters of a module."""
    for p in module.parameters():
        p.data.zero_()
    return module


class TimestepModule(nn.Module):
    def forward(self, x, emb, target_tensor = None):
        raise NotImplementedError


class TimestepSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepModule):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class SiLU(nn.SiLU):
    def forward(self, input: SparseTensor):
        return fapply(input, super().forward)


class Dropout(nn.Dropout):
    def forward(self, input: SparseTensor):
        return fapply(input, super().forward)


def nearest_upsample(input: SparseTensor):
    in_stride = input.stride
    out_stride = tuple([in_stride[k] // 2 for k in range(3)])

    in_coords = input.coords
    out_coords = input.cmaps[out_stride]

    in_feature = input.feats

    out_in_pos = torch.div(out_coords[:, :3], in_stride[0], rounding_mode='floor') * in_stride[0]
    out_in_coord = torch.cat([out_in_pos, out_coords[:, 3:]], dim=1).int()

    queries = spF.sphash(out_in_coord)
    references = spF.sphash(in_coords)
    results = spF.sphashquery(queries, references)
    assert torch.all(results >= 0)
    out_feature = in_feature[results]

    output = SparseTensor(feats=out_feature, coords=out_coords, stride=out_stride)
    output.cmaps = input.cmaps
    output.kmaps = input.kmaps
    return output


def avg_pooling(input: SparseTensor):
    out_coords = spF.spdownsample(input.coords, stride=2, kernel_size=2, tensor_stride=input.stride)
    offsets = get_kernel_offsets(2, input.stride, device=input.feats.device)

    references = spF.sphash(input.coords)
    queries = spF.sphash(out_coords, offsets)
    results = spF.sphashquery(queries, references)

    in_feature = torch.cat([
        torch.zeros((1, input.feats.size(1)), device=input.feats.device, dtype=input.feats.dtype), input.feats
    ], dim=0)
    out_feature = torch.mean(in_feature[results + 1], dim=0)       # [8, N, C] -> [N, C]

    output = SparseTensor(feats=out_feature, coords=out_coords,
                          stride=tuple(input.stride[k] * 2 for k in range(3)))
    output.cmaps = input.cmaps
    output.cmaps.setdefault(output.stride, output.coords)
    output.kmaps = output.kmaps

    return output


class Upsample(nn.Module):
    def __init__(self, channels, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dims = dims

    def forward(self, data: SparseTensor):
        data = nearest_upsample(data)
        return data


class Downsample(nn.Module):
    def __init__(self, channels, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dims = dims

    def forward(self, data: SparseTensor):
        data = avg_pooling(data)
        return data


class ResBlock(TimestepModule):
    def __init__(self, channels: int, emb_channels: int, dropout: float,
                 out_channels: Optional[int] = None,
                 up: bool = False, down: bool = False, stride: int = 1):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.stride = stride

        # Nonlinear operations to time/class embeddings
        #   (added between in_layers and out_layers in the res branch)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.emb_channels, 2 * self.out_channels)
        )

        self.in_layers = nn.Sequential(
            spnn.GroupNorm(num_groups=32, num_channels=channels),
            SiLU(),
            spnn.Conv3d(self.channels, self.out_channels, 3, bias=True)
        )

        self.up, self.down = up, down
        if self.up:
            self.up_module = Upsample(channels)
        elif self.down:
            self.down_module = Downsample(channels)

        self.out_layers = nn.Sequential(
            spnn.GroupNorm(num_groups=32, num_channels=self.out_channels),
            SiLU(),
            Dropout(p=self.dropout),
            # Zero out res output since this is the residual
            zero_module(spnn.Conv3d(self.out_channels, self.out_channels, 3, bias=True))
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = spnn.Conv3d(channels, self.out_channels, 1, bias=True)

    def forward(self, data: SparseTensor, emb: torch.Tensor):
        if self.up or self.down:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            data_h = in_rest(data)
            data_h = self.up_module(data_h) \
                if self.up else self.down_module(data_h)
            data_h = in_conv(data_h)
            data = self.up_module(data) \
                if self.up else self.down_module(data)
        else:
            data_h = self.in_layers(data)

        emb_h = self.emb_layers(emb)    # (B, 2C)
        scale, shift = emb_h.chunk(2, dim=-1)   # (B, C), (B, C)

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        batch_idx = data_h.coords[:, 3].long()
        data_h.feats = out_norm(data_h).feats * (1 + scale[batch_idx]) + shift[batch_idx]
        data_h = out_rest(data_h)

        data = data_h + self.skip_connection(data)
        return data


class UNetModel(nn.Module):
    def __init__(self, in_channels: int, model_channels: int, num_res_blocks: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 channel_mult: Tuple = (1, 2, 4, 8), num_classes: Optional[int] = None):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.out_channels = out_channels or in_channels
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes

        time_emb_dim = 4 * self.model_channels
        self.time_emb = nn.Sequential(
            nn.Linear(self.model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        if self.num_classes is not None:
            self.label_emb = nn.Linear(self.num_classes, time_emb_dim)

        # Encoder
        self.encoder_blocks = nn.ModuleList([TimestepSequential(
            spnn.Conv3d(self.in_channels, self.model_channels, 3, bias=True),
        )])

        encoder_channels = [self.model_channels]
        current_channels = self.model_channels
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [ResBlock(
                    current_channels, time_emb_dim, self.dropout,
                    out_channels=self.model_channels * mult
                )]
                current_channels = self.model_channels * mult
                self.encoder_blocks.append(TimestepSequential(*layers))
                encoder_channels.append(current_channels)
            # Downsample for all but the last block
            if level < len(self.channel_mult) - 1:
                layers = [ResBlock(
                    current_channels, time_emb_dim, self.dropout,
                    out_channels=current_channels,
                    down=True, stride=2
                )]
                self.encoder_blocks.append(TimestepSequential(*layers))
                encoder_channels.append(current_channels)

        # Middle block (won't change dimension)
        self.middle_block = TimestepSequential(
            ResBlock(current_channels, time_emb_dim, self.dropout),
            ResBlock(current_channels, time_emb_dim, self.dropout)
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            # Use one more block for decoder
            for i in range(self.num_res_blocks + 1):
                skip_channels = encoder_channels.pop()
                layers = [ResBlock(
                    current_channels + skip_channels,
                    time_emb_dim, self.dropout,
                    out_channels=self.model_channels * mult
                )]
                current_channels = self.model_channels * mult
                # Upsample for all but the finest block
                if level > 0 and i == self.num_res_blocks:
                    layers.append(ResBlock(
                        current_channels, time_emb_dim, self.dropout,
                        out_channels=current_channels,
                        up=True, stride=2
                    ))
                self.decoder_blocks.append(TimestepSequential(*layers))

        # Output block
        assert current_channels == self.model_channels
        self.out_block = nn.Sequential(
            spnn.GroupNorm(num_groups=32, num_channels=current_channels),
            SiLU(),
            zero_module(spnn.Conv3d(current_channels, self.out_channels, 3, bias=True))
        )

    def timestep_encoding(self, timesteps: torch.Tensor, max_period: int = 10000):
        dim = self.model_channels
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, data: SparseTensor, timesteps: torch.Tensor,
                y: Optional[torch.Tensor] = None):
        assert (y is not None) == (self.num_classes is not None), \
            "Must provide labels if num_classes is not None"

        t_emb = self.timestep_encoding(timesteps)
        emb = self.time_emb(t_emb)
        if y is not None:
            emb += self.label_emb(y)

        hs = []
        for block in self.encoder_blocks:
            data = block(data, emb)
            hs.append(data)
        data = self.middle_block(data, emb)
        for block in self.decoder_blocks:
            data = spop.cat([hs.pop(), data])
            data = block(data, emb)

        data = self.out_block(data)
        return data
