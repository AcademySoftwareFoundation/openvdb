import torch.nn as nn
from torch.profiler import record_function

from fvdb_benchmark.wrapper import Wrapper
from fvdb_benchmark.utils import encode_range_name


class SparseResBlock(nn.Module):
    def __init__(self,
                 name: str,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 backend: str = 'ts'):

        super().__init__()
        self.backend = backend
        self.wrapper = Wrapper(backend=self.backend)
        self.name = name

        self.net = self.wrapper.sequential(
            self.wrapper.conv3d(
                f"{name}_conv1",
                in_channels,
                out_channels,
                kernel_size,
                stride=1),
            self.wrapper.bn(out_channels),
            self.wrapper.relu(True),
            self.wrapper.conv3d(
                f"{name}_conv2",
                out_channels,
                out_channels,
                kernel_size),
            self.wrapper.bn(out_channels)
        )

        if in_channels != out_channels:
            self.downsample = self.wrapper.sequential(
                self.wrapper.conv3d(
                    f"{name}_iconv",
                    in_channels,
                    out_channels,
                    1),
                self.wrapper.bn(out_channels),
            )
        else:
            self.downsample = self.wrapper.sequential()

        self.relu = self.wrapper.relu(True)

    def forward(self, x):
        with record_function(encode_range_name(self.name, self.backend, {})):
            return self.relu(self.net(x) + self.downsample(x))



class MinkUNet(nn.Module):
    """Minkowski U-Net for 3D semantic segmentation."""

    def __init__(self,
                 backend: str,
                 in_channels: int = 4,
                 cr: float = 1.0,
                 num_classes: int = 50):
        super().__init__()

        cs = [64, 64, 64, 128, 256, 256, 128, 64, 64]
        cs = [int(cr * x) for x in cs]
        # make sure #channels is even
        for i, x in enumerate(cs):
            if x % 2 != 0:
                cs[i] = x + 1

        self.backend = backend
        self.wrapper = Wrapper(backend=self.backend)

        self.num_blocks = 4

        self.stem = self.wrapper.sequential(
            self.wrapper.conv3d("pre", in_channels, cs[0], kernel_size=3, stride=1),
            self.wrapper.bn(cs[0]), self.wrapper.relu(True),
            self.wrapper.conv3d("pre", cs[0], cs[0], kernel_size=3, stride=1),
            self.wrapper.bn(cs[0]), self.wrapper.relu(True))

        self.stage1 = nn.Sequential(
            self.wrapper.maxpool("down2", 2),
            SparseResBlock("block2a", cs[0], cs[1], kernel_size=3, backend=self.backend),
            SparseResBlock("block2b", cs[1], cs[1], kernel_size=3, backend=self.backend),
        )

        self.stage2 = nn.Sequential(
            self.wrapper.maxpool("down4", 2),
            SparseResBlock("block4a", cs[1], cs[2], kernel_size=3, backend=self.backend),
            SparseResBlock("block4b", cs[2], cs[2], kernel_size=3, backend=self.backend))

        self.stage3 = nn.Sequential(
            self.wrapper.maxpool("down8", 2),
            SparseResBlock("block8a", cs[2], cs[3], kernel_size=3, backend=self.backend),
            SparseResBlock("block8b", cs[3], cs[3], kernel_size=3, backend=self.backend),
        )

        self.stage4 = nn.Sequential(
            self.wrapper.maxpool("down16", 2),
            SparseResBlock("block16a", cs[3], cs[4], kernel_size=3, backend=self.backend),
            SparseResBlock("block16b", cs[4], cs[4], kernel_size=3, backend=self.backend),
        )

        self.up1 = nn.ModuleList([
            self.wrapper.upsample("up8", 2),
            nn.Sequential(
                SparseResBlock("block8c", cs[5] + cs[3], cs[5], kernel_size=3, backend=self.backend),
                SparseResBlock("block8d", cs[5], cs[5], kernel_size=3, backend=self.backend),
            )
        ])

        self.up2 = nn.ModuleList([
            self.wrapper.sequential(
                self.wrapper.upsample("up4", 2),
                self.wrapper.conv3d("block4c", cs[5], cs[6], kernel_size=1, stride=1, bias=False),
            ),
            nn.Sequential(
                SparseResBlock("block4d", cs[6] + cs[2], cs[6], kernel_size=3, backend=self.backend),
                SparseResBlock("block4e", cs[6], cs[6], kernel_size=3, backend=self.backend),
            )
        ])

        self.up3 = nn.ModuleList([
            self.wrapper.sequential(
                self.wrapper.upsample("up2", 2),
                self.wrapper.conv3d("block2c", cs[6], cs[7], kernel_size=1, stride=1, bias=False),
            ),
            nn.Sequential(
                SparseResBlock("block2d", cs[7] + cs[1], cs[7], kernel_size=3, backend=self.backend),
                SparseResBlock("block2e", cs[7], cs[7], kernel_size=3, backend=self.backend),
            )
        ])

        self.up4 = nn.ModuleList([
            self.wrapper.sequential(
                self.wrapper.upsample("up1", 2),
                self.wrapper.conv3d("block1c", cs[7], cs[8], kernel_size=1, stride=1, bias=False),
            ),
            nn.Sequential(
                SparseResBlock("block1d", cs[8] + cs[0], cs[8], kernel_size=3, backend=self.backend),
                SparseResBlock("block1e", cs[8], cs[8], kernel_size=3, backend=self.backend),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8], num_classes))

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        if self.wrapper.backend == 'fvdb':
            y1 = self.up1[0](x4, ref_fine_data=x3)
        else:
            y1 = self.up1[0](x4)

        y1 = self.wrapper.cat([y1, x3])
        y1 = self.up1[1](y1)

        if self.wrapper.backend == 'fvdb':
            y2 = self.up2[0](y1, ref_fine_data=x2)
        else:
            y2 = self.up2[0](y1)
        y2 = self.wrapper.cat([y2, x2])
        y2 = self.up2[1](y2)

        if self.wrapper.backend == 'fvdb':
            y3 = self.up3[0](y2, ref_fine_data=x1)
        else:
            y3 = self.up3[0](y2)
        y3 = self.wrapper.cat([y3, x1])
        y3 = self.up3[1](y3)

        if self.wrapper.backend == 'fvdb':
            y4 = self.up4[0](y3, ref_fine_data=x0)
        else:
            y4 = self.up4[0](y3)
        y4 = self.wrapper.cat([y4, x0])
        y4 = self.up4[1](y4)

        if self.wrapper.backend == 'fvdb':
            out = self.classifier(y4.feature.jdata)
        else:
            out = self.classifier(y4.F)

        return out
