# A Simple Convolutional U-Net

In this tutorial, you will be guided on how to build a simple sparse convolutional neural network using fVDB.
If you were using MinkowskiEngine to tackle sparse 3D data previously, we will also guide you step-by-step to help you smoothly transfer from it and enjoy speed-ups and memory-savings.

In our simplistic U-Net case, we want to build a Res-UNet with four layers, and each layer contains several blocks.
First, we import basic `fvdb` libraries:

```python
import fvdb
import fvdb.nn as fvnn
from fvdb.nn import VDBTensor
```

Here `fvdb.nn` is a namespace similar to `torch.nn`, containing a broad definition of different neural layers.
`VDBTensor` is a very thin wrapper around a grid (with type `GridBatch`) and its corresponding feature (with type `JaggedTensor`), and internally makes sure that the two members align.
It also overloads a bunch of operators such as arithmetic computations. Please refer to our API docs to learn more.

We could then build a basic block as follows:

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, downsample=None, bn_momentum: float = 0.1):
        super().__init__()
        self.conv1 = fvnn.SparseConv3d(in_channels, out_channels, kernel_size=3, stride=1)
        self.norm1 = fvnn.BatchNorm(out_channels, momentum=bn_momentum)
        self.conv2 = fvnn.SparseConv3d(out_channels, out_channels, kernel_size=3, stride=1)
        self.norm2 = fvnn.BatchNorm(out_channels, momentum=bn_momentum)
        self.relu = fvnn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: VDBTensor):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
```

This defines a similar block as `MinkowskiEngine`:

```python
import MinkowskiEngine as ME


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, downsample=None, bn_momentum: float = 0.1):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dimension=3)
        self.norm1 = ME.MinkowskiBatchNorm(out_channels, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            out_channels, out_channels, kernel_size=3, stride=1, dilation=1, dimension=3)
        self.norm2 = ME.MinkowskiBatchNorm(out_channels, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
```

All the network layers are fully compatible with `torch.nn`. The only difference is that they take `VDBTensor` as input and return a `VDBTensor`.
A full network definition could then be built as:

```python
class FVDBUNetBase(nn.Module):
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    CHANNELS = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    def __init__(self, in_channels, out_channels, D=3):
        super().__init__()

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = fvnn.SparseConv3d(in_channels, self.inplanes, kernel_size=5, stride=1, bias=False)
        self.bn0 = fvnn.BatchNorm(self.inplanes)

        self.conv1p1s2 = fvnn.SparseConv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2, bias=False)
        self.bn1 = fvnn.BatchNorm(self.inplanes)

        self.block1 = self._make_layer(BasicBlock, self.CHANNELS[0], self.LAYERS[0])

        self.conv2p2s2 = fvnn.SparseConv3d(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, bias=False)
        self.bn2 = fvnn.BatchNorm(self.inplanes)

        self.block2 = self._make_layer(BasicBlock, self.CHANNELS[1], self.LAYERS[1])

        self.conv3p4s2 = fvnn.SparseConv3d(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, bias=False)

        self.bn3 = fvnn.BatchNorm(self.inplanes)
        self.block3 = self._make_layer(BasicBlock, self.CHANNELS[2], self.LAYERS[2])

        self.conv4p8s2 = fvnn.SparseConv3d(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, bias=False)
        self.bn4 = fvnn.BatchNorm(self.inplanes)
        self.block4 = self._make_layer(BasicBlock, self.CHANNELS[3], self.LAYERS[3])

        self.convtr4p16s2 = fvnn.SparseConv3d(
            self.inplanes, self.CHANNELS[4], kernel_size=2, stride=2, transposed=True, bias=False)
        self.bntr4 = fvnn.BatchNorm(self.CHANNELS[4])

        self.inplanes = self.CHANNELS[4] + self.CHANNELS[2]
        self.block5 = self._make_layer(BasicBlock, self.CHANNELS[4], self.LAYERS[4])
        self.convtr5p8s2 = fvnn.SparseConv3d(
            self.inplanes, self.CHANNELS[5], kernel_size=2, stride=2, transposed=True, bias=False)
        self.bntr5 = fvnn.BatchNorm(self.CHANNELS[5])

        self.inplanes = self.CHANNELS[5] + self.CHANNELS[1]
        self.block6 = self._make_layer(BasicBlock, self.CHANNELS[5], self.LAYERS[5])
        self.convtr6p4s2 = fvnn.SparseConv3d(
            self.inplanes, self.CHANNELS[6], kernel_size=2, stride=2, transposed=True, bias=False)
        self.bntr6 = fvnn.BatchNorm(self.CHANNELS[6])

        self.inplanes = self.CHANNELS[6] + self.CHANNELS[0]
        self.block7 = self._make_layer(BasicBlock, self.CHANNELS[6], self.LAYERS[6])
        self.convtr7p2s2 = fvnn.SparseConv3d(
            self.inplanes, self.CHANNELS[7], kernel_size=2, stride=2, transposed=True, bias=False)
        self.bntr7 = fvnn.BatchNorm(self.CHANNELS[7])

        self.inplanes = self.CHANNELS[7] + self.INIT_DIM
        self.block8 = self._make_layer(BasicBlock, self.CHANNELS[7], self.LAYERS[7])

        self.final = fvnn.SparseConv3d(self.CHANNELS[7], out_channels, kernel_size=1)
        self.relu = fvnn.ReLU(inplace=True)

    def _make_layer(self, block, planes, blocks):
        downsample = None
        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                fvnn.SparseConv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1
                ),
                fvnn.BatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            BasicBlock(
                self.inplanes, planes,
                downsample=downsample
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        grid1 = out_p1.grid

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        grid2 = out_b1p2.grid

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        grid4 = out_b2p4.grid

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        grid8 = out_b3p8.grid

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out, out_grid=grid8)
        out = self.bntr4(out)
        out = self.relu(out)

        out = fvnn.cat([out, out_b3p8], dim=1)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out, out_grid=grid4)
        out = self.bntr5(out)
        out = self.relu(out)

        out = fvnn.cat([out, out_b2p4], dim=1)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out, out_grid=grid2)
        out = self.bntr6(out)
        out = self.relu(out)

        out = fvnn.cat([out, out_b1p2], dim=1)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out, out_grid=grid1)
        out = self.bntr7(out)
        out = self.relu(out)

        out = fvnn.cat([out, out_p1], dim=1)
        out = self.block8(out)

        return self.final(out)
```

Please note that here, when we apply transposed convolution layers, we additionally introduce the `out_grid` keyword arguments.
This is needed to guide the output domain of the network, because for perception networks, the output grid topology should align with the input topology.
Note that fVDB will NOT cache the grids to maintain maximum flexibility.

To perform inference with the network, you could simply create a VDBTensor and feed it into the model:

```python
xyz: JaggedTensor, features: JaggedTensor = ...
grid = fvdb.sparse_grid_from_points(coords, voxel_sizes=[voxel_size] * 3)
features: JaggedTensor = grid.splat_trilinear(xyz, features)
sinput = fvnn.VDBTensor(grid, features)
soutput = model(sinput)
```

The output `soutput` will carry gradients during training, and you could train the sparse network accordingly.
Please find a fully working example at `examples/perception_example.py`. The same network is implemented using `MinkowskiEngine` for reference.
