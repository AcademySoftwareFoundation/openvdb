# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import torch.nn as nn
from fvdb_benchmark.utils import encode_range_name
from fvdb_benchmark.wrapper import Wrapper
from torch.profiler import record_function

import fvdb
import fvdb.nn as fvdbnn


class UpDown(nn.Module):

    def __init__(self, backend: str, factor: int = 2):
        super().__init__()

        self.backend = backend
        self.wrapper = Wrapper(backend=backend)
        self.factor = factor

        self.up = self.wrapper.upsample_generative("up", factor)
        self.down = self.wrapper.maxpool("down", factor)

    def forward(self, x):
        # Isolate grid creation time for fvdb
        if self.backend == "fvdb":
            with record_function(encode_range_name("grid_creation", self.backend, {})):
                assert isinstance(x, fvdbnn.VDBTensor)
                up_grid = x.grid.subdivided_grid(self.factor)
                up_kwargs = {"ref_fine_data": up_grid}
                down_kargs = {"ref_coarse_data": x.grid}
        else:
            up_kwargs = down_kargs = {}

        x_big = self.up(x, **up_kwargs)
        x = self.down(x_big, **down_kargs)

        return x
