import glob
import os.path
import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load


def load_torch_extension(name, additional_files=None, ignore_files=None, **kwargs):
    if ignore_files is None:
        ignore_files = []

    if additional_files is None:
        additional_files = []

    def path_should_keep(pth):
        for file_name in ignore_files:
            if file_name in pth:
                return False
        return True

    base_path = Path(__file__).parent / name
    cpp_files = glob.glob(str(base_path / "*.cpp"), recursive=True)
    cpp_files = filter(path_should_keep, cpp_files)
    cu_files = glob.glob(str(base_path / "*.cu"), recursive=True)
    cu_files = filter(path_should_keep, cu_files)

    return load(
        name="fvdb_test_" + name,
        sources=list(cpp_files) + list(cu_files) + [base_path / t for t in additional_files],
        verbose='COMPILE_VERBOSE' in os.environ.keys(),
        extra_ldflags = ["-L%s/lib" %os.environ.get("CONDA_PREFIX")] if os.environ.get("CONDA_PREFIX") else None,
        **kwargs
    )


common = load_torch_extension(
    'common', extra_cflags=['-O2'], extra_cuda_cflags=['-O2', '-Xcompiler -fno-gnu-unique']
)


class CuckooHashTable:
    # Note: This is supposed to be replaced by fVDB.
    def __init__(self, data: torch.Tensor = None, hashed_data: torch.Tensor = None, enlarged: bool = False):
        self.is_empty = False
        if data is not None:
            self.dim = data.size(1)
            source_hash = self._sphash(data)
        else:
            self.dim = -1   # Never equals me.
            source_hash = hashed_data
        self.object = common.build_hash_table(source_hash, torch.tensor([]), enlarged)

    @classmethod
    def _sphash(cls, coords: torch.Tensor, offsets=None) -> torch.Tensor:     # Int64
        assert coords.dtype in [torch.int, torch.long], coords.dtype
        coords = coords.contiguous()
        if offsets is None:
            assert coords.ndim == 2 and coords.shape[1] in [2, 3, 4], coords.shape
            if coords.size(0) == 0:
                return torch.empty((coords.size(0), ), dtype=torch.int64, device=coords.device)
            return common.hash_cuda(coords)
        else:
            assert offsets.dtype == torch.int, offsets.dtype
            assert offsets.ndim == 2 and offsets.shape[1] == 3, offsets.shape
            assert coords.ndim == 2 and coords.shape[1] in [3, 4], coords.shape
            if coords.size(0) == 0 or offsets.size(0) == 0:
                return torch.empty((offsets.size(0), coords.size(0)), dtype=torch.int64, device=coords.device)
            offsets = offsets.contiguous()
            return common.kernel_hash_cuda(coords, offsets)

    def query(self, coords, offsets=None):
        assert coords.size(1) == self.dim
        hashed_query = self._sphash(coords, offsets)
        return self.query_hashed(hashed_query)

    def query_hashed(self, hashed_query: torch.Tensor):
        sizes = hashed_query.size()
        hashed_query = hashed_query.view(-1)

        if hashed_query.size(0) == 0:
            return torch.zeros(sizes, dtype=torch.int64, device=hashed_query.device) - 1

        output = common.hash_table_query(self.object, hashed_query.contiguous())
        output = (output - 1).view(*sizes)

        return output
