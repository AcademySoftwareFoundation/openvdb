import torch
import numpy as np
import time
from fvdb import JaggedTensor, SparseIndexGrid


def main():
    device = 'cuda'
    dtype = torch.float32
    jagged_point_data = [
        torch.randint(low=-1024, high=1024, size=(1_000_000 + np.random.randint(1000), 3), device=device, dtype=torch.int32)
        for _ in range(8)
    ]
    print([jagged_point_data[i].shape for i in range(len(jagged_point_data))])
    jag = JaggedTensor(jagged_point_data)
    grid = SparseIndexGrid(device=device)
    start = time.time()
    grid.set_from_ijk_coords(jag, voxel_size=0.01)
    torch.cuda.synchronize()
    print(f"Done in {time.time() - start:.3f} seconds")

    print(grid.active_grid_coords().data().shape)

if __name__ == "__main__":
    main()