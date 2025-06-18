#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import sys

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def save_tensors_torchscript(tensor_list, filepath, tensor_names=None):
    """
    Save a list of tensors using TorchScript format for C++ compatibility.

    Args:
        tensor_list: List of tensors to save
        filepath: Path to save the tensors to
        tensor_names: List of names for the tensors (if None, will use generic names)
    """

    # Create a container module
    class TensorContainer(torch.nn.Module):
        def __init__(self, tensors, names=None):
            super().__init__()
            self.tensor_names = []
            for i, tensor in enumerate(tensors):
                if names and i < len(names):
                    name = names[i]
                else:
                    name = f"tensor_{i}"
                self.tensor_names.append(name)
                self.register_buffer(name, tensor)

        def forward(self):
            # Create a NamedTuple dynamically based on tensor names
            from collections import namedtuple

            OutputTuple = namedtuple("OutputTuple", self.tensor_names)
            return OutputTuple(*(getattr(self, name) for name in self.tensor_names))

        def get_tensor_names(self):
            # Helper method to expose tensor names
            return self.tensor_names

    # Use descriptive names if not provided
    if tensor_names is None:
        # Default names for input tensors
        if len(tensor_list) == 6:  # Input tensors
            tensor_names = ["means2d", "conics", "colors", "opacities", "tile_offsets", "tile_gaussian_ids"]
        elif len(tensor_list) == 3:  # Output tensors
            tensor_names = ["rendered_colors", "rendered_alphas", "last_ids"]

    # Create and script the container
    container = TensorContainer(tensor_list, tensor_names)

    # Print tensor information for debugging
    print(f"Creating TorchScript container with tensors:")
    for i, tensor in enumerate(tensor_list):
        name = tensor_names[i] if tensor_names and i < len(tensor_names) else f"tensor_{i}"
        print(f"  - {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

    try:
        # Trace the module instead of scripting for better compatibility
        example_inputs = ()  # forward takes no inputs
        scripted_module = torch.jit.trace(container, example_inputs)

        # Save the module
        torch.jit.save(scripted_module, filepath)
        print(f"Successfully saved TorchScript module to {filepath}")

        # Test load to verify it works
        try:
            test_load = torch.jit.load(filepath)
            print(f"Verified the saved module can be loaded")
        except Exception as e:
            print(f"WARNING: Saved module failed test loading: {e}")

    except Exception as e:
        print(f"ERROR during TorchScript saving: {e}")
        # Fall back to legacy format if TorchScript fails
        print(f"Falling back to legacy format...")
        legacy_path = filepath + ".legacy"
        torch.save(tensor_list, legacy_path, _use_new_zipfile_serialization=False)
        print(f"Saved tensors in legacy format to {legacy_path}")
        return legacy_path

    if tensor_names:
        print(f"Saved {len(tensor_list)} tensors with names: {tensor_names}")
    else:
        print(f"Saved {len(tensor_list)} tensors with generic names")

    return filepath


def generate_gaussian_splat_dataset(
    out_dir,
    batch_size=1,
    image_width=1297,
    image_height=840,
    origin_w=0,
    origin_h=0,
    tile_size=16,
    num_gaussians=500,
    seed=42,
    device="cuda",
    density=0.7,
):
    """
    Generate a dataset of gaussian splats for testing the GaussianRasterizeForward function.

    Parameters:
    - out_dir: Directory to save the output files
    - batch_size: Number of batches (usually number of cameras)
    - image_width: Width of the image in pixels
    - image_height: Height of the image in pixels
    - origin_w: X-coordinate of the image origin
    - origin_h: Y-coordinate of the image origin
    - tile_size: Size of the tiles for rasterization
    - num_gaussians: Number of gaussian splats to generate
    - seed: Random seed for reproducibility
    - device: Device to use for computation (cuda or cpu)
    - density: Controls the density of the gaussians (0.0-1.0)
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Calculate tile grid size
    tile_height = (image_height + tile_size - 1) // tile_size
    tile_width = (image_width + tile_size - 1) // tile_size

    print(f"Generating {num_gaussians} gaussian splats for {batch_size} batches")
    print(f"Image dimensions: {image_width}x{image_height}")
    print(f"Tile grid: {tile_width}x{tile_height} (tile size: {tile_size})")

    # Generate 2D means (positions of the gaussians)
    means2d = torch.zeros(batch_size, num_gaussians, 2, device=device)
    for b in range(batch_size):
        # Distribute means across the image
        for i in range(num_gaussians):
            x = torch.rand(1, device=device) * image_width
            y = torch.rand(1, device=device) * image_height
            means2d[b, i, 0] = x  # X position
            means2d[b, i, 1] = y  # Y position

    # Generate conics (2D covariance matrices in the form [a, b, c] for [xx, xy, yy])
    conics = torch.zeros(batch_size, num_gaussians, 3, device=device)
    for b in range(batch_size):
        for i in range(num_gaussians):
            # Create random but valid covariance matrices
            sigma_x = torch.rand(1, device=device) * 30.0 + 5.0  # Between 5 and 35 pixels
            sigma_y = torch.rand(1, device=device) * 30.0 + 5.0  # Between 5 and 35 pixels

            # Add some correlation between x and y for more interesting shapes
            # but keep it small to ensure positive definiteness
            rho = torch.rand(1, device=device) * 0.6 - 0.3  # Between -0.3 and 0.3

            # Convert to conics representation
            # For covariance matrix [sigma_x^2, rho*sigma_x*sigma_y; rho*sigma_x*sigma_y, sigma_y^2]
            # The conic form is [1/sigma_x^2, -rho/(sigma_x*sigma_y), 1/sigma_y^2]
            conics[b, i, 0] = 1.0 / (sigma_x * sigma_x)  # xx component
            conics[b, i, 1] = -rho / (sigma_x * sigma_y)  # xy component
            conics[b, i, 2] = 1.0 / (sigma_y * sigma_y)  # yy component

    # Generate colors (can specify different dimensions)
    color_dim = 3  # RGB by default
    colors = torch.rand(batch_size, num_gaussians, color_dim, device=device)

    # Generate opacities (between 0.2 and 0.8 for more realistic values)
    opacities = torch.rand(batch_size, num_gaussians, device=device) * 0.6 + 0.2

    # Generate intersection data
    # Initialize tile offsets
    tile_offsets = torch.zeros(batch_size, tile_height, tile_width, dtype=torch.int32, device=device)

    # List to collect gaussian IDs that intersect with each tile
    all_gaussian_ids = []

    # Compute actual intersections of gaussians with tiles
    for b in range(batch_size):
        tile_offset = 0
        for i in range(tile_height):
            for j in range(tile_width):
                # Record start of this tile's intersections
                tile_offsets[b, i, j] = tile_offset

                # Compute tile bounds
                tile_min_x = j * tile_size
                tile_min_y = i * tile_size
                tile_max_x = min((j + 1) * tile_size, image_width)
                tile_max_y = min((i + 1) * tile_size, image_height)

                # Find gaussians that intersect with this tile
                for g in range(num_gaussians):
                    x, y = means2d[b, g, 0].item(), means2d[b, g, 1].item()

                    # Estimate the gaussian radius (3 * sigma as a rough approximation)
                    sigma_x = 1.0 / torch.sqrt(conics[b, g, 0]).item() * 3
                    sigma_y = 1.0 / torch.sqrt(conics[b, g, 2]).item() * 3

                    # Check if the gaussian intersects the tile
                    if (
                        x - sigma_x <= tile_max_x
                        and x + sigma_x >= tile_min_x
                        and y - sigma_y <= tile_max_y
                        and y + sigma_y >= tile_min_y
                    ):
                        all_gaussian_ids.append(g)
                        tile_offset += 1

    # Convert all_gaussian_ids to tensor
    total_intersections = len(all_gaussian_ids)
    if total_intersections == 0:
        # Ensure we have at least some intersections for testing
        all_gaussian_ids = list(range(min(100, num_gaussians)))
        total_intersections = len(all_gaussian_ids)

    tile_gaussian_ids = torch.tensor(all_gaussian_ids, dtype=torch.int32, device=device)

    print(f"Generated {total_intersections} gaussian-tile intersections")

    # Save the input tensors using TorchScript format
    inputs = [means2d, conics, colors, opacities, tile_offsets, tile_gaussian_ids]
    input_names = ["means2d", "conics", "colors", "opacities", "tile_offsets", "tile_gaussian_ids"]
    inputs_path = os.path.join(out_dir, "gaussian_splat_input.pt")
    print(f"Saving inputs to {inputs_path}")
    save_tensors_torchscript(inputs, inputs_path, input_names)

    # Return the inputs so we can use them to generate outputs if needed
    return inputs


def main():
    parser = argparse.ArgumentParser(description="Generate gaussian splat dataset for testing")
    parser.add_argument("--out-dir", type=str, default="./data", help="Output directory for the dataset")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of batches (usually cameras)")
    parser.add_argument("--image-width", type=int, default=1297, help="Width of the image in pixels")
    parser.add_argument("--image-height", type=int, default=840, help="Height of the image in pixels")
    parser.add_argument("--num-gaussians", type=int, default=500, help="Number of gaussian splats to generate")
    parser.add_argument("--tile-size", type=int, default=16, help="Size of the tiles for rasterization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    # Generate the dataset
    inputs = generate_gaussian_splat_dataset(
        args.out_dir,
        batch_size=args.batch_size,
        image_width=args.image_width,
        image_height=args.image_height,
        tile_size=args.tile_size,
        num_gaussians=args.num_gaussians,
        seed=args.seed,
        device=args.device,
    )

    print("Done!")


if __name__ == "__main__":
    main()
