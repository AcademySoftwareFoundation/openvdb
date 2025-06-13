#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch is not installed. Please install it to run this script.")
    sys.exit(1)


def generate_reproduction_command(tensors_dict, pt_filepath):
    """ƒ
    Generates and prints a command line to reproduce the dataset.
    """
    if "means2d" not in tensors_dict:
        print("\nCannot generate reproduction command: 'means2d' tensor not found in the loaded data.")
        return

    means2d = tensors_dict["means2d"]

    try:
        batch_size = means2d.shape[0]
        num_gaussians = means2d.shape[1]
        device = str(means2d.device)

        image_width, image_height = None, None
        # Try to get image dimensions from image_dims tensor first
        if "image_dims" in tensors_dict:
            image_dims = tensors_dict["image_dims"]
            if isinstance(image_dims, torch.Tensor) and image_dims.numel() >= 2:
                # Assuming image_dims is [width, height, ...]
                image_width = int(image_dims[0].item())
                image_height = int(image_dims[1].item())
            else:
                print(
                    "\nWarning: 'image_dims' tensor found but not in the expected format (tensor with at least 2 elements). Falling back to estimation."
                )

        # Estimate image_width and image_height from the extent of means2d if not found
        if image_width is None or image_height is None:
            # Add a small epsilon before ceil to handle cases where max_coord is exactly an integer,
            # ensuring it's at least that integer.
            # Max over all batches and all gaussians for each dimension.
            if means2d.numel() > 0:
                max_x = torch.max(means2d[..., 0])
                max_y = torch.max(means2d[..., 1])
                image_width = torch.ceil(max_x + 1e-6).int().item()
                image_height = torch.ceil(max_y + 1e-6).int().item()
            else:  # Handle empty means2d tensor if it occurs
                image_width = 0
                image_height = 0

        out_dir = os.path.dirname(pt_filepath)
        if not out_dir:  # If pt_filepath is just a filename
            out_dir = "."

        print("\nTo regenerate a similar dataset with generate_gaussian_splat_dataset.py:")
        print("-------------------------------------------------------------------------")
        # Assuming generate_gaussian_splat_dataset.py is in the same directory or in PATH
        # If it\'s in src/tests/scripts/, the command would be:
        # python src/tests/scripts/generate_gaussian_splat_dataset.py \\
        print(f"python generate_gaussian_splat_dataset.py \\")
        print('    --out-dir "' + out_dir + '" \\')
        print(f"    --batch-size {batch_size} \\")
        print(f"    --image-width {image_width} \\")
        print(f"    --image-height {image_height} \\")
        print(f"    --num-gaussians {num_gaussians} \\")
        print('    --device "' + device + '" \\')
        print(f"    --tile-size 16 \\  # Assumed default, original value not stored")
        print(f"    --seed <SEED> \\  # Original seed not stored, replace <SEED> with a desired value")
        print(
            f"    # Parameters like origin_w, origin_h, and density use defaults from the script as they are not stored."
        )
        print("-------------------------------------------------------------------------")
        print("Note: ")
        print("  - 'image_width' and 'image_height' are estimated based on the maximum coordinates of the gaussians.")
        print("  - 'out-dir' is set to the directory of the loaded .pt file.")
        print(
            "  - Parameters like 'seed', 'tile_size' (if not default 16), 'origin_w', 'origin_h', and 'density' are not stored in the .pt file."
        )
        print("    The command above uses common defaults or placeholders for these.")

    except Exception as e:
        print(f"\nError generating reproduction command: {e}")


def load_and_print_tensors(filepath):
    """
    Load tensors from a TorchScript file and print their information.

    Args:
        filepath: Path to the TorchScript file (.pt)
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    print(f"Loading TorchScript module from: {filepath}")
    try:
        # Load the scripted module
        # Ensure the module is loaded to the same device it was saved from, or CPU if not specified
        loaded_module = torch.jit.load(filepath)
        print("Successfully loaded TorchScript module.")

        tensors_by_name = {}
        # This list will store tensor names in the order they are discovered, preferring get_tensor_names list.
        ordered_tensor_names = []

        # Attempt 1: Try to get tensor names from get_tensor_names() method first
        get_tensor_names_list = None
        if hasattr(loaded_module, "get_tensor_names") and callable(loaded_module.get_tensor_names):
            try:
                get_tensor_names_list = loaded_module.get_tensor_names()
                if get_tensor_names_list:
                    print(f"Retrieved tensor names via get_tensor_names(): {get_tensor_names_list}")
                    ordered_tensor_names = list(get_tensor_names_list)  # Prioritize this order
            except Exception as e_get_names:
                print(f"Could not retrieve tensor names using get_tensor_names(): {e_get_names}")

        # Attempt 2: Try to call forward() and map outputs
        try:
            print("Attempting to call forward()...")
            outputs_from_forward = loaded_module.forward()  # Expected to be a tuple or NamedTuple
            print("forward() called successfully.")

            if isinstance(outputs_from_forward, torch.Tensor):
                print("forward() returned a single tensor.")
                name_to_use = ordered_tensor_names[0] if ordered_tensor_names else "output_tensor"
                tensors_by_name[name_to_use] = outputs_from_forward
                if not ordered_tensor_names:
                    ordered_tensor_names.append(name_to_use)
            elif isinstance(outputs_from_forward, tuple):
                if ordered_tensor_names and len(ordered_tensor_names) == len(outputs_from_forward):
                    print("Mapping forward() outputs to names from get_tensor_names().")
                    for i, name in enumerate(ordered_tensor_names):
                        tensors_by_name[name] = outputs_from_forward[i]
                elif hasattr(outputs_from_forward, "_fields"):  # Check if it's a NamedTuple
                    potential_named_tuple = outputs_from_forward
                    print("forward() outputs appear to be a NamedTuple, using its fields.")
                    ordered_tensor_names = list(potential_named_tuple._fields)  # type: ignore[attr-defined]
                    for name in ordered_tensor_names:
                        tensors_by_name[name] = getattr(potential_named_tuple, name)
                else:  # Plain tuple
                    print("forward() returned a plain tuple. Using generic names or existing ordered_tensor_names.")
                    temp_names = []
                    for i, tensor_val in enumerate(outputs_from_forward):
                        name = ordered_tensor_names[i] if i < len(ordered_tensor_names) else f"tensor_{i}"
                        tensors_by_name[name] = tensor_val
                        temp_names.append(name)
                    if not ordered_tensor_names or len(ordered_tensor_names) < len(outputs_from_forward):
                        ordered_tensor_names = temp_names  # Update if we generated new names
            else:
                print(f"forward() returned an unexpected type: {type(outputs_from_forward)}")

        except AttributeError as e_forward:
            if "forward" not in str(e_forward).lower():
                raise  # Not an error about 'forward' missing, so re-raise to outer handler
            print("forward() method not found or failed. Trying attribute/buffer access.")
            # Fall through to Attempt 3 logic below if tensors_by_name is not populated.
        except Exception as e_other_forward:
            print(f"An error occurred attempting to use forward(): {e_other_forward}")
            # Fall through to see if tensors_by_name is populated, then to Attempt 3.

        # Attempt 3: If forward() failed or didn't populate, try direct attribute access (if names exist) or named_buffers
        if not tensors_by_name:
            if ordered_tensor_names:  # From get_tensor_names_list
                print("Attempting to retrieve tensors using pre-fetched tensor names and getattr().")
                try:
                    possible_tensors = {}
                    for name in ordered_tensor_names:
                        possible_tensors[name] = getattr(loaded_module, name)
                    tensors_by_name = possible_tensors  # Commit if all getattr succeeded
                    print(f"Successfully retrieved {len(tensors_by_name)} tensors using getattr().")
                except Exception as e_getattr:
                    print(
                        f"Error using getattr() with pre-fetched names: {e_getattr}. Clearing potentially partial results."
                    )
                    tensors_by_name.clear()  # Clear partial results before trying buffers
                    # ordered_tensor_names remains from get_tensor_names() if that succeeded

            if not tensors_by_name:  # If getattr failed or was not applicable (no ordered_tensor_names)
                print("Attempting to retrieve tensors using named_buffers().")
                try:
                    if hasattr(loaded_module, "named_buffers"):
                        # Use list comprehension to preserve order from named_buffers if get_tensor_names failed
                        buffers_list = list(loaded_module.named_buffers())
                        if buffers_list:
                            temp_ordered_names = []
                            for name, tensor in buffers_list:
                                tensors_by_name[name] = tensor
                                temp_ordered_names.append(name)
                            if not ordered_tensor_names:  # Only overwrite if get_tensor_names failed
                                ordered_tensor_names = temp_ordered_names
                            print(f"Successfully retrieved {len(tensors_by_name)} tensors using named_buffers().")
                        else:
                            print("named_buffers() yielded no tensors.")
                    else:
                        print("Module does not have a named_buffers method.")
                except Exception as e_buffers:
                    print(f"Error using named_buffers(): {e_buffers}")

                if not tensors_by_name:  # If named_buffers also failed
                    print("Attempting to retrieve tensors using state_dict() as a fallback.")
                    try:
                        state_dict = loaded_module.state_dict()
                        if state_dict:
                            temp_ordered_names = []
                            for name, tensor in state_dict.items():
                                tensors_by_name[name] = tensor
                                temp_ordered_names.append(name)
                            if not ordered_tensor_names:
                                ordered_tensor_names = temp_ordered_names
                            print(f"Successfully retrieved {len(tensors_by_name)} tensors using state_dict().")
                        else:
                            print("state_dict() yielded no tensors.")
                    except Exception as e_state_dict:
                        print(f"Error using state_dict(): {e_state_dict}")

        # Check if any method succeeded
        if not tensors_by_name:
            print("Failed to retrieve tensors using forward(), getattr(), or named_buffers().")
            raise RuntimeError("Primary tensor extraction methods failed, will attempt legacy loading.")

        # Printing and command generation based on successfully retrieved tensors_by_name
        print("\nTensor Information:")
        if not ordered_tensor_names and tensors_by_name:  # Fallback if order wasn't set but names exist
            ordered_tensor_names = list(tensors_by_name.keys())

        for name in ordered_tensor_names:  # Print in retrieved order if available
            if name in tensors_by_name:  # Ensure name is in dict, in case ordered_tensor_names is stale
                tensor = tensors_by_name[name]
                if isinstance(tensor, torch.Tensor):
                    print(f"  - Name: {name}, Shape: {tensor.shape}, DType: {tensor.dtype}, Device: {tensor.device}")
                else:
                    print(f"  - Name: {name}, Type: {type(tensor)}, Value: {tensor}")
            else:
                print(f"  - Warning: Name '{name}' in ordered list but not found in extracted tensors.")

        # Print any tensors that might have been missed by ordered_tensor_names (e.g. if order was partial)
        for name, tensor in tensors_by_name.items():
            if name not in ordered_tensor_names:
                print(
                    f"  - Name (unordered): {name}, Shape: {tensor.shape}, DType: {tensor.dtype}, Device: {tensor.device}"
                )

        generate_reproduction_command(tensors_by_name, filepath)

    except Exception as e:  # Outer catch-all, includes RuntimeError from above
        print(f"An error occurred while loading or processing the TorchScript module: {e}")
        print("This might be a legacy .pt file. Attempting to load with torch.load()...")
        try:
            legacy_tensors = torch.load(filepath)
            if isinstance(legacy_tensors, list):
                print("\nTensor Information (Legacy Format):")
                for i, tensor in enumerate(legacy_tensors):
                    if isinstance(tensor, torch.Tensor):
                        print(f"  - Tensor {i}: Shape: {tensor.shape}, DType: {tensor.dtype}, Device: {tensor.device}")
                    else:
                        print(f"  - Item {i}: Type: {type(tensor)}, Value: {tensor}")
            elif isinstance(legacy_tensors, torch.Tensor):  # Single tensor
                print("\nTensor Information (Legacy Format):")
                print(
                    f"  - Tensor 0: Shape: {legacy_tensors.shape}, DType: {legacy_tensors.dtype}, Device: {legacy_tensors.device}"
                )
            else:
                print(
                    f"Loaded legacy file, but the content is not a list of tensors or a single tensor. Type: {type(legacy_tensors)}"
                )

        except Exception as legacy_e:
            print(f"Failed to load as a legacy PyTorch file: {legacy_e}")


def main():
    if not TORCH_AVAILABLE:
        # TORCH_AVAILABLE check already prints a message and exits if False at the top
        return

    parser = argparse.ArgumentParser(description="Load a TorchScript file and print tensor information.")
    parser.add_argument("filepath", type=str, help="Path to the .pt TorchScript file to load.")
    args = parser.parse_args()

    load_and_print_tensors(args.filepath)


if __name__ == "__main__":
    main()
