#!/usr/bin/env python3
"""
Shrink Wrap - USD Mesh to Level Set Conversion

Reads a mesh from a USD file, converts it to OpenVDB level sets,
and optionally converts back to mesh or saves as .vdb file.

Dependencies:
    - openvdb (Python bindings)
    - numpy
    - usd-core (install with: pip install usd-core)

Example usage:
    # Convert USD mesh to VDB level set (creates input-out.vdb)
    python shrink_wrap.py input.usd --voxel 0.1

    # Convert to mesh with adaptivity (creates input-out.usd)
    python shrink_wrap.py input.usd --to-mesh --voxel 0.1 --adaptivity 0.005

    # Use dimension-based sizing
    python shrink_wrap.py input.usd --dim 256
"""

import argparse
import sys
import os
import numpy as np

# Import required dependencies with helpful error messages
try:
    import openvdb
except ImportError:
    print("Error: openvdb module not found.", file=sys.stderr)
    print("Build OpenVDB Python bindings with: cmake -DOPENVDB_BUILD_PYTHON_MODULE=ON", file=sys.stderr)
    sys.exit(1)

try:
    from usd_utils import USDInterface
except ImportError:
    print("Error: usd_utils module not found.", file=sys.stderr)
    print("Make sure usd_utils.py is in openvdb/openvdb/python/examples/", file=sys.stderr)
    print("Install USD support with: pip install usd-core", file=sys.stderr)
    sys.exit(1)


def generate_output_filename(input_path, to_mesh=False):
    """Generate output filename from input path.

    Output is created in the same directory as the input file.

    Args:
        input_path: Input USD file path
        to_mesh: Whether converting to mesh (determines extension)

    Returns:
        Output filename with format: <input_dir>/<basename>-out.vdb or .usd
    """
    # Get directory and base filename without extension
    input_dir = os.path.dirname(input_path)
    base = os.path.splitext(os.path.basename(input_path))[0]

    # Determine extension based on mode
    ext = "usd" if to_mesh else "vdb"

    # Generate output filename in same directory as input
    output_filename = f"{base}-out.{ext}"
    if input_dir:
        output_filename = os.path.join(input_dir, output_filename)

    return output_filename


def read_mesh_from_usd(usd_path, verbose=False):
    """Read mesh data from USD file using USDInterface.

    This function loads all meshes from the USD file, applies world transformations,
    merges them, and triangulates all faces.

    Args:
        usd_path: Path to USD file
        verbose: Whether to print loading information (default: False)

    Returns:
        Tuple of (vertices, triangles) where:
        - vertices: (N, 3) float32 numpy array
        - triangles: (M, 3) numpy array

    Raises:
        ValueError: If no mesh found in USD file or if USD loading fails
    """
    try:
        usd_interface = USDInterface(usd_path, verbose=verbose)
    except Exception as e:
        raise ValueError(f"Failed to load USD file: {e}")

    vertices = usd_interface.merged_verts.astype(np.float32)
    triangles = usd_interface.merged_faces.astype(np.uint32)

    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError(f"No valid mesh data found in {usd_path}")

    return vertices, triangles


def polygon_soup_to_level_set(vertices, triangles, quads=None,
                              min_voxel_size=None, dim=None,
                              bbox_min=None, bbox_max=None,
                              erode=8.0, thres=0.0, half_width=3.0):
    """Convert polygon soup to level set grids (LOD pyramid).

    Args:
        vertices: (N, 3) float32 numpy array
        triangles: (M, 3) uint32 numpy array or None
        quads: (K, 4) uint32 numpy array or None
        min_voxel_size: Minimum voxel size for finest grid (required for voxel mode)
        dim: Grid dimension (alternative to min_voxel_size)
        bbox_min: Bounding box minimum (optional, tuple of 3 floats)
        bbox_max: Bounding box maximum (optional, tuple of 3 floats)
        erode: Maximum deformation allowed (default: 8.0)
        thres: Closing threshold (default: 0.0)
        half_width: Narrow band half-width in voxels (default: 3.0)

    Returns:
        List of openvdb.FloatGrid objects (finest to coarsest)

    Raises:
        ValueError: If no triangles or quads provided, or if required parameters missing
    """
    # Validate inputs
    if triangles is None and quads is None:
        raise ValueError("Must provide triangles or quads (or both)")

    # Auto-compute bbox if not provided
    if bbox_min is None or bbox_max is None:
        bbox_min = tuple(vertices.min(axis=0))
        bbox_max = tuple(vertices.max(axis=0))

    # Choose overload based on provided parameters
    if dim is not None:
        # Overload 2: dimension + bbox
        grids = openvdb.FloatGrid.convertPolygonSoupToLevelSet(
            dim=dim,
            bboxMin=bbox_min,
            bboxMax=bbox_max,
            points=vertices,
            triangles=triangles,
            quads=quads,
            erode=erode,
            thres=thres,
            halfWidth=half_width
        )

    elif min_voxel_size is not None:
        # Overload 3: minVoxelSize + bbox
        grids = openvdb.FloatGrid.convertPolygonSoupToLevelSet(
            minVoxelSize=min_voxel_size,
            bboxMin=bbox_min,
            bboxMax=bbox_max,
            points=vertices,
            triangles=triangles,
            quads=quads,
            erode=erode,
            thres=thres,
            halfWidth=half_width
        )

    else:
        raise ValueError("Must specify either --voxel or --dim parameter")

    return grids


def levelset_to_mesh(grid, isovalue=0.0, adaptivity=0.0):
    """Convert level set grid back to mesh.

    Args:
        grid: openvdb.FloatGrid
        isovalue: Isosurface value (default: 0.0 for zero crossing)
        adaptivity: Mesh simplification (0=none, 1=max, default: 0.0)

    Returns:
        (points, triangles, quads) numpy arrays
    """
    points, triangles, quads = grid.convertToPolygons(
        isovalue=isovalue,
        adaptivity=adaptivity
    )
    return points, triangles, quads


def write_grids_to_vdb(vdb_path, grids, grid_names=None):
    """Write list of grids to .vdb file.

    Args:
        vdb_path: Output .vdb file path
        grids: List of openvdb grids
        grid_names: Optional list of names for grids (default: "grid_0", "grid_1", ...)

    """
    # Set grid names
    if grid_names is None:
        grid_names = [f"grid_{i}" for i in range(len(grids))]

    for grid, name in zip(grids, grid_names):
        grid.name = name

    # Write all grids to single file
    openvdb.write(vdb_path, grids)


def main():
    parser = argparse.ArgumentParser(
        description="Shrink Wrap - Convert USD mesh to OpenVDB level sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert USD mesh to VDB with voxel size (creates input-out.vdb)
  %(prog)s input.usd --voxel 0.1

  # Convert to mesh with adaptivity (creates input-out.usd)
  %(prog)s input.usd --to-mesh --voxel 0.1 --adaptivity 0.005

  # Use dimension-based sizing
  %(prog)s input.usd --dim 256
        """
    )

    # Input/output
    parser.add_argument("input_usd", help="Input USD mesh file")

    # Conversion mode
    parser.add_argument("--to-mesh", action="store_true",
                       help="Convert grids back to mesh (output as USD)")

    # Level set parameters
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--voxel", type=float,
                      help="Voxel size for level set (required, uses minVoxelSize + bbox)")
    group.add_argument("--dim", type=int,
                      help="Grid dimension for finest level (uses dimension + bbox)")

    parser.add_argument("--bbox-min", type=float, nargs=3, metavar=("X", "Y", "Z"),
                       help="Bounding box minimum (optional, auto-computed from mesh if not provided)")
    parser.add_argument("--bbox-max", type=float, nargs=3, metavar=("X", "Y", "Z"),
                       help="Bounding box maximum (optional, auto-computed from mesh if not provided)")

    parser.add_argument("--erode", type=float, default=8.0,
                       help="Maximum deformation allowed (default: 8.0)")
    parser.add_argument("--threshold", type=float, default=0.0,
                       help="Closing threshold (default: 0.0)")
    parser.add_argument("--half-width", type=float, default=3.0,
                       help="Narrow band half-width in voxels (default: 3.0)")

    # Mesh conversion parameters
    parser.add_argument("--adaptivity", type=float, default=0.0,
                       help="Mesh adaptivity for convertToPolygons (0-1, default: 0.0)")
    parser.add_argument("--isovalue", type=float, default=0.0,
                       help="Isovalue for mesh extraction (default: 0.0)")

    # Output options
    parser.add_argument("--finest-only", action="store_true",
                       help="Only write finest grid (do not write entire LOD pyramid)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_usd):
        print(f"Error: Input file not found: {args.input_usd}", file=sys.stderr)
        return 1

    # Generate output filename based on input and mode
    args.output = generate_output_filename(args.input_usd, args.to_mesh)
    print(f"Output file: {args.output}")

    # Step 1: Read mesh from USD
    print(f"Reading mesh from {args.input_usd}...")
    try:
        vertices, triangles = read_mesh_from_usd(args.input_usd, verbose=False)
    except Exception as e:
        print(f"Error reading USD file: {e}", file=sys.stderr)
        return 1

    print(f"  Vertices: {len(vertices)}")
    print(f"  Triangles: {len(triangles)}")

    # Step 2: Convert to level sets
    print("\nConverting to level sets...")

    bbox_min = tuple(args.bbox_min) if args.bbox_min else None
    bbox_max = tuple(args.bbox_max) if args.bbox_max else None

    try:
        grids = polygon_soup_to_level_set(
            vertices, triangles,
            min_voxel_size=args.voxel,
            dim=args.dim,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            erode=args.erode,
            thres=args.threshold,
            half_width=args.half_width
        )
    except Exception as e:
        print(f"Error converting to level set: {e}", file=sys.stderr)
        return 1

    print(f"  Generated {len(grids)} LOD grid(s):")
    for i, grid in enumerate(grids):
        voxel_size = grid.transform.voxelSize()[0]
        active_count = grid.activeVoxelCount()
        print(f"    Grid {i}: voxel_size={voxel_size:.4f}, active_voxels={active_count:,}")

    # Step 3: Output
    if args.to_mesh:
        # Convert back to mesh and save as USD
        print(f"\nConverting grids to mesh...")

        if args.finest_only:
            grids_to_convert = [grids[0]]
        else:
            grids_to_convert = grids

        try:
            # Convert each grid and write separate USD files (or combine)
            if len(grids_to_convert) == 1:
                grid = grids_to_convert[0]
                points, tris, quads_out = levelset_to_mesh(grid, args.isovalue, args.adaptivity)

                print(f"  Output mesh: {len(points)} vertices, {len(tris)} triangles, {len(quads_out)} quads")

                # Write using USDInterface (preserves quads)
                USDInterface.write_file(args.output, points, tris, quads_out)
                print(f"\nWrote mesh to {args.output}")
            else:
                # Multiple grids - write each to separate USD
                base, ext = os.path.splitext(args.output)
                for i, grid in enumerate(grids_to_convert):
                    points, tris, quads_out = levelset_to_mesh(grid, args.isovalue, args.adaptivity)
                    output_path = f"{base}_lod{i}{ext}"

                    USDInterface.write_file(output_path, points, tris, quads_out, mesh_name=f"Mesh_LOD{i}")
                    print(f"  Wrote LOD {i} to {output_path}")
        except Exception as e:
            print(f"Error converting to mesh: {e}", file=sys.stderr)
            return 1
    else:
        # Save as .vdb file
        print(f"\nWriting grids to {args.output}...")

        if args.finest_only:
            grids_to_save = [grids[0]]
            grid_names = ["level_set"]
        else:
            grids_to_save = grids
            grid_names = [f"level_set_lod{i}" for i in range(len(grids))]

        try:
            write_grids_to_vdb(args.output, grids_to_save, grid_names)
            print(f"  Wrote {len(grids_to_save)} grid(s) to {args.output}")
        except Exception as e:
            print(f"Error writing VDB file: {e}", file=sys.stderr)
            return 1

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
