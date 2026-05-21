# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Validate a NanoVDB grid and round-trip its checksum.

Phase 4c bound the grid-quality surface: tools.validateGrid (single
grid), tools.checkGrid -> (ok, error), tools.isValid, plus the
checksum helpers tools.evalChecksum / validateChecksum /
updateChecksum. This example walks through a typical "load and
verify" workflow.

Run with: python validate.py
"""
import nanovdb


def main():
    # A well-formed grid: createLevelSetSphere returns a NanoGrid<float>
    # with stats and checksum populated to the defaults.
    handle = nanovdb.tools.createLevelSetSphere(radius=10.0)
    grid = handle.grid()

    # Whole-handle validation. validateGrids returns a single bool.
    # The verbose=True flag prints failure details to std::cerr
    # (visible from Python via std::cerr -> sys.stderr on most stdlibs).
    all_ok = nanovdb.tools.validateGrids(
        handle, nanovdb.CheckMode.Default, verbose=False)
    print(f"validateGrids(handle, Default) = {all_ok}")

    # Per-grid validation, with a helpful message on failure.
    one_ok = nanovdb.tools.validateGrid(handle, 0, nanovdb.CheckMode.Full)
    print(f"validateGrid(handle, 0, Full)   = {one_ok}")

    # validateGrid returns False (no raise) on out-of-range gridID;
    # CheckMode.Disable short-circuits and always returns True.
    print(f"validateGrid(handle, 99)        = "
          f"{nanovdb.tools.validateGrid(handle, 99)}  (out of range)")
    print(f"validateGrid(handle, 99, Disable) = "
          f"{nanovdb.tools.validateGrid(handle, 99, nanovdb.CheckMode.Disable)}"
          f"  (Disable short-circuit)")

    # tools.checkGrid returns the structural check result plus a
    # human-readable error message (empty on success).
    ok, msg = nanovdb.tools.checkGrid(grid, nanovdb.CheckMode.Full)
    print(f"checkGrid(grid, Full)           = ok={ok}, msg={msg!r}")

    # tools.isValid is checkGrid + checksum verification, returning
    # one bool.
    print(f"isValid(grid, Default)          = "
          f"{nanovdb.tools.isValid(grid, nanovdb.CheckMode.Default)}")

    # Checksum round-trip. evalChecksum is non-mutating; updateChecksum
    # writes back into the grid header.
    cs1 = nanovdb.tools.evalChecksum(grid, nanovdb.CheckMode.Full)
    nanovdb.tools.updateChecksum(grid, nanovdb.CheckMode.Full)
    cs2 = nanovdb.tools.evalChecksum(grid, nanovdb.CheckMode.Full)
    print(f"evalChecksum equal after no-op update: {cs1 == cs2}")
    print(f"validateChecksum(grid, Full)    = "
          f"{nanovdb.tools.validateChecksum(grid, nanovdb.CheckMode.Full)}")


if __name__ == "__main__":
    main()
