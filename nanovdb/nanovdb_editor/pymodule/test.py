# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Minimal test to ensure the editor can be created and shown

from nanovdb_editor import Compiler, Compute, Editor, Raster

if __name__ == "__main__":

    compiler = Compiler()
    compiler.create_instance()

    compute = Compute(compiler)
    compute.device_interface().create_device_manager()
    compute.device_interface().create_device()

    raster = Raster(compute)

    editor = Editor(compute, compiler)
    editor.show()
