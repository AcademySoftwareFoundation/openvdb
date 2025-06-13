# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from nanovdb_editor import Compiler, Compute, Editor

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_NANOVDB = os.path.join(SCRIPT_DIR, "../data/dragon_node2.nvdb")

if __name__ == "__main__":

    compiler = Compiler()
    compiler.create_instance()

    compute = Compute(compiler)
    nvdb_array = compute.load_nanovdb(TEST_NANOVDB)

    compute.device_interface().create_device_manager()
    compute.device_interface().create_device()

    editor = Editor(compute, compiler)
    editor.add_nanovdb(nvdb_array)

    editor.show()

    compute.destroy_array(nvdb_array)
