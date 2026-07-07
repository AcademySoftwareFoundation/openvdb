#!/usr/bin/env python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: run every example script and require a clean exit.

The examples are self-contained and degrade gracefully (exit 0 with a
skip message) when an optional dependency such as NumPy or OpenVDB is
missing, so a non-zero exit or an uncaught exception always indicates
a real breakage — typically the bindings drifting under the examples.

The script list is deliberately explicit rather than a glob so that a
missing listed example fails loudly and unrelated files placed under
examples/ are not executed.
"""

import os
import subprocess
import sys
import tempfile
import unittest

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "examples")

EXAMPLE_SCRIPTS = [
    "build_grid.py",
    "bulk_leaf_numpy.py",
    "collide_level_set.py",
    "index_grid_channels.py",
    "io_roundtrip.py",
    "load_inspect.py",
    "make_funny_nanovdb.py",
    "make_typed_grids.py",
    "node_manager.py",
    "openvdb_interop.py",
    "quantize.py",
    "raytrace_fog_volume.py",
    "raytrace_level_set.py",
    "validate.py",
]


class TestExamples(unittest.TestCase):
    pass


def _make_test(script_name):
    def test(self):
        script = os.path.join(EXAMPLES_DIR, script_name)
        self.assertTrue(os.path.isfile(script),
                        f"example script is missing: {script}")
        # A fresh cwd per run keeps any output files out of the source
        # tree (the examples themselves also write to tempdirs).
        result = subprocess.run(
            [sys.executable, script],
            cwd=tempfile.mkdtemp(prefix="nanovdb_example_"),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode, 0,
            f"{script_name} exited with {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}")

    return test


for _name in EXAMPLE_SCRIPTS:
    _test_name = "test_" + _name.replace(".py", "")
    setattr(TestExamples, _test_name, _make_test(_name))


if __name__ == "__main__":
    unittest.main()
