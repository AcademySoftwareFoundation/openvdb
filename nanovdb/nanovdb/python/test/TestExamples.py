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
import runpy
import subprocess
import sys
import tempfile
import unittest

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "examples")
RUN_EXAMPLE_ARG = "--run-example"

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


def _run_example(script):
    """Run one example after configuring dependent-DLL lookup on Windows."""
    dll_directory_handles = []
    if hasattr(os, "add_dll_directory"):
        for path in os.environ.get("PATH", "").split(os.pathsep):
            if os.path.isdir(path):
                try:
                    # Keep each handle alive until the example has finished.
                    dll_directory_handles.append(os.add_dll_directory(path))
                except OSError:
                    pass

    sys.argv = [script]
    sys.path[0] = os.path.dirname(os.path.abspath(script))
    runpy.run_path(script, run_name="__main__")


def _make_test(script_name):
    def test(self):
        script = os.path.join(EXAMPLES_DIR, script_name)
        self.assertTrue(os.path.isfile(script),
                        f"example script is missing: {script}")
        command = [sys.executable, script]
        env = os.environ.copy()
        if hasattr(os, "add_dll_directory"):
            # Match TestNanoVDB.py's in-tree OpenVDB DLL lookup. The child
            # starts in a temporary directory, so resolve this while the
            # parent is still in the CMake configuration directory.
            config = os.path.basename(os.getcwd())
            openvdb_dll_directory = os.path.abspath(os.path.join(
                os.getcwd(), os.pardir, os.pardir, os.pardir, os.pardir,
                "openvdb", "openvdb", config))
            env["PATH"] = os.pathsep.join(
                (openvdb_dll_directory, env.get("PATH", "")))
            # os.add_dll_directory() registrations are process-local, so run
            # through this file's child mode to register them in the process
            # that imports nanovdb.
            command = [
                sys.executable, os.path.abspath(__file__),
                RUN_EXAMPLE_ARG, script,
            ]
        # A fresh cwd per run keeps any output files out of the source
        # tree (the examples themselves also write to tempdirs).
        result = subprocess.run(
            command,
            cwd=tempfile.mkdtemp(prefix="nanovdb_example_"),
            env=env,
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
    if len(sys.argv) == 3 and sys.argv[1] == RUN_EXAMPLE_ARG:
        _run_example(sys.argv[2])
    else:
        unittest.main()
