# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from skbuild import setup

setup(
    name="nanovdb_editor",
    version="0.1.0",
    description="NanoVDB Editor Python Module",
    packages=['nanovdb_editor'],
    package_data={
        'nanovdb_editor': ['lib/*.dll', 'lib/*.so', 'lib/*.dylib', 'shaders/includes',
                           'shaders/compute/*', 'shaders/editor/*', 'shaders/imgui/*',
                           'shaders/nanovdb/*', 'shaders/raster/*']
    },
    cmake_install_dir="nanovdb_editor",
    cmake_source_dir="../CMakeLists.txt",
    cmake_args=[
        "-DSKBUILD=ON",
    ]
)
