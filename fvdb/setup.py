# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from pathlib import Path

from skbuild import setup


def get_cwd() -> Path:
    return Path(__file__).resolve().parent


def retrieve_version(file_path: Path = get_cwd() / "pyproject.toml") -> str:
    """Read version from pyproject.toml"""
    try:
        with open(file_path, "rb") as f:
            import tomllib

            pyproject = tomllib.load(f)
            return pyproject.get("project", {}).get("version", "0.0.0")
    except (FileNotFoundError, KeyError):
        return "0.0.0"


setup(
    name="fvdb",
    version=retrieve_version(),
    description="A deep learning framework for sparse, large-scale, high-performance spatial intelligence",
    author="The fvdb team",
    license="Apache-2.0",
    packages=["fvdb"],
    python_requires=">=3.10",
    cmake_args=[
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DBUILD_SHARED_LIBS=ON",
        "-DBUILD_TESTING=OFF",  # This is for scikit-build's own tests
    ],
)
