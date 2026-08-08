#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Validate cross-references inside the OpenVDB documentation tree.

Run from the repo root (no arguments) or pass paths to validate explicitly.
The script scans Markdown (`.md`) and Doxygen narrative (`.txt`) files under
``doc/`` and checks two classes of cross-reference for staleness:

1. ``#include <path/to/header.h>`` lines that appear inside Markdown
   ``cpp``/``cuda``/``c``/``c++`` fenced code blocks or inside Doxygen
   ``@code``/``\\code`` blocks.  Each include is resolved to a real file in
   ``openvdb/``, ``openvdb_ax/``, ``openvdb_houdini/`` or ``nanovdb/``.

2. ``https://github.com/AcademySoftwareFoundation/openvdb/blob/<ref>/<path>``
   URLs.  The ``<path>`` portion is checked against the local checkout.

The script exits non-zero on any unresolved reference and prints a
``file:line`` style error so the failure is actionable in CI logs.  It is
intentionally dependency-free so it can run in a minimal Python container.

This guards against the class of bug tracked in issue #2161, where renames
under ``nanovdb/util/ -> nanovdb/tools/`` and ``nanovdb/util/IO.h ->
nanovdb/io/IO.h`` silently left the docs referencing paths that no longer
exist.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Map a top-level include namespace to one or more on-disk directories that it
# resolves against.  ``#include <nanovdb/io/IO.h>`` resolves to
# ``nanovdb/nanovdb/io/IO.h`` because the repo uses a double-named layout;
# ``#include <houdini_utils/ParmFactory.h>`` resolves to
# ``openvdb_houdini/openvdb_houdini/ParmFactory.h`` because the Houdini plugin
# CMake configuration synthesises a ``houdini_utils/`` subtree at build time
# (see ``openvdb_houdini/openvdb_houdini/CMakeLists.txt``).
INCLUDE_NAMESPACE_ROOTS = {
    "openvdb": ("openvdb/openvdb", "openvdb"),
    "openvdb_ax": ("openvdb_ax/openvdb_ax", "openvdb_ax"),
    "openvdb_houdini": ("openvdb_houdini/openvdb_houdini", "openvdb_houdini"),
    "nanovdb": ("nanovdb/nanovdb", "nanovdb"),
    "houdini_utils": ("openvdb_houdini/openvdb_houdini",),
}

# Top-level include path components that come from outside this repo.  When a
# header's leading component matches one of these, we skip resolution entirely.
# Houdini SDK headers (``UT/``, ``GA/``, ``GEO_*``, etc.) live in the Houdini
# install, not in our checkout.
EXTERNAL_INCLUDE_PREFIXES = (
    "boost/",
    "tbb/",
    "cuda",            # cuda_runtime.h, cuda.h, ...
    "thrust/",
    "Eigen/",
    "OpenEXR/",
    "Imath/",
    "openexr/",
    "GL/",
    # Houdini SDK prefixes
    "UT/",
    "GA/",
    "GU/",
    "GEO/",
    "GT/",
    "GR/",
    "OP/",
    "PRM/",
    "SOP/",
    "SYS/",
    "CH/",
    "DEP/",
    "PI/",
    "RE/",
    # Standard library headers (no '/' is the usual signal, but list a few
    # explicitly to be safe).
    "stdio.h",
    "iostream",
    "vector",
    "string",
    "memory",
    "cstdint",
)

# Match a Markdown fenced code block whose language is one of cpp/cuda/c/c++.
_MARKDOWN_FENCE_RE = re.compile(
    r"^```(?:cpp|cuda|c\+\+|c)\s*$(?P<body>.*?)^```\s*$",
    re.DOTALL | re.MULTILINE,
)

# Doxygen ``@code``/``\code`` ... ``@endcode``/``\endcode`` blocks.  The
# language hint (``@code{.cpp}``) is optional.
_DOXY_CODE_RE = re.compile(
    r"[@\\]code(?:\{[^}]*\})?\s*$(?P<body>.*?)^[@\\]endcode\s*$",
    re.DOTALL | re.MULTILINE,
)

_INCLUDE_RE = re.compile(r"#\s*include\s*<([^>\n]+)>")

# Match URLs pointing into a blob in this repo.  ``<ref>`` may be a branch
# name or a sha; ``<path>`` is the portion that must exist on disk today.
_GITHUB_BLOB_RE = re.compile(
    r"https?://github\.com/AcademySoftwareFoundation/openvdb/blob/"
    r"(?P<ref>[^/\s]+)/(?P<path>[^\s)\"'#?]+)"
)


def is_external_include(header: str) -> bool:
    """Return True for headers we cannot resolve from this repo."""
    head, _, _ = header.partition("/")
    if head in INCLUDE_NAMESPACE_ROOTS:
        return False
    return any(header.startswith(p) for p in EXTERNAL_INCLUDE_PREFIXES) or "/" not in header


def resolve_include(header: str) -> Path | None:
    """Resolve ``#include <header>`` to a path on disk, or None if not found."""
    head, _, rest = header.partition("/")
    roots = INCLUDE_NAMESPACE_ROOTS.get(head)
    if roots is None or not rest:
        return None
    for root in roots:
        candidate = REPO_ROOT / root / rest
        if candidate.is_file():
            return candidate
    return None


def iter_code_block_lines(text: str) -> Iterable[Tuple[int, str]]:
    """Yield (line_number, line_text) for every line inside a code block.

    Line numbers are 1-based and refer to the original ``text``.  Handles both
    Markdown fences (```cpp ...```) and Doxygen ``@code``/``\\code`` blocks.
    """
    for match in _MARKDOWN_FENCE_RE.finditer(text):
        body_start = match.start("body")
        yield from _lines_with_offsets(text, match.group("body"), body_start)
    for match in _DOXY_CODE_RE.finditer(text):
        body_start = match.start("body")
        yield from _lines_with_offsets(text, match.group("body"), body_start)


def _lines_with_offsets(full_text: str, body: str, body_offset: int) -> Iterable[Tuple[int, str]]:
    base_line = full_text.count("\n", 0, body_offset) + 1
    for i, line in enumerate(body.splitlines()):
        yield base_line + i, line


def find_doc_files(extra_paths: Sequence[str]) -> List[Path]:
    """Return the list of doc files to scan.

    Without arguments, scan ``doc/**/*.md`` and ``doc/**/*.txt``.  With
    arguments, scan only those files (resolved relative to the repo root).
    """
    if extra_paths:
        return [
            (Path(p) if Path(p).is_absolute() else REPO_ROOT / p).resolve()
            for p in extra_paths
        ]
    doc_dir = REPO_ROOT / "doc"
    return sorted(
        list(doc_dir.rglob("*.md")) + list(doc_dir.rglob("*.txt"))
    )


def check_file(path: Path) -> List[str]:
    """Return a list of human-readable error strings for the given doc file."""
    errors: List[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{path}: failed to read: {exc}"]

    rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path

    for line_no, line in iter_code_block_lines(text):
        for include_match in _INCLUDE_RE.finditer(line):
            header = include_match.group(1).strip()
            if is_external_include(header):
                continue
            if resolve_include(header) is None:
                errors.append(
                    f"{rel}:{line_no}: include <{header}> does not resolve "
                    "to a file in this checkout"
                )

    for url_match in _GITHUB_BLOB_RE.finditer(text):
        ref = url_match.group("ref")
        url_path = url_match.group("path")
        on_disk = REPO_ROOT / url_path
        if not on_disk.exists():
            # Compute the 1-based line number of the URL for the error report.
            line_no = text.count("\n", 0, url_match.start()) + 1
            errors.append(
                f"{rel}:{line_no}: github URL "
                f"AcademySoftwareFoundation/openvdb/blob/{ref}/{url_path} "
                "does not exist in this checkout"
            )

    return errors


def main(argv: Sequence[str]) -> int:
    files = find_doc_files(argv[1:])
    total_errors: List[str] = []
    for path in files:
        if not path.exists():
            total_errors.append(f"{path}: file does not exist")
            continue
        total_errors.extend(check_file(path))

    if total_errors:
        print("Documentation reference validation FAILED.", file=sys.stderr)
        print("", file=sys.stderr)
        for err in total_errors:
            print(err, file=sys.stderr)
        print("", file=sys.stderr)
        print(
            f"{len(total_errors)} broken reference(s) across "
            f"{len(files)} file(s).",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {len(files)} doc file(s) validated, no broken references.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
