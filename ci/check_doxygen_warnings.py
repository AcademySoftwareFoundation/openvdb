#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Filter the Doxygen warning log and gate CI on what remains.

Doxygen has no per-file or per-pattern warning suppression mechanism; its
``WARN_AS_ERROR`` knob is all-or-nothing.  This script implements per-file
suppressions on top of it so we can enforce a strict "no new warnings"
gate without blocking on the existing backlog.

Workflow:

1. Doxygen runs as usual and writes every warning to
   ``${CMAKE_CURRENT_BINARY_DIR}/doxygen.warnings.log``
   (configured by ``DOXYGEN_WARN_LOGFILE`` in ``doc/CMakeLists.txt``).
2. This script reads the log and the suppression file
   ``doc/.doxygen_warning_suppress.txt`` (Python regex per line).
3. Suppressed lines are dropped; remaining lines are printed and counted.
4. The script exits non-zero if any warnings survive the filter, unless
   ``--soft`` is passed (in which case it always exits 0 but still prints).

The suppression file is the *backlog* of known doc rot we've consciously
chosen not to fix yet.  Every entry is a license to ship a regression in
a specific file; whittle it down over time.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Pattern, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO_ROOT / "build" / "doc" / "doxygen.warnings.log"
DEFAULT_SUPPRESSIONS = REPO_ROOT / "doc" / ".doxygen_warning_suppress.txt"


def load_suppressions(path: Path) -> List[Tuple[str, Pattern[str]]]:
    """Parse the suppression file into a list of (raw_pattern, compiled) pairs."""
    if not path.exists():
        return []
    patterns: List[Tuple[str, Pattern[str]]] = []
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            patterns.append((stripped, re.compile(stripped)))
        except re.error as exc:
            sys.stderr.write(
                f"{path}:{line_no}: invalid regex {stripped!r}: {exc}\n"
            )
            sys.exit(2)
    return patterns


def classify(line: str) -> str:
    """Best-effort categorisation so the summary is actually readable."""
    if "unable to resolve link" in line:
        return "unresolved \\link"
    if "is not found in the argument list" in line or "is not documented" in line:
        return "@param mismatch / undocumented param"
    if "unknown command" in line:
        return "unknown Doxygen command"
    if "recursive class relation" in line:
        return "recursive template detection"
    if "explicit link request" in line:
        return "broken \\link to #define"
    if "<div>" in line or "</div>" in line:
        return "mismatched HTML tags"
    if "end of comment block while expecting" in line:
        return "unterminated HTML in comment"
    if "documented symbol" in line and "was not declared" in line:
        return "documented but undeclared symbol"
    if "Problems running latex" in line:
        return "latex not installed"
    if "is not an input file" in line:
        return "@file not in input set"
    return "other"


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Doxygen warning log (default: {DEFAULT_LOG.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--suppressions",
        type=Path,
        default=DEFAULT_SUPPRESSIONS,
        help=(
            "Regex-per-line suppression file "
            f"(default: {DEFAULT_SUPPRESSIONS.relative_to(REPO_ROOT)})"
        ),
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Print the diagnostics but never fail (informational mode).",
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=50,
        help="How many surviving warning lines to print verbatim.",
    )
    args = parser.parse_args(argv[1:])

    if not args.log.exists():
        print(f"warning log not found at {args.log}; nothing to check.")
        return 0

    suppressions = load_suppressions(args.suppressions)
    raw_lines = [
        line.rstrip("\n")
        for line in args.log.read_text(errors="replace").splitlines()
        if line.strip()
    ]

    suppressed: List[str] = []
    remaining: List[str] = []
    suppression_hits: Counter = Counter()
    for line in raw_lines:
        matched_pattern = next(
            (raw for raw, rx in suppressions if rx.search(line)),
            None,
        )
        if matched_pattern is None:
            remaining.append(line)
        else:
            suppressed.append(line)
            suppression_hits[matched_pattern] += 1

    # Summary header
    print(f"Doxygen warning log: {args.log}")
    print(f"Suppression file:    {args.suppressions}")
    print(
        f"Totals: {len(raw_lines)} raw, "
        f"{len(suppressed)} suppressed, "
        f"{len(remaining)} surviving."
    )

    if suppression_hits:
        print("")
        print("Suppression hits (helps you see when an entry is stale):")
        for pattern, count in suppression_hits.most_common():
            print(f"  {count:4d}  {pattern}")
        # Flag any suppression patterns that did not match anything: they are
        # dead code and should be removed from the suppression file.
        unused = [
            raw for raw, _ in suppressions if raw not in suppression_hits
        ]
        if unused:
            print("")
            print(
                "WARNING: the following suppression patterns matched nothing "
                "and can probably be removed:"
            )
            for raw in unused:
                print(f"  {raw}")

    if not remaining:
        print("")
        print("No surviving warnings. Doxygen output is clean.")
        return 0

    # Category breakdown for surviving warnings (the real signal).
    categories: Counter = Counter(classify(line) for line in remaining)
    print("")
    print("Surviving warning categories (these will fail the build):")
    for cat, count in categories.most_common():
        print(f"  {count:4d}  {cat}")

    preview = remaining[: args.preview_lines]
    print("")
    print(f"First {len(preview)} surviving warning line(s):")
    for line in preview:
        print(f"  {line}")
    if len(remaining) > len(preview):
        print(f"  ... and {len(remaining) - len(preview)} more (see uploaded artifact)")

    if args.soft:
        print("")
        print(
            "--soft mode: would have failed with "
            f"{len(remaining)} surviving warning(s) but exiting 0."
        )
        return 0

    print("")
    print(
        f"FAILED: {len(remaining)} Doxygen warning(s) not covered by "
        f"{args.suppressions.name}."
    )
    print(
        "Either fix the warnings, or (for genuinely pre-existing tech debt) "
        "add a narrowly-scoped regex to the suppression file with a comment "
        "explaining why."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
