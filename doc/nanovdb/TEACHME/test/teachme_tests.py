#!/usr/bin/env python3
"""TEACHME lesson test harness: reference validation + block extraction.

Two subcommands:

  check-refs <md...>            Validate every #include in code blocks resolves
                                against the NanoVDB include root. Reports
                                compile-coverage. Exits nonzero on dangling refs.

  extract <md...> --out <dir>   Extract compile-tagged code blocks into .cu
                                translation units under <dir>, ready for CMake.

See README.md for the block tagging convention.
"""

import argparse
import os
import re
import sys

# Repo layout: this file is at doc/nanovdb/TEACHME/test/teachme_tests.py, so the
# repo root is four levels up. The NanoVDB include root (the dir that contains
# the `nanovdb/` header subdir) is the repo's `nanovdb/`; OpenVDB's is `openvdb/`.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
NANOVDB_INCLUDE_ROOT = os.path.join(REPO_ROOT, "nanovdb")
OPENVDB_INCLUDE_ROOT = os.path.join(REPO_ROOT, "openvdb")
INCLUDE_ROOTS = [NANOVDB_INCLUDE_ROOT, OPENVDB_INCLUDE_ROOT]

FENCE_RE = re.compile(r"^```(\w+)?(.*)$")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*<([^>]+)>')

# Inline-code spans in prose that look like header paths, e.g.
# `nanovdb/tools/GridBuilder.h`. Used to catch stale "Read: ..." references.
PROSE_PATH_RE = re.compile(r'`((?:nanovdb|openvdb)/[\w./]+\.(?:h|cuh|cc|cpp))`')
# A line carrying this marker is exempt from prose-path checking (for
# intentionally-stale counter-examples).
STALE_OK = "teachme:stale-ok"


def resolves(inc):
    """True if an include path exists under any include root."""
    return any(os.path.isfile(os.path.join(r, inc)) for r in INCLUDE_ROOTS)

# Languages whose blocks we care about.
CODE_LANGS = {"cpp", "cu", "c++"}

# Opt-OUT model: every cpp/cu block is compiled unless tagged `no-compile`.
# A line that begins a file-scope definition forces file-scope emission even
# without an explicit `global` tag.
FILE_SCOPE_RE = re.compile(r'^\s*(template\b|struct\b|class\b|namespace\b|'
                           r'__global__\b|enum\b|using\b|typedef\b)')


class Block:
    __slots__ = ("path", "start_line", "lang", "tags", "lines")

    def __init__(self, path, start_line, lang, tags):
        self.path = path
        self.start_line = start_line
        self.lang = lang
        self.tags = tags
        self.lines = []

    @property
    def text(self):
        return "".join(self.lines)


def parse_blocks(path):
    """Yield Block objects for every fenced code block in a markdown file."""
    blocks = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    in_block = None
    for lineno, line in enumerate(lines, 1):
        m = FENCE_RE.match(line)
        if m:
            if in_block is None:
                lang = (m.group(1) or "").lower()
                tags = set(m.group(2).split())
                in_block = Block(path, lineno, lang, tags)
            else:
                blocks.append(in_block)
                in_block = None
            continue
        if in_block is not None:
            in_block.lines.append(line)
    if in_block is not None:
        sys.stderr.write(f"{path}: unterminated code fence opened at "
                         f"line {in_block.start_line}\n")
    return blocks


def is_exempt_include(inc):
    """Includes that are intentionally not real headers."""
    if "..." in inc:           # ellipsis placeholder, e.g. openvdb/...
        return True
    return False


def scan_prose_paths(path):
    """Yield (lineno, token) for header-path references in prose inline code,
    skipping lines inside fenced code blocks and lines marked stale-ok."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    in_fence = False
    for lineno, line in enumerate(lines, 1):
        if FENCE_RE.match(line) and line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or STALE_OK in line:
            continue
        for tok in PROSE_PATH_RE.findall(line):
            yield lineno, tok


def cmd_check_refs(args):
    code_blocks = 0
    compiled_blocks = 0
    dangling = []          # (path, lineno, inc, kind)
    for path in args.md:
        # 1) #include lines inside code blocks
        for b in parse_blocks(path):
            if b.lang not in CODE_LANGS:
                continue
            code_blocks += 1
            if "no-compile" not in b.tags:
                compiled_blocks += 1
            if "skip-refs" in b.tags:
                continue
            for off, line in enumerate(b.lines):
                im = INCLUDE_RE.match(line)
                if not im:
                    continue
                inc = im.group(1).strip()
                if is_exempt_include(inc):
                    continue
                if not resolves(inc):
                    dangling.append((path, b.start_line + 1 + off, inc, "include"))
        # 2) header-path references in prose ("Read: `nanovdb/.../X.h`")
        for lineno, tok in scan_prose_paths(path):
            if not resolves(tok):
                dangling.append((path, lineno, tok, "prose"))

    if dangling:
        sys.stderr.write("DANGLING REFERENCES:\n")
        for path, lineno, ref, kind in dangling:
            rel = os.path.relpath(path)
            sys.stderr.write(f"  {rel}:{lineno}: [{kind}] {ref!r} does not "
                             f"resolve under any include root\n")

    pct = (100.0 * compiled_blocks / code_blocks) if code_blocks else 0.0
    print(f"code blocks: {code_blocks}  compiled: {compiled_blocks} "
          f"({pct:.0f}%)  no-compile: {code_blocks - compiled_blocks}")
    if dangling:
        print(f"FAIL: {len(dangling)} dangling reference(s)")
        return 1
    print("OK: all references resolve")
    return 0


def split_block(block):
    """Split one block's lines into ordered (kind, lines) sub-segments.

    `kind` is 'global' (file scope) or 'fn' (function body). Explicit `global`
    / `main` tags force the whole block. Otherwise we brace-walk: a top-level
    definition (matched by FILE_SCOPE_RE at brace depth 0, e.g. a __global__
    kernel) becomes a file-scope chunk spanning until its braces balance;
    everything else at depth 0 (launches, statements) becomes a function chunk.
    This lets a single block mix a kernel definition and its launch.
    """
    if "global" in block.tags:
        return [("global", block.lines)]
    if "main" in block.tags:
        return [("main", block.lines)]

    segs = []
    fn_buf = []
    lines = block.lines
    i, n = 0, len(lines)

    def flush_fn():
        if fn_buf:
            segs.append(("fn", fn_buf[:]))
            fn_buf.clear()

    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("#"):       # preprocessor: neutral (hoisted later)
            fn_buf.append(line)
            i += 1
            continue
        if FILE_SCOPE_RE.match(line):      # start of a top-level definition
            flush_fn()
            chunk = []
            depth = 0
            saw_brace = False
            while i < n:
                l = lines[i]
                chunk.append(l)
                i += 1
                depth += l.count("{") - l.count("}")
                if "{" in l:
                    saw_brace = True
                if saw_brace and depth <= 0:
                    break
                if not saw_brace and l.rstrip().endswith(";"):
                    break              # brace-less decl, e.g. `using X = ...;`
            segs.append(("global", chunk))
            continue
        fn_buf.append(line)                # ordinary statement → function body
        i += 1
    flush_fn()
    return segs if segs else [("fn", lines)]


def cmd_extract(args):
    os.makedirs(args.out, exist_ok=True)
    # A TU has a list of (kind, lines, line_no) sub-segments + an openvdb flag.
    tus = []
    current = None

    for path in args.md:
        for b in parse_blocks(path):
            if b.lang not in CODE_LANGS:
                continue
            # Opt-out: compile everything except blocks tagged no-compile.
            if "no-compile" in b.tags:
                continue
            subsegs = [(k, lines, b.start_line) for (k, lines) in split_block(b)]
            cont = "continuation" in b.tags
            if cont and current is not None:
                current["segs"].extend(subsegs)
                if "openvdb" in b.tags:
                    current["openvdb"] = True
            else:
                current = {
                    "segs": list(subsegs),
                    "openvdb": "openvdb" in b.tags,
                    "src": os.path.basename(path),
                    "line": b.start_line,
                }
                tus.append(current)

    manifest = []
    for n, tu in enumerate(tus):
        stem = f"tu_{n:03d}_{os.path.splitext(tu['src'])[0]}_{tu['line']}"
        fname = os.path.join(args.out, stem + ".cu")
        # Hoist all preprocessor directives (#include/#define/#pragma) to file
        # scope, deduped in first-seen order, so function-wrapped segments
        # never carry an #include into block scope (illegal for headers that
        # define static __device__ functions).
        hoisted = []
        seen_hoist = set()
        seg_render = []   # (kind, body_lines, line)
        for kind, lines, line in tu["segs"]:
            body = []
            for ln in lines:
                if ln.lstrip().startswith("#"):
                    if ln not in seen_hoist:
                        seen_hoist.add(ln)
                        hoisted.append(ln)
                else:
                    body.append(ln)
            if body:
                seg_render.append((kind, body, line))

        out = ['#include "fixture.cuh"\n']
        out += hoisted
        out.append("\n")
        fn_count = 0
        for kind, body, line in seg_render:
            out.append(f"// --- from {tu['src']}:{line} ---\n")
            if kind in ("global", "main"):
                out += body
            else:  # fn
                out.append(f"static void _seg_{fn_count}() {{\n")
                out += body
                out.append("\n}\n")
                fn_count += 1
            out.append("\n")
        with open(fname, "w", encoding="utf-8") as f:
            f.writelines(out)
        manifest.append((stem, tu["openvdb"]))

    # Emit a manifest CMake can read: <stem> <needs_openvdb 0|1>
    with open(os.path.join(args.out, "manifest.txt"), "w") as f:
        for stem, ovdb in manifest:
            f.write(f"{stem} {1 if ovdb else 0}\n")

    n_ovdb = sum(1 for _, o in manifest if o)
    print(f"extracted {len(manifest)} translation unit(s) "
          f"({n_ovdb} need OpenVDB) into {args.out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("check-refs", help="validate #include references")
    p1.add_argument("md", nargs="+", help="markdown files")
    p1.set_defaults(func=cmd_check_refs)

    p2 = sub.add_parser("extract", help="extract compile-tagged blocks to .cu")
    p2.add_argument("md", nargs="+", help="markdown files")
    p2.add_argument("--out", required=True, help="output directory")
    p2.set_defaults(func=cmd_extract)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
