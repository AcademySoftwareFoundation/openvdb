# TEACHME test harness

Keeps the code in the lesson honest. Two gates:

1. **Reference validation** (`teachme_tests.py check-refs`) — every
   `#include <...>` in a code block must resolve to a real header in the
   NanoVDB tree (or be explicitly exempt). Catches stale-path rot — e.g. a
   header that was renamed out from under the docs.

2. **Compilation** (`teachme_tests.py extract` + CMake) — code blocks tagged
   as compilable are extracted into `.cu` translation units and compiled
   (compile-only, `-c`) against the real NanoVDB headers. Catches API-shape
   drift — wrong namespace, a free function that's actually a class, a
   renamed method, a changed signature.

Run both with `./run.sh` from this directory.

## Why compile-only

The harness compiles (`nvcc -c`) but does not link or run. NanoVDB is
header-only and the goal is to catch *API* errors (does this call name a
real symbol with a compatible signature?), which all surface at compile
time. Skipping the link step means a fixture header can declare the
"given" symbols a fragment assumes (`grid`, `acc`, device pointers) as
`extern` globals — unresolved at link, but fully type-checked at compile.

Everything compiles as CUDA (`.cu`) because even host-only examples pull
in CUDA code paths (under `__CUDACC__`) when they include headers like
`NodeManager.h` or `io/IO.h`.

## Block tagging

Code blocks are fenced with an info string: a language followed by
whitespace-separated tags, e.g.

    ```cpp compile
    ```cpp compile-global continuation
    ```cpp skip-refs

**Opt-out model:** every `cpp` / `cu` block is compiled by default. A block
is wrapped in a function body unless it opens a top-level definition
(`__global__`, `template`, `struct`, `class`, `namespace`, `enum`, `using`,
`typedef`), in which case it is auto-emitted at file scope. Tags only adjust
the defaults:

| Tag            | Meaning                                                                                  |
|----------------|------------------------------------------------------------------------------------------|
| *(none)*       | Compiled. Auto: file scope if it opens a top-level definition, else function-wrapped.    |
| `no-compile`   | Reference-check includes only; do **not** compile. For illustrative/partial fragments.   |
| `global`       | Force file-scope emission (override the auto-detection).                                  |
| `main`         | Block is already a complete program (its own `main`); compile verbatim.                  |
| `continuation` | Append this block to the previous block's translation unit (shares its symbols).         |
| `openvdb`      | Block needs OpenVDB; compiled only when OpenVDB is found.                                 |
| `skip-refs`    | Do **not** validate includes in this block — for intentionally-stale counter-examples.   |

A translation unit is a sequence of segments. A non-`continuation` block
starts a new TU; each following `continuation` block adds a segment. At emit
time, file-scope segments go to file scope (in order) and function segments
each get their own wrapper function — so a kernel definition followed by a
`continuation` launch line compose into one valid TU.

Every TU includes `fixture.cuh`, which pulls in the common NanoVDB headers
(a deliberately broad "kitchen-sink" set, so terse fragments compile without
restating includes) and declares the stand-in "given" symbols (`grid`,
`acc`, device pointers, etc.) as `extern`. Including the heavy `tools/cuda/*`
headers in every TU makes per-TU compilation slower; that's an accepted
trade for letting fragments stay terse. If the suite grows large, switch to
per-TU includes driven by each block's own `#include` lines.

## Coverage

`check-refs` reports how many code blocks are compile-tagged vs
reference-only, so the fraction of the lesson that is actually
compile-gated is visible (and ideally climbing).

## Exemptions

Includes that are intentionally not real are skipped by the validator:
- ellipsis placeholders such as `#include <openvdb/...>`
- any include inside a `skip-refs` block (e.g. the deprecated
  `nanovdb/util/cuda/CudaDeviceBuffer.h` shown as a "this path is stale"
  counter-example)

## Files

- `teachme_tests.py` — the extractor + reference validator
- `fixture.cuh` — common headers + `extern` stand-in symbols
- `CMakeLists.txt` — compiles the generated TUs
- `run.sh` — driver: check-refs → extract → cmake configure+build → report
