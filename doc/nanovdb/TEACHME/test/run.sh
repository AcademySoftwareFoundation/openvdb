#!/usr/bin/env bash
# Drive the TEACHME test harness: reference-check -> extract -> compile.
#
# Usage: ./run.sh
# Exits nonzero if any reference is dangling or any compile-tagged block
# fails to compile.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHME="$(dirname "$HERE")"
GEN="$HERE/gen"
BUILD="$HERE/build"

MD=(
  "$TEACHME/nanovdb_user_lesson.md"
  "$TEACHME/nanovdb_user_cheatsheet.md"
  "$TEACHME/README.md"
)

echo "==> [1/3] reference check"
python3 "$HERE/teachme_tests.py" check-refs "${MD[@]}"

echo "==> [2/3] extract compile-tagged blocks"
rm -rf "$GEN"
mkdir -p "$GEN"
python3 "$HERE/teachme_tests.py" extract "${MD[@]}" --out "$GEN"

n_tu=$(find "$GEN" -name '*.cu' | wc -l | tr -d ' ')
if [ "$n_tu" -eq 0 ]; then
  echo "==> [3/3] compile: no compile-tagged blocks yet — nothing to build"
  echo "PASS (reference check only)"
  exit 0
fi

echo "==> [3/3] compile $n_tu translation unit(s)"
mkdir -p "$BUILD"
cmake -S "$HERE" -B "$BUILD" -DGEN_DIR="$GEN" -DCMAKE_BUILD_TYPE=Release \
  > "$BUILD/configure.log" 2>&1 || { cat "$BUILD/configure.log"; exit 1; }

# Cap parallelism: each TU pulls in the kitchen-sink fixture (most of NanoVDB's
# heavy headers), so nvcc is memory-hungry. Default to all cores locally; CI
# sets TEACHME_JOBS lower so the parallel nvcc processes stay within the
# runner's RAM (8 unbounded jobs OOM a 32 GB runner).
jobs="${TEACHME_JOBS:-$(nproc)}"

# Keep going (--keep-going / -k) so every failing TU is reported, not just the
# first. Capture cmake's status directly (don't pipe through tee, which would
# mask it).
set +e
cmake --build "$BUILD" --parallel "$jobs" -- -k > "$BUILD/build.log" 2>&1
status=$?
set -e
cat "$BUILD/build.log"
if [ "$status" -eq 0 ]; then
  echo "PASS: all $n_tu translation unit(s) compiled"
else
  echo "FAIL: one or more translation units did not compile (see above)"
  exit 1
fi
