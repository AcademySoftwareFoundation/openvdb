// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

// A simple minimal benchmark to serve as a starting point for fVDB benchmarks.

#include <torch/torch.h>

#include <benchmark/benchmark.h>

static void
BM_SimpleTensor(benchmark::State &state) {
    for (auto _: state) {
        torch::Tensor tensor = torch::eye(100);
    }
}

// Register the function as a benchmark
BENCHMARK(BM_SimpleTensor);

// Run the benchmark
BENCHMARK_MAIN();
