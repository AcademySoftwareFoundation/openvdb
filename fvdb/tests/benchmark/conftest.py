import gc
import sys
import logging

import torch

from pytest_benchmark.fixture import BenchmarkFixture

from fvdb_benchmark.utils import create_l2_cache, flush_l2_cache

# pytest_plugins = ["pytest_fvdbench.plugin"]

def _make_runner(self, function_to_benchmark, args, kwargs):
    create_l2_cache()

    def runner(loops_range, timer=self._timer):
        gc_enabled = gc.isenabled()
        if self._disable_gc:
            gc.disable()
        tracer = sys.gettrace()
        sys.settrace(None)
        try:
            end = torch.cuda.Event(enable_timing=True)
            start = torch.cuda.Event(enable_timing=True)

            if loops_range:
                total = 0.0

                for _ in loops_range:
                    flush_l2_cache()
                    torch.cuda.synchronize()
                    torch.cuda._sleep(100_000)

                    start.record()
                    function_to_benchmark(*args, **kwargs)
                    end.record()
                    torch.cuda.synchronize()
                    total += start.elapsed_time(end) * 0.001

                return total

            else:
                flush_l2_cache()
                torch.cuda.synchronize()
                torch.cuda._sleep(1_000_000)
                start.record()
                result = function_to_benchmark(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                return start.elapsed_time(end) * 0.001, result
        finally:
            sys.settrace(tracer)
            if gc_enabled:
                gc.enable()

    return runner


logging.getLogger().debug("monkey patching BenchmarkFixture._make_runner")
BenchmarkFixture._make_runner = _make_runner
