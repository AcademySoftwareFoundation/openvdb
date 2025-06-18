# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from nanovdb_editor import Compiler, Compute, pnanovdb_CompileTarget, MemoryBuffer
from ctypes import *

import os
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_SHADER = os.path.join(SCRIPT_DIR, "../test/shaders/test_matrix.slang")
TEST_SHADER_IN = os.path.join(SCRIPT_DIR, "../test/shaders/test_matrix_in.slang")

# Test data
MATRIX_SIZE = 4
constants_data = np.array([i for i in range(MATRIX_SIZE * MATRIX_SIZE)], dtype=np.float32)
array_dtype_out = np.dtype(np.int32)

compiler = Compiler()
compute = Compute(compiler)

### TEST setup
# test_matrix.slang - shader creates the matrix in row-major order
#                     and stores it in the output buffer in row-major order
# test_matrix_in.slang - shader reads the matrix from the row-major initilized constants buffer
#                        and stores it in the output buffer in row-major order

### TEST results
# test_matrix.slang - the output buffer is filled with the matrix in row-major order regardless of row major setting
# test_matrix_in.slang - when row major is False, the matrix is stored in column-major order in the output buffer

def run_test(test_shader, target, is_row_major=False):

    def validate_results(constants_data, result):
        for i, val in enumerate(constants_data):
            if test_shader == TEST_SHADER_IN and not is_row_major:
                    row = i // MATRIX_SIZE
                    col = i % MATRIX_SIZE
                    i = col * MATRIX_SIZE + row

            if result[i] != val:
                return False
        return True

    matrix_order = "row major" if is_row_major else "column major"

    if target == pnanovdb_CompileTarget.VULKAN:
        print(f">>> TESTING '{test_shader}'")
        print(f">>>  target: Vulkan, row major: {is_row_major}")

        input_data = np.zeros(len(constants_data), dtype=array_dtype_out)
        output_data = np.zeros(len(constants_data), dtype=array_dtype_out)

        compiler.compile_shader(test_shader, entry_point_name="computeMain", is_row_major=is_row_major)

        input_array = compute.create_array(input_data)
        constants_array = compute.create_array(constants_data)
        output_array = compute.create_array(output_data)

        success = compute.dispatch_shader_on_array(
            test_shader,
            (MATRIX_SIZE, MATRIX_SIZE, 1),
            input_array,
            constants_array,
            output_array
        )
        if success:
            result = compute.map_array(output_array, array_dtype_out)
            print(result)

            if validate_results(constants_data, result):
                print(f"=== Vulkan shader test ({matrix_order}) was successful")
            else:
                print(f"Error: Vulkan shader test ({matrix_order}) failed!")
                success = False
        else:
            print("Error: Vulkan shader test execution failed!")

        compute.unmap_array(output_array)

        compute.destroy_array(input_array)
        compute.destroy_array(constants_array)
        compute.destroy_array(output_array)

        return success

    elif target == pnanovdb_CompileTarget.CPU:
        print(f">>> TESTING '{test_shader}'")
        print(f">>>  target: CPU, row major: {is_row_major}")

        input_data = np.zeros(len(constants_data), dtype=array_dtype_out)
        output_data = np.zeros(len(constants_data), dtype=array_dtype_out)

        compiler.compile_shader(
            test_shader,
            entry_point_name="computeMain",
            compile_target=pnanovdb_CompileTarget.CPU,
            is_row_major=is_row_major
        )

        class Constants(Structure):
            """Definition equivalent to constants_t in the shader."""
            _fields_ = [
                ("matrix", c_float * 16),
            ]

        constants = Constants()
        constants.matrix = (c_float * 16)(*[float(x) for x in constants_data])

        class UniformState(Structure):
            _fields_ = [
                ("data_in", MemoryBuffer),
                ("constants", c_void_p),        # Constant buffer must be passed as a pointer
                ("data_out", MemoryBuffer),
            ]

        uniform_state = UniformState()
        uniform_state.data_in = MemoryBuffer(input_data)
        uniform_state.constants = c_void_p(addressof(constants))
        uniform_state.data_out = MemoryBuffer(output_data)

        success = compiler.execute_cpu(
            test_shader,
            (1, 1, 1),
            None,
            c_void_p(addressof(uniform_state))
        )
        if success:
            data_out = uniform_state.data_out.to_ndarray(array_dtype_out)
            print(data_out)

            if validate_results(constants_data, data_out):
                print(f"=== CPU shader test ({matrix_order}) was successful")
            else:
                print(f"Error: CPU shader test ({matrix_order}) failed!")
                success = False
        else:
            print("Error: CPU shader test execution failed!")

        return success


if __name__ == "__main__":

    print(f"Current Process ID (PID): {os.getpid()}")

    compiler.create_instance()
    compute.device_interface().create_device_manager()
    compute.device_interface().create_device()

    shaders = [TEST_SHADER, TEST_SHADER_IN]
    configs = [
        (pnanovdb_CompileTarget.VULKAN, False),
        (pnanovdb_CompileTarget.VULKAN, True),
        (pnanovdb_CompileTarget.CPU, False),
        (pnanovdb_CompileTarget.CPU, True),
    ]
    results = {shader: [] for shader in shaders}

    for shader in shaders:
        for config in configs:
            results[shader].append(run_test(shader, config[0], config[1]))

    for shader in shaders:
        print(f">>> RESULTS FOR '{shader}'")
        if all(results[shader]):
            print("All tests passed!")
        else:
            print("Some tests failed!")
            for i, config in enumerate(configs):
                target = "Vulkan" if config[0] == pnanovdb_CompileTarget.VULKAN else "CPU"
                matrix_order = "row major" if config[1] else "column major"
                print(f"{target} - {matrix_order}: {results[shader][i]}")
