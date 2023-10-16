import time
import math
import numpy as np
from matplotlib import pyplot as plt

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

from pycuda.compiler import SourceModule


# ----------------------------------------------------------------
# CUDA C function to calculate sum of all vector elements
module = SourceModule(
    """
    __global__ void sum_elements_cuda(float *result, float *vector, int size)
    {
        int start = threadIdx.x + blockIdx.x * blockDim.x;
        int step = gridDim.x * blockDim.x;
        
        float sum = 0;
        for (int i = start; i < size; i += step)
        {
            sum += vector[i];
        }
        
        atomicAdd(result, sum);
    }
    """
)


# ----------------------------------------------------------------
# Modifies functions so that they also return time of execution
def calculate_time_decorator(function):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()

        return result, end_time - start_time

    return wrapper


# ----------------------------------------------------------------
# Calculate using CUDA C function
@calculate_time_decorator
def sum_elements_gpu(vector: np.ndarray) -> float:
    N = vector.shape[0]

    # Convert numpy vector to GPUArray
    input_gpu_array = gpuarray.to_gpu(vector.astype(np.float32))

    # Initialize the output array with 1 element
    result_gpu_array = gpuarray.zeros(1, dtype=np.float32)

    # Set block and grid sizes
    k = min(N, 1000)
    block_size = (k, 1, 1)
    grid_size = (math.ceil(N / k), 1)

    # Get CUDA function
    sum_elements_cuda = module.get_function("sum_elements_cuda")

    # Apply CUDA function to the vector
    sum_elements_cuda(
        result_gpu_array,
        input_gpu_array,
        np.int32(N),
        block=block_size,
        grid=grid_size,
    )

    # Get the result
    result = result_gpu_array.get()[0]

    return result


# ----------------------------------------------------------------
# Calculate using built-in gpuarray function
@calculate_time_decorator
def sum_elements_gpu_v2(vector: np.ndarray) -> float:
    # Convert numpy vector to GPUArray
    vector_gpu = gpuarray.to_gpu(vector.astype(np.float32))
    return gpuarray.sum(vector_gpu)


# ----------------------------------------------------------------
# Calculate using built-in numpy function
@calculate_time_decorator
def sum_elements_numpy(vector: np.ndarray) -> float:
    return vector.sum()


# ----------------------------------------------------------------
# Calculate using loop
@calculate_time_decorator
def sum_elements_iterative(vector: np.ndarray) -> float:
    vector_sum = 0
    for index in range(vector.shape[0]):
        vector_sum += vector[index]

    return vector_sum


# ----------------------------------------------------------------
if __name__ == "__main__":
    vector_sizes = range(10_000, 1_000_000, 10_000)

    calculation_time_gpu = []
    calculation_time_gpu_v2 = []
    calculation_time_numpy = []
    calculation_time_iterative = []

    # Calculate sum of all vector elements for vectors with different sizes
    for index, N in enumerate(vector_sizes):
        print("----------------------------------------------------------------")
        print(f"Processing vector of size {N}...")

        # Create random 1d array
        vector = np.random.randn(N).astype(np.float32)

        # Calculate result with GPU
        result_gpu, time_gpu = sum_elements_gpu(vector)
        calculation_time_gpu.append(time_gpu)
        print(f"GPU: {time_gpu:0.3f} seconds")

        # Calculate result with built-in GPU function
        result_gpu_v2, time_gpu_v2 = sum_elements_gpu_v2(vector)
        calculation_time_gpu_v2.append(time_gpu_v2)
        print(f"GPU with built-in method: {time_gpu:0.3f} seconds")

        # Calculate result with numpy
        result_numpy, time_numpy = sum_elements_numpy(vector)
        calculation_time_numpy.append(time_numpy)
        print(f"Numpy: {time_numpy:0.3f} seconds")

        # Calculate result iteratively
        result_iterative, time_iterative = sum_elements_iterative(vector)
        calculation_time_iterative.append(time_iterative)
        print(f"Iterative: {time_iterative:0.3f} seconds")

        print()

    # Skip first element because GPU loads for a long time at the 1st iteration
    skip = 1
    vector_sizes = vector_sizes[skip:]
    calculation_time_gpu = calculation_time_gpu[skip:]
    calculation_time_gpu_v2 = calculation_time_gpu_v2[skip:]
    calculation_time_numpy = calculation_time_numpy[skip:]
    calculation_time_iterative = calculation_time_iterative[skip:]

    plt.plot(vector_sizes, calculation_time_gpu)
    plt.plot(vector_sizes, calculation_time_gpu_v2)
    plt.plot(vector_sizes, calculation_time_numpy)
    plt.plot(vector_sizes, calculation_time_iterative)

    plt.xlabel("Vector size (N)")
    plt.ylabel("Time of calculation, seconds")
    plt.legend(["GPU", "GPU with built-in method", "Numpy", "Iterative"])

    plt.show()
