import time
import numpy as np
from matplotlib import pyplot as plt

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


# ----------------------------------------------------------------
# CUDA C function to multiply matrices of size NxN
mod = SourceModule(
    """
    __global__ void multiply_matrices_cuda(float *C, float *A, float *B, int N)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        float sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
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
# Multipy using CUDA C function
@calculate_time_decorator
def multiply_matrices_gpu(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    # Check if matrices cannot be multiplied
    assert matrix1.shape[1] == matrix2.shape[0]

    # Initialize the output matrix C with zeros
    result_matrix = np.zeros((matrix1.shape[0], matrix2.shape[1]), dtype=np.float32)

    # Set block and grid sizes
    block_size = (16, 16, 1)
    grid_size = (1, 1)

    # Get CUDA function
    multiply_matrices_cuda = mod.get_function("multiply_matrices_cuda")

    # Apply CUDA function to the arrays
    multiply_matrices_cuda(
        drv.Out(result_matrix),
        drv.In(matrix1),
        drv.In(matrix2),
        np.int32(matrix1.shape[0]),
        block=block_size,
        grid=grid_size,
    )

    return result_matrix


# ----------------------------------------------------------------
# Multiply using numpy matrix operations
@calculate_time_decorator
def multiply_matrices_numpy(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    # Check if matrices cannot be multiplied
    assert matrix1.shape[1] == matrix2.shape[0]

    return np.matmul(matrix1, matrix2)  # matrix1 @ matrix2


# ----------------------------------------------------------------
# Multiply using 3D loops
@calculate_time_decorator
def multiply_matrices_iterative(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    # Check if matrices cannot be multiplied
    assert matrix1.shape[1] == matrix2.shape[0]

    # Initialize the output matrix C with zeros
    result_matrix = np.zeros((matrix1.shape[0], matrix2.shape[1]), dtype=np.float32)

    for y in range(matrix1.shape[0]):
        for x in range(matrix2.shape[1]):
            element = 0
            for k in range(matrix1.shape[1]):
                element += matrix1[y, k] * matrix2[k, x]
            result_matrix[y, x] = element

    return result_matrix


# ----------------------------------------------------------------
matrix_sizes = [100, 200, 300, 500, 800, 1000, 1200, 1500, 1800, 2000]

calculation_time_gpu = []
calculation_time_numpy = []
calculation_time_iterative = []

# Multiply matrices of different sizes
for index, N in enumerate(matrix_sizes):
    print("----------------------------------------------------------------")
    print(f"Processing matrices of size {N}...")

    # Create random input matrices
    matrix1 = np.random.randn(N, N).astype(np.float32)
    matrix2 = np.random.randn(N, N).astype(np.float32)

    # Calculate result wit GPU
    result_gpu, time_gpu = multiply_matrices_gpu(matrix1, matrix2)
    calculation_time_gpu.append(time_gpu)
    print(f"GPU: {time_gpu:0.3f} seconds")

    # Calculate result with numpy
    result_numpy, time_numpy = multiply_matrices_gpu(matrix1, matrix2)
    calculation_time_numpy.append(time_numpy)
    print(f"Numpy: {time_numpy:0.3f} seconds")

    # # Calculate result iteratively
    # result_iterative, time_iterative = multiply_matrices_iterative(matrix1, matrix2)
    # calculation_time_iterative.append(time_iterative)
    # print(f"Iterative: {time_iterative:0.3f} seconds")

    print()


plt.plot(matrix_sizes, calculation_time_gpu)
plt.plot(matrix_sizes, calculation_time_numpy)
# plt.plot(matrix_sizes, calculation_time_iterative)

plt.xlabel("The size of matrices (N)")
plt.ylabel("Time of calculation, seconds")
plt.legend(["GPU", "Numpy", "Iterative"])

plt.show()
