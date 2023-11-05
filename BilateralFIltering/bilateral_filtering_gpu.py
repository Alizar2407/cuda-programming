import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


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
# Define the CUDA kernel code of bilateral filtering
cuda_kernel_code = """
#include <pycuda-complex.hpp>

__device__ const float M_PI = 3.14159265358979323846;

__device__ int get_f_value(int ai){
    return ai;
}

// Coefficient depending on distance to a central pixel
__device__ double get_g_value(double x, double y, double sigma){
    double exponent = exp(-((x*x) + (y*y)) / (2.0 * sigma * sigma));
    double result = exponent / (2.0 * M_PI * sigma * sigma);
    return result;
}

// Function calculating new intensity without normalizing coefficients
__device__ double get_r_value(int ai, double a0, double sigma){
    double term = (get_f_value(ai) - get_f_value(a0)) * (get_f_value(ai) - get_f_value(a0));

    if (term <-50)
        term = -50;

    if (term>50)
        term = 50;
    
    double log_exponent = -term / (2 * sigma * sigma);
    double log_result = log_exponent - 0.5 * log(2 * M_PI * sigma * sigma);
    double result = exp(log_result);

    return result;
}


// Calculates the new value of pixel intensity
__device__  double get_h_value(unsigned char* original_image, int x,  int y, int width, int height, int filter_size, int sigma){
    // Get kernel borders
    int delta = filter_size / 2;
    int x_start = x - delta;
    int x_end = x + delta;
    int y_start = y - delta;
    int y_end = y + delta;

    // Normalizing constant to prevent intensity increase
    double k = 0;

    // Calculate weighted sum of all pixels in the kernel
    double summ = 0;

    // Iterate through every pixel of the original image
    int main_pixel_index = y * width + x;
    int main_pixel_brightness = original_image[main_pixel_index];
    for (int y = y_start; y <= y_end; ++y) {
        for (int x = x_start; x <= x_end; ++x) {
            int current_pixel_brightness;
            if (0 <=x && x < width && 0 <= y && y < height) {
                int index = y * width + x;
                current_pixel_brightness = original_image[index];
            }
            else {
                current_pixel_brightness = 0;
            }

            double f = get_f_value(current_pixel_brightness);
            double g = get_g_value(abs(x - x_start), abs(y - y_start), sigma);
            double r = get_r_value(current_pixel_brightness, main_pixel_brightness, sigma);

            k += g * r;
            summ += f * g * r;
        }
    }

    if (k == 0)
        return original_image[main_pixel_index];
    else
        return summ / k;
}


__global__ void gaussian_filter(unsigned char* original_image, unsigned char* output,
    int width, int height, int filter_size, int sigma) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int output_index = y * width + x;
        double new_brightness = get_h_value(original_image, x, y, width, height, filter_size, sigma);
        output[output_index] = static_cast<unsigned char>(new_brightness);
    }
}
"""


# ----------------------------------------------------------------
# Processes the image using GPU and creates a new blurred image
@calculate_time_decorator
def get_blurred_image(original_image, filter_size=7, sigma=10):
    # Copy the image data to the GPU
    original_image_gpu = cuda.mem_alloc(original_image.nbytes)
    cuda.memcpy_htod(original_image_gpu, original_image)

    # Create an output array on the GPU
    output_gpu = cuda.mem_alloc(original_image.nbytes)

    # Compile the CUDA kernel code
    module = SourceModule(cuda_kernel_code)
    gaussian_filter_kernel = module.get_function("gaussian_filter")

    # Define block and grid sizes
    block_size = (32, 32)
    grid_size = (
        (original_image.shape[1] + block_size[0] - 1) // block_size[0],
        (original_image.shape[0] + block_size[1] - 1) // block_size[1],
    )

    # Launch the CUDA kernel to apply bilateral filtering
    gaussian_filter_kernel(
        original_image_gpu,
        output_gpu,
        np.int32(original_image.shape[1]),
        np.int32(original_image.shape[0]),
        np.int32(filter_size),
        np.int32(sigma),
        block=(32, 32, 1),
        grid=grid_size,
    )

    # Copy blurred image back to the CPU
    blurred_image = np.empty_like(original_image)
    cuda.memcpy_dtoh(blurred_image, output_gpu)

    return blurred_image
