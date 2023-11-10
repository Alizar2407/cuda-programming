import os
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

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
# Define the CUDA kernel code
kernel_code = """
// Declaring 2D texture memory
texture<unsigned char, 2, cudaReadModeElementType> tex;

// Median filtering function
__global__ void median_filter(unsigned char* input, unsigned char* output, int width, int height) {
    
    // Get indices of the pixel
    int main_pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    int main_pixel_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (main_pixel_x < width && main_pixel_y < height) {
        int pixels[9];
        int texturePixels[9];
        int pixel_index = 0;

        // Extract surrounding pixels using texture memory
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                texturePixels[pixel_index] = tex2D(tex, main_pixel_x + j, main_pixel_y + i);
                pixels[pixel_index++] = input[width * (main_pixel_y + i) + (main_pixel_x + j)];
            }
        }

        // Sort pixels by their brightness
        for (int i = 0; i < 9 - 1; ++i) {
            for (int j = 0; j < 9 - i - 1; ++j) {
                if (pixels[j] > pixels[j + 1]) {
                    int temp = pixels[j];
                    pixels[j] = pixels[j + 1];
                    pixels[j + 1] = temp;
                }
            }
        }

        // Calculate median (always 4th pixel) and store calculated value in the output array
        output[main_pixel_y * width + main_pixel_x] = pixels[4];
    }
}
"""


# ----------------------------------------------------------------
# Define function that uses CUDA kernel code to apply median filtering to the image
@calculate_time_decorator
def get_filtered_image(original_image):
    # Get image shape
    height, width = original_image.shape

    # Compile the CUDA kernel code
    module = SourceModule(kernel_code)

    # Get the mdeian filtering function
    median_filter_func = module.get_function("median_filter")

    # Allocate device memory for the input image
    original_image_gpu = cuda.mem_alloc(original_image.nbytes)

    # Copy input image to the device
    cuda.memcpy_htod(original_image_gpu, original_image)

    # Load input image data to the texture memory
    texref = module.get_texref("tex")
    cuda.matrix_to_texref(original_image.astype(np.uint8), texref, order="C")

    # Allocate memory for the output image
    output_image_gpu = cuda.mem_alloc(original_image.nbytes)

    # Define block and grid dimensions
    block = (32, 32, 1)
    grid = (width // block[0] + 1, height // block[1] + 1, 1)

    # Call the CUDA kernel function
    median_filter_func(
        original_image_gpu,
        output_image_gpu,
        np.int32(width),
        np.int32(height),
        block=block,
        grid=grid,
    )

    # Copy the result from device back to the host
    output_image = np.empty_like(original_image)
    cuda.memcpy_dtoh(output_image, output_image_gpu)

    return output_image
