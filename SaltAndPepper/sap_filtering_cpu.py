import os
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt


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
# Define function that applies median filtering to the image
@calculate_time_decorator
def get_filtered_image(original_image):
    height, width = original_image.shape

    filtered_image = original_image.copy()

    for main_pixel_y in range(height):
        for main_pixel_x in range(width):
            pixels = []
            for y in range(main_pixel_y - 1, main_pixel_y + 2):
                for x in range(main_pixel_x - 1, main_pixel_x + 2):
                    try:
                        pixels += [original_image[y, x]]
                    except IndexError as e:
                        pixels += [original_image[main_pixel_y, main_pixel_x]]

            for i in range(9 - 1):
                for j in range(9 - i - 1):
                    if pixels[j] > pixels[j + 1]:
                        pixels[j], pixels[j + 1] = pixels[j + 1], pixels[j]

            filtered_image[main_pixel_y, main_pixel_x] = pixels[4]

            # filtered_image[main_pixel_y, main_pixel_x] = np.median(pixels)

    return filtered_image
