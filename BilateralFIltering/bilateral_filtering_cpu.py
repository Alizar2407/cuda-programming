import time
import numpy as np


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
# Functions for bilateral filtering algorithm
def get_f_value(ai):
    return ai


# Coefficient depending on distance to a central pixel
def get_g_value(x, y, sigma):
    exponent = np.exp(-((x**2) - (y**2)) / (2 * sigma**2))
    result = exponent / (2 * np.pi * sigma**2)
    return result


# Function calculating new intensity without normalizing coefficients
def get_r_value(ai, a0, sigma):
    term = (get_f_value(ai) - get_f_value(a0)) ** 2
    term = np.clip(term, -50, 50)
    log_exponent = -term / (2 * sigma**2)
    log_result = log_exponent - 0.5 * np.log(2 * np.pi * sigma**2)
    return np.exp(log_result)


# Calculates the new value of pixel intensity
def get_h_value(original_image, x, y, filter_size, sigma):
    # Get kernel borders
    delta = filter_size // 2
    x_start = x - delta
    x_end = x + delta
    y_start = y - delta
    y_end = y + delta

    # Normalizing constant to prevent intensity increase
    k = 0

    # Calculate weighted sum of all pixels in the kernel
    summ = 0

    # Iterate through every pixel of the original image
    main_pixel_brightness = original_image[y, x]
    for y in range(y_start, y_end + 1):
        for x in range(x_start, x_end + 1):
            try:
                current_pixel_brightness = original_image[y, x]
            except IndexError as e:
                current_pixel_brightness = 0

            f = get_f_value(current_pixel_brightness)
            g = get_g_value(x=abs(x - x_start), y=abs(y - y_start), sigma=sigma)
            r = get_r_value(
                ai=current_pixel_brightness, a0=main_pixel_brightness, sigma=sigma
            )

            k += g * r
            summ += f * g * r

    if k == 0:
        return original_image[y, x]
    else:
        return summ / k


# ----------------------------------------------------------------
# Processes the image pixel-by-pixel and creates a new blurred image
@calculate_time_decorator
def get_blurred_image(original_image, filter_size=7, sigma=10):
    height, width = original_image.shape
    print(f"Processing image of size {width}x{height}")

    blurred_image = np.zeros_like(original_image)

    for y in range(height):
        if y % 10 == 0:
            print(f"Processing row {y+1}/{height}")
        for x in range(width):
            new_brightness = get_h_value(original_image, x, y, filter_size, sigma)
            blurred_image[y, x] = new_brightness

    return blurred_image
