import os
import cv2
from matplotlib import pyplot as plt

import bilateral_filtering_cpu
import bilateral_filtering_gpu

# ----------------------------------------------------------------
# Test bilateral filtering function on a single image
if __name__ == "__main__":
    # Read the original image
    image_path = os.path.join(
        os.path.dirname(__file__), "images", "test_image_1024.png"
    )
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    height, width = original_image.shape
    print("--------------------------------")
    print(f"Processing image of size {height}x{width}\n")

    # ----------------------------------------------------------------
    # Processing image with CPU
    blurred_image, processing_time = bilateral_filtering_cpu.get_blurred_image(
        original_image
    )
    print(f"Elapsed time (CPU): {processing_time:0.3f} seconds")

    # Show original image
    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original image")

    # Show blurred image
    plt.subplot(2, 2, 2)
    plt.imshow(blurred_image, cmap="gray")
    plt.title(f"Blurred image (CPU)\nElapsed time: {processing_time:0.3f} seconds")

    # ----------------------------------------------------------------
    # Processing image with GPU
    blurred_image, processing_time = bilateral_filtering_gpu.get_blurred_image(
        original_image
    )
    print(f"Elapsed time (GPU): {processing_time:0.3f} seconds")

    # Show original image
    plt.subplot(2, 2, 3)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original image")

    # Show blurred image
    plt.subplot(2, 2, 4)
    plt.imshow(blurred_image, cmap="gray")
    plt.title(f"Blurred image (GPU)\nElapsed time: {processing_time:0.3f} seconds")

    plt.show()
