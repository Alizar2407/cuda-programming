import os
import cv2

from matplotlib import pyplot as plt

import sap_filtering_cpu
import sap_filtering_gpu

# ----------------------------------------------------------------
# Test median filtering function
if __name__ == "__main__":
    # Load the image
    image_path = os.path.join(
        os.path.dirname(__file__), "images", "test_image_1024.png"
    )
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Print image dimensions
    height, width = original_image.shape
    print("--------------------------------")
    print(f"Processing image of size {height}x{width}\n")

    # ----------------------------------------------------------------
    # Processing image with CPU
    filtered_image, processing_time = sap_filtering_cpu.get_filtered_image(
        original_image
    )
    print(f"Elapsed time (CPU): {processing_time:0.3f} seconds")

    # Show original image
    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original image")

    # Show filtered image
    plt.subplot(2, 2, 2)
    plt.imshow(filtered_image, cmap="gray")
    plt.title(f"Filtered image (CPU)\nElapsed time: {processing_time:0.3f} seconds")

    # ----------------------------------------------------------------
    # Processing image with GPU
    filtered_image, processing_time = sap_filtering_gpu.get_filtered_image(
        original_image
    )
    print(f"Elapsed time (GPU): {processing_time:0.3f} seconds")

    # Show original image
    plt.subplot(2, 2, 3)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original image")

    # Show filtered image
    plt.subplot(2, 2, 4)
    plt.imshow(filtered_image, cmap="gray")
    plt.title(f"Filtered image (GPU)\nElapsed time: {processing_time:0.3f} seconds")

    plt.show()
