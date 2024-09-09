import os
import numpy as np
from PIL import Image
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
from scipy.ndimage import median_filter as scipy_median_filter
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Apply median filter on grayscale image
def median_filter(img, size=5):
    # Open the image as grayscale
    image = np.array(img)
    
    # Apply the median filter using scipy's median_filter function
    filtered_image = scipy_median_filter(image, size=size)
    return filtered_image

# Deskew the image
def deskew(_img):
    image = np.array(Image.open(_img))
    
    # Check if the image is already grayscale
    if len(image.shape) == 2:  # Grayscale image
        grayscale = image
    else:  # RGB image
        grayscale = rgb2gray(image)
    
    angle = determine_skew(grayscale)
    rotated = rotate(image, angle, resize=True) * 255
    return rotated.astype(np.uint8)

# Function to apply preprocessing (deskew and median filter) on a single image
def preprocess_image(image_path, output_dir, median_filter_size=5):
    try:
        # Apply deskew
        deskewed_image = deskew(image_path)
        
        # Apply median filter
        filtered_image = median_filter(deskewed_image, size=median_filter_size)
        
        # Save the preprocessed image
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        Image.fromarray(filtered_image).save(output_path)
        print(f"Processed: {filename}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Function to process all images in parallel using a thread pool
def process_images_in_parallel(image_dir, output_dir, num_threads=30, median_filter_size=5):
    # Get all image paths from the directory
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use ThreadPoolExecutor to parallelize the execution with a given number of threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for image_path in image_paths:
            # Submit each image for processing
            futures.append(executor.submit(preprocess_image, image_path, output_dir, median_filter_size))
        
        # Ensure all threads are completed
        for future in as_completed(futures):
            future.result()  # Wait for each thread to complete and catch exceptions

# Example usage
if __name__ == "__main__":
    input_directory = '//home/hadi/images'  # Replace with the actual directory of your images
    output_directory = "//home/images/preprocessed"  # Replace with where you want to save the processed images
    
    # Start processing images in parallel with 30 threads
    process_images_in_parallel(input_directory, output_directory, num_threads=30, median_filter_size=5)
