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
from skimage import morphology, color,measure

def find_optimal_min_size(image_bin):
    #get regions of small black holes
    label_image = morphology.label(~image_bin)  # Invert image_bin to detect black holes as regions
    regions = measure.regionprops(label_image)
    
    # Get the size of each black hole (region)
    hole_sizes = [region.area for region in regions]
    
    if not hole_sizes:
        return 1  
    
    # Calculate a reasonable min_size based on hole size statistics (e.g., 75th percentile)
    percentile_size = np.percentile(hole_sizes, 70)  
    return int(percentile_size)

# Remove small black noise from the image
def remove_small_black_noise(image_path):
    # Load the binary image (it's already binary with values 0 and 1)
    im = np.array(Image.open(image_path))
    
    # Invert the image to treat black (0) regions as holes
    binary_image = im.astype(bool)
    
    # Find the optimal min_size for the current image
    optimal_min_size = find_optimal_min_size(binary_image)
    
    # Remove small black holes (fill small black regions in white areas)
    cleaned = morphology.remove_small_holes(binary_image, area_threshold=optimal_min_size)
    
    # Convert the binary image back to the original format (0 and 255 for display purposes)
    im = cleaned.astype(np.uint8) * 255
    
    return im

# Deskew the image
def deskew(_img):
    image = _img.copy()
    
    # Check if the image is already grayscale
    if len(image.shape) == 2:  # Grayscale image
        grayscale = image
    else:  # RGB image
        grayscale = rgb2gray(image)
    
    angle = determine_skew(grayscale)
    if angle in range(-10,10):
        rotated = rotate(image, angle, resize=True) * 255
    else: 
        rotated = image
    return rotated.astype(np.uint8)

# Function to apply preprocessing (deskew and median filter) on a single image
def preprocess_image(image_path, output_dir, median_filter_size=5):
    try:
        
        clean_img = remove_small_black_noise(image_path)
        # Apply deskew
        deskewed_image = deskew(clean_img)
        
        
        # Save the preprocessed image
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        Image.fromarray(deskewed_image).save(output_path)
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
    input_directory = '//home/hadi/dataset-dev/plates_png'  # Replace with the actual directory of your images
    output_directory = "preprocessed_imgs"  # Replace with where you want to save the processed images
    
    # Start processing images in parallel with 30 threads
    process_images_in_parallel(input_directory, output_directory, num_threads=30, median_filter_size=5)
