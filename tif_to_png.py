from PIL import Image
import os

# Define the directories
input_dir = '/Users/mac/Desktop/patent_seg/dataset-dev/plates'
output_dir = '/Users/mac/Desktop/patent_seg/dataset-dev/plates_png'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Convert TIFF files to PNG
for filename in os.listdir(input_dir):
    if filename.endswith('.tif') and filename.startswith("GB."):
        # Open the TIFF file
        img = Image.open(os.path.join(input_dir, filename))
        # Convert the filename from .tif to .png
        new_filename = filename.replace('.tif', '.png')
        # Save the image in PNG format in the output directory
        img.save(os.path.join(output_dir, new_filename))
        print(f"Converted {filename} to {new_filename}")
