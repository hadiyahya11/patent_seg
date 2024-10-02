from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser(description='Convert .tif images to .png')
parser.add_argument('-i',
                    '--input-dir',
                    help='default: dataset-dev/plates',
                    default='dataset-dev/plates')
parser.add_argument('-o',
                    '--output-dir',
                    help='default: dataset-dev/plates_png',
                    default='dataset-dev/plates_png')
parser.add_argument('--prefix',
                    help='only convert files starting with PREFIX (default: "GB.")',
                    default='GB.')
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Convert TIFF files to PNG
for filename in os.listdir(input_dir):
    if filename.endswith('.tif') and filename.startswith(args.prefix):
        # Open the TIFF file
        img = Image.open(os.path.join(input_dir, filename))
        # Convert the filename from .tif to .png
        new_filename = filename.replace('.tif', '.png')
        # Save the image in PNG format in the output directory
        img.save(os.path.join(output_dir, new_filename))
        print(f"Converted {filename} to {new_filename}")
