import os
from PIL import Image

def convert_png_to_jpg(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Iterate through each file
    for file in files:
        if file.endswith(".png"):
            # Open the PNG file
            png_image = Image.open(os.path.join(input_folder, file))

            # Create a new filename with .jpg extension
            new_filename = os.path.splitext(file)[0] + ".jpg"

            # Convert and save as JPEG
            jpg_image_path = os.path.join(output_folder, new_filename)
            png_image.convert("RGB").save(jpg_image_path, "JPEG")

            print(f"Converted {file} to {new_filename}")

# Get the current working directory
current_directory = os.getcwd()

# Specify input and output folders
input_folder = os.path.join(current_directory, "dataset\manga")
output_folder = os.path.join(current_directory, "dataset\converted_manga")

# Convert PNG to JPEG
convert_png_to_jpg(input_folder, output_folder)
