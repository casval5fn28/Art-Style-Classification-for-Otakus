import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Iterate through each file
    for file in files:
        if file.endswith((".jpg")):
            # Open the image file
            image_path = os.path.join(input_folder, file)
            with Image.open(image_path) as img:
                # Resize the image to 320x320
                img_resized = img.resize((320, 320), Image.ANTIALIAS)
                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, file)
                img_resized.save(output_path)
                print(f"Resized {file} to 320x320 and saved as {output_path}")

# Get the directory where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify input and output folders (assuming the folder is named "images")
input_folder = os.path.join(current_directory, "manga")
output_folder = os.path.join(current_directory, "converted_manga")

# Resize images
resize_images_in_folder(input_folder, output_folder)