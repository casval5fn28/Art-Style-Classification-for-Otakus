import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

def augment_images_in_folder(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define parameters for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Iterate through each file
    for file in files:
        if file.endswith((".jpg")):
            # Open the image file
            image_path = os.path.join(input_folder, file)
            with Image.open(image_path) as img:
                # Resize the image to 150x150
                img_resized = img.resize((150, 150))

                # Convert the image to an array
                x = np.array(img_resized)
                x = np.expand_dims(x, axis=0)  # Add batch dimension

                # Generate augmented images
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=output_folder, save_prefix=file.split('.')[0], save_format='jpg'):
                    i += 1
                    if i >= 6:  # Generate 6 times more images
                        break
                print(f"Augmented {file} and saved to {output_folder}")

# Specify input and output folders
input_folder = "dataset/cartoon"
output_folder = "dataset/new_cartoon"

# Apply data augmentation
augment_images_in_folder(input_folder, output_folder)
