import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img = img.resize((100, 100))  # Resize the image to a common size
        img = img.convert('L')  # Convert image to grayscale
        if img is not None:
            images.append(np.array(img))
            filenames.append(filename)
    return images, filenames

# Load images from the folder
folder = 'dataset_unsupervised'
images, filenames = load_images_from_folder(folder)

# Convert images to feature vectors
X = np.array(images).reshape(len(images), -1)

# Scale the feature vectors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply k-means clustering
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)  # Setting n_init explicitly
cluster_labels = kmeans.fit_predict(X_scaled)

# Create a DataFrame for results
result_df = pd.DataFrame({'filename': filenames, 'class': cluster_labels})

# Print the count of each class
class_counts = result_df['class'].value_counts().sort_index()
print("Class Counts:")
print(class_counts)

# Save the result to a CSV file
result_df.to_csv('result.csv', index=False)

"""
anime	1  2~1317
cartoon	0  1318~1553
comic	1(3)  1554~1937
manga	2  1938~3161

"""