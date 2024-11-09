import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# load images
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img = img.resize((100, 100))
        img = img.convert('L')  # to grayscale
        if img is not None:
            images.append(np.array(img))
            filenames.append(filename)
    return images, filenames

folder = 'dataset_unsupervised_aug'
images, filenames = load_images_from_folder(folder)

# turn images to feature vectors
X = np.array(images).reshape(len(images), -1)
# scaling feature vectors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k-means clustering
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)  # set n_init explicitly
cluster_labels = kmeans.fit_predict(X_scaled)

# create a dataframe for results
result_df = pd.DataFrame({'filename': filenames, 'class': cluster_labels})
# print count of each class
class_counts = result_df['class'].value_counts().sort_index()
print("Class Counts:")
print(class_counts)

result_df.to_csv('result_aug.csv', index=False)

"""
anime	1  2~1317
cartoon	0  1318~1553
comic	1(3)  1554~1937
manga	2  1938~3161

"""
"""
Data augmentation

anime	3(0)  2~1317
cartoon	2  1318~2733
comic	3  2734~5037
manga	1  5038~6261

"""