import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# load images from folder & turn them into grayscale
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img = img.resize((256, 256))  # Resize the image to a common size
        img = img.convert('L')  # Convert image to grayscale
        if img is not None:
            images.append(np.array(img))
    return images

# load images and their labels
def load_data():
    labels = []
    data = []
    categories = ['comic', 'manga', 'cartoon', 'anime']
    for i, category in enumerate(categories):
        folder = os.path.join('dataset', category)
        images = load_images_from_folder(folder)
        data.extend(images)
        labels.extend([i] * len(images))
    return np.array(data), np.array(labels)

# load dataset
X, y = load_data()
# flatten image data
X_flat = X.reshape(len(X), -1)
# specify different numbers of cross-validation sets
cv_sets = [3, 5, 7, 10]

# train Decision Tree classifier with cross-validation
for cv in cv_sets:
    classifier = DecisionTreeClassifier(random_state=42)
    cv_scores = cross_val_score(classifier, X_flat, y, cv=cv)
    print(f"Mean Cross-Validation Score with {cv} folds:", np.mean(cv_scores))

# split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# train Decision Tree classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# predictions & evaluation
y_pred = classifier.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

"""
Accuracy: 0.6392405063291139

Classification Report:
               precision    recall  f1-score   support

           0       0.29      0.21      0.24        92
           1       0.84      0.80      0.82       242
           2       0.26      0.35      0.30        55
           3       0.66      0.71      0.68       243

    accuracy                           0.64       632
   macro avg       0.51      0.52      0.51       632
weighted avg       0.64      0.64      0.64       632 

"""
"""
less data
Accuracy: 0.5840909090909091

Classification Report:
               precision    recall  f1-score   support

           0       0.41      0.36      0.39        85
           1       0.80      0.69      0.74       131
           2       0.24      0.35      0.29        40
           3       0.63      0.66      0.65       184

    accuracy                           0.58       440
   macro avg       0.52      0.52      0.51       440
weighted avg       0.60      0.58      0.59       440
"""

"""
Data Augmented
Accuracy: 0.48881789137380194

Classification Report:
               precision    recall  f1-score   support

           0       0.56      0.49      0.52       478
           1       0.77      0.67      0.72       241
           2       0.35      0.40      0.37       267
           3       0.34      0.41      0.37       266

    accuracy                           0.49      1252
   macro avg       0.51      0.49      0.50      1252
weighted avg       0.51      0.49      0.50      1252

"""