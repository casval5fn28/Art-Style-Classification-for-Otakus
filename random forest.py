import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img = img.resize((100, 100))
        img = img.convert('L') 
        if img is not None:
            images.append(np.array(img))
    return images

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

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten the image data
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# Train Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_flat, y_train)

# Predictions
y_pred = classifier.predict(X_test_flat)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

"""
Accuracy: 0.8006329113924051

Classification Report:
               precision    recall  f1-score   support

           0       0.72      0.32      0.44        92
           1       0.88      0.97      0.93       242
           2       1.00      0.29      0.45        55
           3       0.73      0.93      0.82       243

    accuracy                           0.80       632
   macro avg       0.83      0.63      0.66       632
weighted avg       0.81      0.80      0.77       632

"""