import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# load images from folder & turn them into grayscale
def load_img(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img = img.resize((256, 256))  
        img = img.convert('L')  # to grayscale
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
        images = load_img(folder)
        data.extend(images)
        labels.extend([i] * len(images))
    return np.array(data), np.array(labels)

# load dataset
X, y = load_data()
# flatten image data
X_flat = X.reshape(len(X), -1)
# specify different numbers of cross-validation sets
cv_sets = [3, 5, 7, 10]

# train SVM classifier with cross-validation
for cv in cv_sets:
    classifier = SVC(kernel='linear', random_state=42)
    cv_scores = cross_val_score(classifier, X_flat, y, cv=cv)
    print(f"Mean Cross-Validation Score with {cv} folds:", np.mean(cv_scores))

# split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# train SVM classifier
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

# predictions & evaluation
y_pred = classifier.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

"""
Mean Cross-Validation Score with 3 folds: 0.6117024759234332
Mean Cross-Validation Score with 5 folds: 0.6281645569620253
Mean Cross-Validation Score with 7 folds: 0.6300614451380694
Mean Cross-Validation Score with 10 folds: 0.6360759493670887

Accuracy: 0.6281645569620253
Classification Report:
               precision    recall  f1-score   support

           0       0.47      0.23      0.31        92
           1       0.85      0.64      0.73       242
           2       0.42      0.33      0.37        55
           3       0.56      0.84      0.67       243

    accuracy                           0.63       632
   macro avg       0.57      0.51      0.52       632
weighted avg       0.64      0.63      0.61       632
"""

"""
less data
Accuracy: 0.5386363636363637

Classification Report:
               precision    recall  f1-score   support

           0       0.45      0.39      0.42        85
           1       0.71      0.53      0.61       131
           2       0.22      0.38      0.28        40
           3       0.59      0.65      0.62       184

    accuracy                           0.54       440
   macro avg       0.49      0.49      0.48       440
weighted avg       0.57      0.54      0.55       440

"""

"""
Data Augmented

Accuracy: 0.4257188498402556
Classification Report:
               precision    recall  f1-score   support

           0       0.49      0.46      0.47       478
           1       0.72      0.45      0.56       241
           2       0.31      0.38      0.34       267
           3       0.32      0.39      0.35       266

    accuracy                           0.43      1252
   macro avg       0.46      0.42      0.43      1252
weighted avg       0.46      0.43      0.43      1252
"""