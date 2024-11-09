import pandas as pd

# Read the CSV file
#df = pd.read_csv("result_test.csv")
#df = pd.read_csv("result_aug_test.csv")
df = pd.read_csv("tmp.csv")
#df = pd.read_csv("tmp_aug.csv")

# init dictionaries to store correct predictions & total predictions for each class
correct_predictions = {0: 0, 1: 0, 2: 0, 3: 0}
total_predictions = {0: 0, 1: 0, 2: 0, 3: 0}

# iterate through each row
for index, row in df.iterrows():
    filename = int(row['filename'])  # convert filename to integer
    class_label = int(row['class'])  # convert class label to integer
    total_predictions[class_label] += 1
    if filename == class_label:
        correct_predictions[class_label] += 1

# compute accuracy for each class
accuracies = {}
for class_label in range(4):
    accuracy = 0 if total_predictions[class_label] == 0 else correct_predictions[class_label] / total_predictions[class_label]
    accuracies[f"class {class_label}"] = accuracy

# print results
for class_label, accuracy in accuracies.items():
    print(f"{class_label}: {accuracy * 100:.2f}% accuracy")
    

"""
anime	1  2~1317
cartoon	0  1318~1553
comic	3  1554~1937
manga	2  1938~3161

class 0: 8.24% accuracy
class 1: 63.77% accuracy
class 2: 81.94% accuracy
class 3: 18.37% accuracy

"""
    
"""
Data augmentation

anime	3  2~1317
cartoon	2  1318~2733
comic	0  2734~5037
manga	1  5038~6261

class 0: 30.47% accuracy
class 1: 57.41% accuracy
class 2: 24.48% accuracy
class 3: 48.03% accuracy

"""
