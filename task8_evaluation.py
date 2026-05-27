import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

from sklearn.model_selection import train_test_split

# -----------------------------------
# Load Dataset
# -----------------------------------

DATASET_PATH = "datasets"

features = []
labels = []

categories = os.listdir(DATASET_PATH)

# -----------------------------------
# Feature Extraction Function
# -----------------------------------

def extract_features(image_path):

    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    resized_image = cv2.resize(
        image_rgb,
        (256, 256)
    )

    # Color histogram
    color_histogram = cv2.calcHist(
        [resized_image],
        [0],
        None,
        [256],
        [0, 256]
    )

    color_histogram = cv2.normalize(
        color_histogram,
        color_histogram
    ).flatten()

    # Texture feature
    gray_image = cv2.cvtColor(
        resized_image,
        cv2.COLOR_RGB2GRAY
    )

    texture_feature = cv2.Laplacian(
        gray_image,
        cv2.CV_64F
    ).var()

    # Shape feature
    edges = cv2.Canny(gray_image, 100, 200)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    shape_feature = 0

    if contours:
        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        shape_feature = cv2.contourArea(
            largest_contour
        )

    feature_vector = np.hstack([
        color_histogram,
        texture_feature,
        shape_feature
    ])

    return feature_vector

# -----------------------------------
# Load Features
# -----------------------------------

print("\nPreparing Evaluation Dataset...\n")

for category in categories:

    category_path = os.path.join(
        DATASET_PATH,
        category
    )

    for image_name in os.listdir(category_path):

        image_path = os.path.join(
            category_path,
            image_name
        )

        try:
            feature_vector = extract_features(
                image_path
            )

            features.append(feature_vector)

            labels.append(category)

        except:
            pass

X = np.array(features)
y = np.array(labels)

# -----------------------------------
# Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# Load Trained Model
# -----------------------------------

model = joblib.load(
    "models/waste_classifier.pkl"
)

# -----------------------------------
# Predictions
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# Accuracy
# -----------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# -----------------------------------
# Classification Report
# -----------------------------------

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred
))

# -----------------------------------
# Confusion Matrix
# -----------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

# -----------------------------------
# Visualization
# -----------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=categories,
    yticklabels=categories
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.show()

# -----------------------------------
# Class Distribution
# -----------------------------------

class_counts = []

for category in categories:

    category_path = os.path.join(
        DATASET_PATH,
        category
    )

    class_counts.append(
        len(os.listdir(category_path))
    )

plt.figure(figsize=(10, 5))

plt.bar(categories, class_counts)

plt.title("Dataset Class Distribution")
plt.xlabel("Waste Categories")
plt.ylabel("Number of Images")

plt.show()