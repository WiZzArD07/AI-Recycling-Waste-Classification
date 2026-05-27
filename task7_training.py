import os
import cv2
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Dataset path
DATASET_PATH = "datasets"

# Store features and labels
features = []
labels = []

# Get categories
categories = os.listdir(DATASET_PATH)

# -----------------------------------
# Feature Extraction Function
# -----------------------------------

def extract_features(image_path):

    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(image_rgb, (256, 256))

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

    # Final feature vector
    feature_vector = np.hstack([
        color_histogram,
        texture_feature,
        shape_feature
    ])

    return feature_vector

# -----------------------------------
# Load Dataset
# -----------------------------------

print("\nExtracting Features...\n")

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

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

print("\nDataset Shape:")
print(X.shape)

# -----------------------------------
# Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# -----------------------------------
# Train Random Forest
# -----------------------------------

print("\nTraining Model...\n")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------
# Predictions
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# Evaluation
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# -----------------------------------
# Save Model
# -----------------------------------

joblib.dump(
    model,
    "models/waste_classifier.pkl"
)

print("\nModel Saved Successfully!")