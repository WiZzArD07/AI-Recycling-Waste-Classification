import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# AI BASED RECYCLING WASTE CLASSIFICATION SYSTEM
# =====================================================

# =====================================================
# DATASET PATH
# =====================================================

DATASET_PATH = "datasets"

# =====================================================
# LOAD CATEGORIES
# =====================================================

categories = os.listdir(DATASET_PATH)

print("\nWaste Categories:\n")

for category in categories:
    print(category)

# =====================================================
# FEATURE EXTRACTION FUNCTION
# =====================================================

def extract_features(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {image_path}"
        )

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize image
    resized_image = cv2.resize(
        image_rgb,
        (256, 256)
    )

    # =================================================
    # PREPROCESSING
    # =================================================

    denoised_image = cv2.GaussianBlur(
        resized_image,
        (5, 5),
        0
    )

    normalized_image = denoised_image / 255.0

    # Convert back to uint8
    normalized_uint8 = (normalized_image * 255).astype(np.uint8)

    # =================================================
    # ENHANCEMENT
    # =================================================

    gray_image = cv2.cvtColor(
        normalized_uint8,
        cv2.COLOR_RGB2GRAY
    )

    enhanced_image = cv2.equalizeHist(
        gray_image
    )

    # =================================================
    # SEGMENTATION
    # =================================================

    _, threshold_image = cv2.threshold(
        enhanced_image,
        127,
        255,
        cv2.THRESH_BINARY
    )

    edges = cv2.Canny(
        threshold_image,
        100,
        200
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # =================================================
    # FEATURE EXTRACTION
    # =================================================

    # Color histogram
    color_histogram = cv2.calcHist(
        [normalized_uint8],
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
    texture_feature = cv2.Laplacian(
        enhanced_image,
        cv2.CV_64F
    ).var()

    # Shape feature
    shape_feature = 0

    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        shape_feature = cv2.contourArea(
            largest_contour
        )

    # =================================================
    # FINAL FEATURE VECTOR
    # =================================================

    feature_vector = np.hstack([
        color_histogram,
        texture_feature,
        shape_feature
    ])

    return feature_vector, image_rgb

# =====================================================
# LOAD DATASET
# =====================================================

features = []
labels = []

print("\nExtracting Features From Dataset...\n")

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

            feature_vector, _ = extract_features(
                image_path
            )

            features.append(feature_vector)

            labels.append(category)

        except Exception as e:

            print(
                f"Error processing {image_name}: {e}"
            )

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# =====================================================
# DATASET INFORMATION
# =====================================================

print("\nDataset Shape:")
print(X.shape)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# =====================================================
# MODEL TRAINING
# =====================================================

print("\nTraining Random Forest Model...\n")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# =====================================================
# MODEL PREDICTION
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# MODEL EVALUATION
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred
))

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:\n")
print(cm)

# =====================================================
# SAVE MODEL
# =====================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/waste_classifier.pkl"
)

print("\nModel Saved Successfully!")

# =====================================================
# TEST IMAGE PREDICTION
# =====================================================

print("\n==============================")
print("REAL IMAGE PREDICTION")
print("==============================")

TEST_FOLDER = "test_images"

valid_extensions = (
    '.jpg',
    '.jpeg',
    '.png'
)

image_files = [

    file for file in os.listdir(TEST_FOLDER)

    if file.lower().endswith(
        valid_extensions
    )
]

if len(image_files) == 0:

    raise FileNotFoundError(
        "No image found in test_images folder"
    )

IMAGE_PATH = os.path.join(
    TEST_FOLDER,
    image_files[0]
)

print(f"\nUsing Image: {IMAGE_PATH}")

# Extract features from test image
test_features, test_image = extract_features(
    IMAGE_PATH
)

# Reshape for prediction
test_features = test_features.reshape(1, -1)

# Predict
prediction = model.predict(
    test_features
)[0]

# Prediction probabilities
probabilities = model.predict_proba(
    test_features
)[0]

confidence = np.max(probabilities) * 100

# =====================================================
# FINAL OUTPUT
# =====================================================

print("\nPrediction:")
print(prediction)

print(f"\nConfidence: {confidence:.2f}%")

# =====================================================
# DISPLAY RESULT
# =====================================================

plt.figure(figsize=(6, 6))

plt.imshow(test_image)

plt.title(
    f"Prediction: {prediction}\nConfidence: {confidence:.2f}%"
)

plt.axis("off")

plt.show()