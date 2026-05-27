import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Load Trained Model
# -----------------------------------

model = joblib.load(
    "models/waste_classifier.pkl"
)

# -----------------------------------
# Detect Test Images
# -----------------------------------

TEST_FOLDER = "test_images"

# Supported image formats
valid_extensions = ('.jpg', '.jpeg', '.png')

# Get all valid image files
image_files = [
    file for file in os.listdir(TEST_FOLDER)
    if file.lower().endswith(valid_extensions)
]

# Check if image exists
if len(image_files) == 0:
    raise FileNotFoundError(
        "No image found in test_images folder"
    )

# Select first image automatically
IMAGE_PATH = os.path.join(
    TEST_FOLDER,
    image_files[0]
)

print(f"\nUsing Image: {IMAGE_PATH}")

# -----------------------------------
# Feature Extraction Function
# -----------------------------------

def extract_features(image_path):

    # Read image
    image = cv2.imread(image_path)

    # Check if image loaded correctly
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

    # -----------------------------------
    # Color Histogram Feature
    # -----------------------------------

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

    # -----------------------------------
    # Texture Feature
    # -----------------------------------

    gray_image = cv2.cvtColor(
        resized_image,
        cv2.COLOR_RGB2GRAY
    )

    texture_feature = cv2.Laplacian(
        gray_image,
        cv2.CV_64F
    ).var()

    # -----------------------------------
    # Shape Feature
    # -----------------------------------

    edges = cv2.Canny(
        gray_image,
        100,
        200
    )

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

    # -----------------------------------
    # Final Feature Vector
    # -----------------------------------

    feature_vector = np.hstack([
        color_histogram,
        texture_feature,
        shape_feature
    ])

    return feature_vector, image_rgb

# -----------------------------------
# Extract Features
# -----------------------------------

features, image_rgb = extract_features(
    IMAGE_PATH
)

# Reshape for ML prediction
features = features.reshape(1, -1)

# -----------------------------------
# Predict Waste Category
# -----------------------------------

prediction = model.predict(features)[0]

# Probability scores
probabilities = model.predict_proba(features)[0]

# Highest confidence score
confidence = np.max(probabilities) * 100

# -----------------------------------
# Output Results
# -----------------------------------

print("\nPrediction:")
print(prediction)

print(f"\nConfidence: {confidence:.2f}%")

# -----------------------------------
# Display Image
# -----------------------------------

plt.figure(figsize=(6, 6))

plt.imshow(image_rgb)

plt.title(
    f"Prediction: {prediction}\nConfidence: {confidence:.2f}%"
)

plt.axis("off")

plt.show()