import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dataset path
DATASET_PATH = "datasets"

# Get categories
categories = os.listdir(DATASET_PATH)

# Select category
category = categories[0]

# Select image
image_name = os.listdir(os.path.join(DATASET_PATH, category))[0]

# Image path
image_path = os.path.join(DATASET_PATH, category, image_name)

# Read image
image = cv2.imread(image_path)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image
resized_image = cv2.resize(image_rgb, (256, 256))

# -----------------------------------
# STEP 1: Color Feature Extraction
# -----------------------------------

color_histogram = cv2.calcHist(
    [resized_image],
    [0],
    None,
    [256],
    [0, 256]
)

# Normalize histogram
color_histogram = cv2.normalize(
    color_histogram,
    color_histogram
).flatten()

# -----------------------------------
# STEP 2: Texture Feature Extraction
# -----------------------------------

gray_image = cv2.cvtColor(
    resized_image,
    cv2.COLOR_RGB2GRAY
)

# Calculate texture using Laplacian variance
texture_feature = cv2.Laplacian(
    gray_image,
    cv2.CV_64F
).var()

# -----------------------------------
# STEP 3: Shape Feature Extraction
# -----------------------------------

edges = cv2.Canny(gray_image, 100, 200)

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Largest contour area
shape_feature = 0

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    shape_feature = cv2.contourArea(largest_contour)

# -----------------------------------
# Combine Features
# -----------------------------------

feature_vector = np.hstack([
    color_histogram,
    texture_feature,
    shape_feature
])

# -----------------------------------
# Output
# -----------------------------------

print("\nFeature Vector Shape:")
print(feature_vector.shape)

print("\nTexture Feature:")
print(texture_feature)

print("\nShape Feature:")
print(shape_feature)

# -----------------------------------
# Visualize Histogram
# -----------------------------------

plt.figure(figsize=(10, 5))

plt.plot(color_histogram)
plt.title("Color Histogram Feature")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.show()