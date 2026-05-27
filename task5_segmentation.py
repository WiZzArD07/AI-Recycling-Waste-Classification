import os
import cv2
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

# Convert to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image
resized_image = cv2.resize(image_rgb, (256, 256))

# Convert to grayscale
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

# -----------------------------------
# STEP 1: Threshold Segmentation
# -----------------------------------

_, threshold_image = cv2.threshold(
    gray_image,
    127,
    255,
    cv2.THRESH_BINARY
)

# -----------------------------------
# STEP 2: Edge Detection
# -----------------------------------

edges = cv2.Canny(gray_image, 100, 200)

# -----------------------------------
# STEP 3: Contour Detection
# -----------------------------------

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Draw contours
contour_image = resized_image.copy()

cv2.drawContours(
    contour_image,
    contours,
    -1,
    (255, 0, 0),
    2
)

# -----------------------------------
# Visualization
# -----------------------------------

plt.figure(figsize=(15, 8))

# Original
plt.subplot(1, 4, 1)
plt.imshow(resized_image)
plt.title("Original")
plt.axis("off")

# Threshold
plt.subplot(1, 4, 2)
plt.imshow(threshold_image, cmap='gray')
plt.title("Threshold")
plt.axis("off")

# Edges
plt.subplot(1, 4, 3)
plt.imshow(edges, cmap='gray')
plt.title("Edges")
plt.axis("off")

# Contours
plt.subplot(1, 4, 4)
plt.imshow(contour_image)
plt.title("Contours")
plt.axis("off")

plt.tight_layout()
plt.show()