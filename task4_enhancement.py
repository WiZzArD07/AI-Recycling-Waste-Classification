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

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image
resized_image = cv2.resize(image_rgb, (256, 256))

# -----------------------------------
# STEP 1: Convert to Grayscale
# -----------------------------------

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

# -----------------------------------
# STEP 2: Histogram Equalization
# -----------------------------------

enhanced_image = cv2.equalizeHist(gray_image)

# -----------------------------------
# Visualization
# -----------------------------------

plt.figure(figsize=(12, 6))

# Original grayscale
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

# Enhanced image
plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title("Enhanced Image")
plt.axis("off")

plt.tight_layout()
plt.show()