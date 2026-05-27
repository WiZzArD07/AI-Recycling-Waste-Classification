import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dataset path
DATASET_PATH = "datasets"

# Get categories
categories = os.listdir(DATASET_PATH)

# Select first category
category = categories[0]

# Select first image
image_name = os.listdir(os.path.join(DATASET_PATH, category))[0]

# Full image path
image_path = os.path.join(DATASET_PATH, category, image_name)

# Read image
image = cv2.imread(image_path)

# Convert BGR to RGB
original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -----------------------------
# STEP 1: Resize Image
# -----------------------------

resized_image = cv2.resize(original_image, (256, 256))

# -----------------------------
# STEP 2: Noise Removal
# -----------------------------

denoised_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

# -----------------------------
# STEP 3: Normalize Image
# -----------------------------

normalized_image = denoised_image / 255.0

# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(12, 8))

# Original
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")

# Denoised
plt.subplot(1, 3, 2)
plt.imshow(denoised_image)
plt.title("Preprocessed")
plt.axis("off")

# Normalized
plt.subplot(1, 3, 3)
plt.imshow(normalized_image)
plt.title("Normalized")
plt.axis("off")

plt.tight_layout()
plt.show()