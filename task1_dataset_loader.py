import os

# Dataset path
DATASET_PATH = "datasets"

# Get all category folders
categories = os.listdir(DATASET_PATH)

print("\nWaste Categories Found:\n")

total_images = 0

for category in categories:
    category_path = os.path.join(DATASET_PATH, category)

    # Count images
    images = os.listdir(category_path)
    image_count = len(images)

    total_images += image_count

    print(f"{category}: {image_count} images")

print("\nTotal Images:", total_images)