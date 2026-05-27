import os
import cv2
import matplotlib.pyplot as plt

DATASET_PATH = "datasets"

categories = os.listdir(DATASET_PATH)

plt.figure(figsize=(12, 8))

for i, category in enumerate(categories):
    category_path = os.path.join(DATASET_PATH, category)

    image_name = os.listdir(category_path)[0]

    image_path = os.path.join(category_path, image_name)

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 3, i + 1)
    plt.imshow(image)
    plt.title(category)
    plt.axis("off")

plt.tight_layout()
plt.show()