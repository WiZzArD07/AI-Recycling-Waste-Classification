import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam

# =====================================================
# DATASET SETTINGS
# =====================================================

DATASET_PATH = "datasets"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

# =====================================================
# DATA AUGMENTATION
# =====================================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'
)

# =====================================================
# TRAIN GENERATOR
# =====================================================

train_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='training'
)

# =====================================================
# VALIDATION GENERATOR
# =====================================================

validation_generator = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='validation'
)

# =====================================================
# LOAD PRETRAINED MOBILENETV2
# =====================================================

base_model = MobileNetV2(

    weights='imagenet',

    include_top=False,

    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# =====================================================
# BUILD CNN MODEL
# =====================================================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(
        train_generator.num_classes,
        activation='softmax'
    )
])

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(

    optimizer=Adam(learning_rate=0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

# =====================================================
# MODEL SUMMARY
# =====================================================

print("\nCNN Model Summary:\n")

model.summary()

# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining CNN Model...\n")

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=10
)

# =====================================================
# SAVE MODEL
# =====================================================

model.save("models/cnn_waste_classifier.h5")

print("\nCNN Model Saved Successfully!")

# =====================================================
# VISUALIZE TRAINING
# =====================================================

# Accuracy Graph
plt.figure(figsize=(10, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("CNN Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# Loss Graph
plt.figure(figsize=(10, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("CNN Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.show()