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
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

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

    rotation_range=15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range=0.1,

    horizontal_flip=True
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
# LOAD PRETRAINED MODEL
# =====================================================

base_model = MobileNetV2(

    weights='imagenet',

    include_top=False,

    input_shape=(224, 224, 3)
)

# =====================================================
# FINE-TUNING
# =====================================================

base_model.trainable = True

# Freeze lower layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# =====================================================
# BUILD MODEL
# =====================================================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),

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

    optimizer=Adam(learning_rate=0.0001),

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

# =====================================================
# CALLBACKS
# =====================================================

early_stopping = EarlyStopping(

    monitor='val_accuracy',

    patience=5,

    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=2,

    verbose=1
)

checkpoint = ModelCheckpoint(

    "models/best_cnn_model.keras",

    monitor='val_accuracy',

    save_best_only=True,

    verbose=1
)

# =====================================================
# MODEL SUMMARY
# =====================================================

print("\nFine-Tuned CNN Model Summary:\n")

model.summary()

# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining Fine-Tuned CNN Model...\n")

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=20,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)

# =====================================================
# FINAL ACCURACY
# =====================================================

final_train_accuracy = history.history[
    'accuracy'
][-1] * 100

final_val_accuracy = history.history[
    'val_accuracy'
][-1] * 100

print("\nFinal Training Accuracy:")
print(f"{final_train_accuracy:.2f}%")

print("\nFinal Validation Accuracy:")
print(f"{final_val_accuracy:.2f}%")

# =====================================================
# VISUALIZE ACCURACY
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Fine-Tuned CNN Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# =====================================================
# VISUALIZE LOSS
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("Fine-Tuned CNN Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.show()