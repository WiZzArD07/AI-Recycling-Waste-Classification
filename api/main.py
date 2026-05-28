import io
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# =====================================================
# LOAD CNN MODEL
# =====================================================

model = tf.keras.models.load_model(
    "models/best_cnn_model.keras"
)

# =====================================================
# CLASS LABELS
# =====================================================

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# =====================================================
# CREATE FASTAPI APP
# =====================================================

app = FastAPI()

# =====================================================
# ENABLE CORS
# =====================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

# =====================================================
# IMAGE PREPROCESSING FUNCTION
# =====================================================

def preprocess_image(image):

    # Resize image
    image = image.resize((224, 224))

    # Convert to numpy array
    image = np.array(image)

    # Normalize pixels
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

# =====================================================
# HOME ROUTE
# =====================================================

@app.get("/")

def home():

    return {
        "message": "AI Waste Classification API Running"
    }

# =====================================================
# PREDICTION ROUTE
# =====================================================

@app.post("/predict")

async def predict_image(

    file: UploadFile = File(...)
):

    # Read uploaded image
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    # Preprocess image
    processed_image = preprocess_image(
        image
    )

    # Prediction
    predictions = model.predict(
        processed_image
    )

    # Get highest probability
    predicted_index = np.argmax(
        predictions[0]
    )

    confidence = float(
        np.max(predictions[0]) * 100
    )

    predicted_class = class_names[
        predicted_index
    ]

    # Return JSON response
    return {

        "prediction": predicted_class,

        "confidence": round(
            confidence,
            2
        )
    }