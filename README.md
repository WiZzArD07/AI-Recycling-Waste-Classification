# ♻️ AI-Based Recycling Waste Classification System

An end-to-end AI-powered waste classification system using **Computer Vision**, **Machine Learning**, **Deep Learning**, **FastAPI**, and **Streamlit**.

The system classifies waste images into categories such as:

* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

---

# 🚀 Features

## Traditional Computer Vision Pipeline

* Image preprocessing
* Image enhancement
* Image segmentation
* Feature extraction
* Random Forest classification

## Deep Learning Pipeline

* MobileNetV2 Transfer Learning
* CNN Fine-Tuning
* Data Augmentation
* Real-time image prediction

## Backend

* FastAPI REST API
* Image upload endpoint
* JSON prediction response

## Frontend

* Streamlit web interface
* Image upload support
* Prediction visualization
* Confidence score display

---

# 🧠 Technologies Used

| Technology         | Purpose              |
| ------------------ | -------------------- |
| Python             | Core programming     |
| OpenCV             | Image processing     |
| NumPy              | Numerical operations |
| Scikit-learn       | Machine learning     |
| TensorFlow / Keras | Deep learning        |
| FastAPI            | Backend API          |
| Streamlit          | Frontend UI          |
| Matplotlib         | Visualization        |
| Pillow             | Image handling       |

---

# 📂 Project Structure

```bash
waste_model/
│
├── api/
│   └── main.py
│
├── datasets/
│
├── models/
│   ├── waste_classifier.pkl
│   └── best_cnn_model.keras
│
├── outputs/
│
├── test_images/
│
├── task1_dataset_loader.py
├── task2_visualize_dataset.py
├── task3_preprocessing.py
├── task4_enhancement.py
├── task5_segmentation.py
├── task6_features.py
├── task7_training.py
├── task8_evaluation.py
├── task9_predict.py
│
├── cnn_train.py
├── cnn_finetune.py
├── frontend.py
├── main.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Dataset

Dataset used:

Garbage Classification Dataset from Kaggle

Dataset Categories:

* cardboard
* glass
* metal
* paper
* plastic
* trash

Place dataset inside:

```bash
datasets/
```

---

# 🏋️ CNN Model Training

Train CNN model:

```bash
python cnn_train.py
```

Fine-tune CNN model:

```bash
python cnn_finetune.py
```

---

# 🤖 Run FastAPI Backend

```bash
uvicorn api.main:app --reload
```

Open API Docs:

```bash
http://127.0.0.1:8000/docs
```

---

# 💻 Run Streamlit Frontend

```bash
streamlit run frontend.py
```

---

# 🔍 Prediction Workflow

```text
Upload Image
      ↓
Preprocessing
      ↓
CNN Prediction
      ↓
Waste Classification
      ↓
Confidence Score
```

---

# 📊 Model Performance

## Traditional Machine Learning

* Random Forest Accuracy: ~61%

## CNN Deep Learning

* Training Accuracy: ~97%
* Validation Accuracy: ~76%

---

# 📈 Future Improvements

* Improve CNN generalization
* Real-time webcam detection
* Smart recycling suggestions
* Mobile application
* Cloud deployment
* IoT smart dustbin integration

---

# 🎯 Learning Outcomes

This project demonstrates:

* Computer Vision
* Image Processing
* Machine Learning
* Deep Learning
* Transfer Learning
* REST API Development
* Frontend-Backend Integration
* AI Model Deployment Workflow

---

# 👨‍💻 Author

Developed by: Aryan

---

# 📜 License

This project is for educational and academic purposes.
