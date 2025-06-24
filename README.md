# 🔫 Gunshot Detection + Direction of Arrival (DoA) Prediction

This project focuses on **detecting gunshot sounds from audio files** and predicting the **Direction of Arrival (DoA)**. It uses a **Convolutional Neural Network (CNN)** trained on the UrbanSound8K dataset. The project can be deployed **locally** or via **Docker** for easy, flexible usage.

---

## ✨ Project Overview

### 🔍 What This Project Does
- 🎯 Detects gunshot sounds from audio clips.
- 🔄 Predicts whether the given audio contains a gunshot or not.
- 💻 Can be run both locally and in a Docker container.
- 🌐 Optional web interface using Streamlit.
- 🏗️ Supports model retraining with the UrbanSound8K dataset.

### ✅ Why This Is Useful
- 🚨 Real-time gunshot detection for public safety systems.
- 🧠 Integration into smart surveillance and monitoring applications.
- 📦 Docker support enables easy portability across machines.

---

## 📂 Project Structure

```text
project_folder/
 ├── main.py                 # Entry point for local or Streamlit execution
 ├── requirements.txt        # Python dependencies
 ├── Dockerfile              # Docker setup
 ├── gunshot_cnn.pth         # Trained model file
 ├── simulated_stereo.wav    # Sample audio input
 ├── data/
 │     ├── UrbanSound8K/     # Dataset folder
 │     │     ├── audio/             # Audio clips
 │     │     └── metadata/UrbanSound8K.csv  # Dataset labels
 ├── gunshot_detection/      # Core package
 │     ├── __init__.py
 │     ├── model.py          # CNN architecture
 │     ├── predict.py        # Gunshot prediction logic
 │     ├── utils.py          # Feature extraction methods
 │     └── train.py          # Model training script
```

---

## 🚀 Local Deployment Guide

### 🔧 Step 1: Set Up Python Environment

1. Install **Python 3.9** from [Python Official Website](https://www.python.org/downloads/release/python-390/).
2. Create a virtual environment:
```bash
python -m venv venv
```
3. Activate the virtual environment:
- **Windows:**
```bash
venv\Scripts\activate
```
- **Linux/Mac:**
```bash
source venv/bin/activate
```

---

### 🔧 Step 2: Install Python Packages

Make sure you have `requirements.txt` in your project folder. Run:
```bash
pip install -r requirements.txt
```
✅ This will install all required libraries like **PyTorch**, **Librosa**, and **Pandas**.

---

### 🔧 Step 3: Run the Project

- To run in terminal:
```bash
python main.py
```
- To run in Streamlit (web interface):
```bash
streamlit run main.py
```

You’re now ready to predict gunshots!

---

## 🐳 Docker Deployment Guide

### 🛠️ Step 1: Install Docker

- Download from: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Make sure Docker Desktop is running properly.

### 🛠️ Step 2: Verify Docker Installation

In PowerShell:
```bash
docker --version
```
If Docker version is displayed, installation is successful.

### 🛠️ Step 3: Build Docker Image

In your project folder:
```bash
docker build -t gunshot-detection .
```
This creates a Docker image named **gunshot-detection**.

### 🛠️ Step 4: Run Docker Container

- Terminal version:
```bash
docker run -it gunshot-detection
```
- Streamlit web app version:
```bash
docker run -it -p 8501:8501 gunshot-detection
```
Access the web app at 👉 **http://localhost:8501**

---

## 🧠 Model Training (Optional)

If you want to retrain the model:
```bash
python gunshot_detection/train.py
```

### 📊 Training Details
- **Dataset:** UrbanSound8K (Gunshot class vs. Negative class)
- **Feature Extraction:** MFCC (Mel-Frequency Cepstral Coefficients)
- **Model:** Convolutional Neural Network (CNN)

👉 The trained model will be saved as **gunshot_cnn.pth**.

---

## ⚙️ Core Components Explained

| File | Purpose |
|------|---------|
| `model.py` | CNN model structure used for gunshot detection |
| `utils.py` | MFCC feature extraction from audio files |
| `predict.py` | Loads the model and predicts gunshot presence |
| `train.py` | Full training loop to build the model |
| `main.py` | Main entry script for predictions |

---

## 🎯 Key Features

- 🔊 **Gunshot sound classification** using MFCC and CNN.
- 🐳 **Docker-ready** setup for easy cross-platform deployment.
- 🌐 **Streamlit web interface** for interactive predictions.
- 🏗️ **Custom retraining** supported with UrbanSound8K.

---

## ✅ System Requirements

- Python 3.9 ✅
- Docker (optional but recommended) 🐳
- Required Python packages (listed in `requirements.txt`)

---

## 🙌 Author

**Anjana Satish**  
*Gunshot Detection + DoA Prediction System*  

Feel free to reach out for contributions or collaborations!

---
