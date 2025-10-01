

---

````markdown
# Gunshot Detection and Direction Estimation

This project demonstrates an end-to-end pipeline for **audio event detection** and **direction-of-arrival (DoA) estimation** using a combination of machine learning, deep learning, and signal processing techniques.  

The system is designed as a proof-of-concept for real-world applications such as **surveillance, safety monitoring, and intelligent IoT devices**.

---

## Features
- **Audio Feature Extraction**: MFCCs, chroma, and spectral contrast features.  
- **Event Classification**: Supports SVM, Random Forest, and CNN models.  
- **Direction Estimation**: Implements GCC-PHAT algorithm to estimate the angle of arrival using multiple microphones.  
- **Acoustic Simulation**: Uses `pyroomacoustics` to simulate room environments and test DoA estimation accuracy.  
- **Evaluation Metrics**: Provides precision, recall, F1-score for classification and MAE/RMSE for DoA error.  

---

## Project Structure
- `model.py` – Model creation (SVM, RF, CNN) and persistence (save/load).  
- `train.py` – Training pipeline with dataset loading and feature extraction.  
- `predict.py` – Prediction script for new audio files.  
- `utils.py` – Signal processing utilities (GCC-PHAT, DoA calculation).  
- `doa_estimations.py` – Runs DoA estimation experiments.  
- `simulate_with_pyroom.py` – Room simulation with virtual microphones and sound sources.  
- `main.py` – Example script combining detection and localization.  

---

## Installation
Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
````

---

## Requirements

Create a `requirements.txt` file with the following:

```
numpy
scikit-learn
tensorflow
librosa
pyroomacoustics
matplotlib
joblib
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train a Model

```bash
python train.py
```

Trains the classifier and saves it as `gunshot_model.pkl`.

### 2. Predict on an Audio File

```bash
python predict.py path/to/audio.wav
```

Outputs the predicted class and confidence score.

### 3. Run DoA Estimation with Simulation

```bash
python doa_estimations.py
```

Simulates a room environment and estimates the source direction.

### 4. Combined Example

```bash
python main.py
```

Runs detection and DoA estimation end-to-end.

---

## Applications

* **Public Safety**: Gunshot detection for urban surveillance.
* **Smart Devices**: IoT-enabled sound monitoring systems.
* **Research**: Acoustic event classification and localization.

---

## Future Work

* Incorporating larger and more diverse datasets.
* Improving robustness against background noise.
* Deployment on embedded devices (TensorFlow Lite / ONNX Runtime).

---

## License

This project is intended for **research and educational purposes**.
Commercial or real-world deployment may require further validation and testing.

```

---

```
