

---

# Gunshot Detection and Direction Estimation

This project implements a system for detecting gunshot sounds and estimating their direction using machine learning and signal processing.

## Features

* Extracts audio features (MFCC, chroma, spectral contrast)
* Trains models (SVM, Random Forest, CNN) for gunshot classification
* Estimates Direction of Arrival (DoA) using the GCC-PHAT algorithm
* Includes room simulation with `pyroomacoustics` for testing

## Project Files

* **model.py** – Model creation and saving/loading
* **train.py** – Train classifier on dataset
* **predict.py** – Run predictions on new audio
* **utils.py** – GCC-PHAT and DoA utilities
* **doa_estimations.py** – Estimate sound direction
* **simulate_with_pyroom.py** – Room simulation for DoA
* **main.py** – Example pipeline combining detection and direction

## Installation

Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

**Train a model**

```bash
python train.py
```

**Predict on audio**

```bash
python predict.py path/to/file.wav
```

**Estimate DoA**

```bash
python doa_estimations.py
```

## Requirements

```
numpy
scikit-learn
tensorflow
librosa
pyroomacoustics
matplotlib
joblib
```

## Applications

* Public safety and surveillance
* IoT sound monitoring devices
* Research on acoustic event detection

---


