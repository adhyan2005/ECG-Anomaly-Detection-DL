# ECG Anomaly Detection using Deep Learning
### Research Project - MIT-BIH Arrhythmia Database

## 📌 Project Overview
This project focuses on detecting cardiac anomalies (specifically Ventricular Ectopic Beats) using an **LSTM Autoencoder**. By training the model only on healthy ECG signals, it learns to identify deviations in rhythm that signify potential heart issues.

## 🛠️ Technology Stack
- **Language:** Python 3.9
- **Libraries:** TensorFlow/Keras, NumPy, Matplotlib, Scipy, WFDB
- **Hardware:** Developed on macOS (ARM64)

## 📡 Methodology
1. **Data Acquisition:** Fetching raw Lead II signals from the MIT-BIH database.
2. **Signal Pre-processing:** Implementing a 3rd-order Butterworth Bandpass Filter (0.5Hz - 45Hz) to remove baseline wander and power-line interference.
3. **Segmentation:** Slicing continuous data into 180-sample windows (centered on R-peaks).
4. **Deep Learning:** Utilizing an unsupervised LSTM Autoencoder for reconstruction-based anomaly detection.

## 📊 Results
Included is a comparison of raw vs. filtered signals, showing significant noise reduction and baseline correction.
