# ECG Anomaly Detection via LSTM Autoencoders

**Principal Researcher:** Adhyan Rajiv Gupta  
**Academic Standing:** Undergraduate Student, B.Tech Electronics and Communication Engineering  
**Institution:** Shri Mata Vaishno Devi University (SMVDU), Katra, J&K  
**Data Source:** MIT-BIH Arrhythmia Database (Record 100)

---

## Technical Summary
* **Training Convergence:** The model achieved a stable Mean Absolute Error (MAE) of ~0.047 after 20 epochs.
* **Anomaly Detection:** Identified Ventricular Ectopic Beats (V-beats) by analyzing reconstruction error deviations against learned healthy R-peaks.
* **Hardware Environment:** Implemented on Apple Silicon (M-series) utilizing TensorFlow's Metal-accelerated backend.

## Implementation Methodology
1. **Signal Conditioning:** Applied a 3rd-order Butterworth bandpass filter (0.5Hz–45Hz) to attenuate baseline wander and power-line interference.
2. **Segmentation:** Lead II signals were windowed into 180-sample segments centered on the R-peak.
3. **Architecture:** Unsupervised LSTM Autoencoder comprising 49,985 trainable parameters.
4. **Logic:** Detection criteria established via reconstruction error thresholds (MAE).

## Research Visualizations
* [Training Loss Plot](training_progress.png) — Analysis of model convergence and validation tracking.
* [Detection Result Analysis](final_detection_result.png) — Comparative reconstruction of healthy vs. pathological heartbeats.

## Engineering Observations
* **Filter Selection:** Utilized a 3rd-order Butterworth filter to balance noise attenuation with phase response preservation; higher-order filters were found to introduce excessive signal distortion.
* **Optimization Stability:** Normalizing signal amplitudes to a [0, 1] range was critical for preventing vanishing gradients during the LSTM training phase.

---
*This repository serves as a technical portfolio for academic and research purposes.*

