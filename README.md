# ECG Anomaly Detection via LSTM Autoencoders
**Research Project | MIT-BIH Arrhythmia Database**

## Technical Summary
* **Training Convergence:** The model reached a stable Mean Absolute Error (MAE) of ~0.047 after 20 epochs.
* **Anomaly Detection:** Successfully identified Ventricular Ectopic Beats (V-beats) by flagging high reconstruction errors compared to normal R-peaks.
* **System Environment:** Developed and tested on Apple Silicon (M-series) using TensorFlow's Metal acceleration.

## Implementation Phases
1. **Signal Processing:** 3rd-order Butterworth filtering to remove 0.5Hz-45Hz noise.
2. **Dataset Architecture:** Segmented Lead II signals into 180-sample windows.
3. **Deep Learning:** Unsupervised LSTM Autoencoder (49,985 parameters).
4. **Validation:** Detection based on Reconstruction Error thresholds.

## Visualizations
* **Training Curve:** `training_progress.png` shows smooth learning without overfitting.
* **Detection Result:** `final_detection_result.png` compares healthy vs. anomalous reconstruction.

## Engineering Observations
* Selected a 3rd-order Butterworth filter over higher orders to minimize phase distortion while maintaining sufficient noise attenuation.
* Observed that model stability improved significantly when normalizing signal amplitudes to a [0, 1] range prior to windowing.
