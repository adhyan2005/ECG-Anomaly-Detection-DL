# 🏥 ECG Anomaly Detection using LSTM Autoencoders
**Research Project | MIT-BIH Arrhythmia Database**

## 📊 Final Results Summary
- **Training Convergence:** The model reached a stable Mean Absolute Error (MAE) of ~0.047 after 20 epochs.
- **Anomaly Detection:** The system successfully identified Ventricular Ectopic Beats (V-beats) by flagging high reconstruction errors compared to normal R-peaks.
- **Performance:** Developed and tested on Apple Silicon (M-series) using TensorFlow's Metal acceleration.

## 🛠️ Project Phases
1. **Signal Processing:** 3rd-order Butterworth filtering to remove 0.5Hz-45Hz noise.
2. **Dataset Architecture:** Segmented Lead II signals into 180-sample windows.
3. **Deep Learning:** Unsupervised LSTM Autoencoder (49,985 parameters).
4. **Validation:** Detection based on Reconstruction Error thresholds.

## 📈 Visualizations
- **Training Curve:** `training_progress.png` shows smooth learning without overfitting.
- **Detection Result:** `final_detection_result.png` compares healthy vs. anomalous reconstruction.
