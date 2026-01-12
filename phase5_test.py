import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt

# Fix for the 'mae' loading error
custom_objects = {'mae': tf.keras.losses.MeanAbsoluteError()}

# 1. Reuse the filter
def ecg_filter(data, lowcut=0.5, highcut=45.0, fs=360.0, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

print("Loading model and patient data...")
# Loading with custom_objects to bypass the 'mae' error
model = tf.keras.models.load_model('ecg_autoencoder.h5', custom_objects=custom_objects)
record = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')
signals = ecg_filter(record.p_signal[:, 0])

indices = annotation.sample
labels = np.array(annotation.symbol)

# Pick a Ventricular beat (V) and a Normal beat (N)
v_indices = np.where(labels == 'V')[0]
n_indices = np.where(labels == 'N')[0]

v_idx = indices[v_indices[0]]
n_idx = indices[n_indices[15]] # Use the 15th normal beat

def get_beat(idx):
    beat = signals[idx-90:idx+90]
    beat = (beat - np.min(beat)) / (np.max(beat) - np.min(beat))
    return beat.reshape(1, 180, 1)

# 4. Predict
normal_beat = get_beat(n_idx)
anomaly_beat = get_beat(v_idx)

pred_norm = model.predict(normal_beat)
pred_anom = model.predict(anomaly_beat)

# 5. Visualize
plt.figure(figsize=(12, 10))

# Plot Normal
plt.subplot(2, 1, 1)
plt.plot(normal_beat.flatten(), label='Original Normal')
plt.plot(pred_norm.flatten(), '--', label='AI Reconstruction')
err_n = np.mean(np.abs(normal_beat - pred_norm))
plt.title(f"Normal Beat - Error: {err_n:.4f}")
plt.legend()

# Plot Anomaly
plt.subplot(2, 1, 2)
plt.plot(anomaly_beat.flatten(), color='red', label='Original Anomaly (V)')
plt.plot(pred_anom.flatten(), color='black', linestyle='--', label='AI Reconstruction')
err_v = np.mean(np.abs(anomaly_beat - pred_anom))
plt.title(f"Anomaly Beat - Error: {err_v:.4f}")
plt.legend()

plt.tight_layout()
plt.savefig('final_detection_result.png')
print(f"\n--- RESULTS ---")
print(f"Normal Error: {err_n:.4f}")
print(f"Anomaly Error: {err_v:.4f}")
print(f"Detection Ratio: {err_v/err_n:.2f}x higher error for anomalies!")
