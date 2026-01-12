import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt

custom_objects = {'mae': tf.keras.losses.MeanAbsoluteError()}

def ecg_filter(data, lowcut=0.5, highcut=45.0, fs=360.0, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

print("Loading model and data for final verification...")
model = tf.keras.models.load_model('ecg_autoencoder.h5', custom_objects=custom_objects)
record = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')
signals = ecg_filter(record.p_signal[:, 0])

indices = annotation.sample
labels = np.array(annotation.symbol)

v_idx = indices[np.where(labels == 'V')[0][0]]
n_idx = indices[np.where(labels == 'N')[0][15]]

def get_beat(idx):
    beat = signals[idx-90:idx+90]
    # Robust normalization to prevent 'nan'
    denom = (np.max(beat) - np.min(beat))
    if denom == 0: denom = 1e-8
    beat = (beat - np.min(beat)) / denom
    return beat.reshape(1, 180, 1)

normal_beat = get_beat(n_idx)
anomaly_beat = get_beat(v_idx)

pred_norm = model.predict(normal_beat)
pred_anom = model.predict(anomaly_beat)

plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(normal_beat.flatten(), label='Original')
plt.plot(pred_norm.flatten(), '--', label='Reconstructed')
err_n = np.mean(np.abs(normal_beat - pred_norm))
plt.title(f"Normal Beat (Error: {err_n:.4f})")
plt.legend()

plt.subplot(2,1,2)
plt.plot(anomaly_beat.flatten(), 'r', label='Original')
plt.plot(pred_anom.flatten(), 'k--', label='Reconstructed')
err_v = np.mean(np.abs(anomaly_beat - pred_anom))
plt.title(f"Anomaly Beat (Error: {err_v:.4f})")
plt.legend()

plt.tight_layout()
plt.savefig('final_detection_result.png')
print(f"\nDetection Ratio: {err_v/err_n:.2f}x")
