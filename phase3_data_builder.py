import wfdb
import numpy as np
from scipy.signal import butter, filtfilt

def ecg_filter(data, lowcut=0.5, highcut=45.0, fs=360.0, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

print("--- Phase 3: Building Training Dataset ---")
record = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')
signals = ecg_filter(record.p_signal[:, 0])

indices = annotation.sample
labels = np.array(annotation.symbol)

X_normal = []
window_size = 180 

print("Slicing heartbeats...")
for i in range(len(labels)):
    if labels[i] == 'N': 
        idx = indices[i]
        if idx > window_size and idx < len(signals) - window_size:
            beat = signals[idx - window_size//2 : idx + window_size//2]
            beat = (beat - np.min(beat)) / (np.max(beat) - np.min(beat))
            X_normal.append(beat)

X_normal = np.array(X_normal)
X_normal = np.expand_dims(X_normal, axis=2)

print(f"✅ Success! Created a training set with {X_normal.shape[0]} normal heartbeats.")
np.save('X_normal.npy', X_normal)
print("Data saved as 'X_normal.npy'. Ready for AI Training!")
