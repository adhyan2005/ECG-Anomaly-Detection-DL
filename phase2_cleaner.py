import wfdb
import matplotlib
# Use 'Agg' backend - it saves files instead of opening windows (fixed Mac crash)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

def ecg_filter(data, lowcut=0.5, highcut=45.0, fs=360.0, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

print("Loading Record 100...")
record = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')
raw_signal = record.p_signal[:, 0]

print("Applying Filter...")
clean_signal = ecg_filter(raw_signal)

indices = annotation.sample
labels = np.array(annotation.symbol)
n_matches = np.where(labels == 'N')[0]
v_matches = np.where(labels == 'V')[0]

if len(n_matches) > 0 and len(v_matches) > 0:
    n_idx = indices[n_matches[5]]
    v_idx = indices[v_matches[0]]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(raw_signal[v_idx-150:v_idx+150], color='gray', alpha=0.5, label='Raw')
    plt.plot(clean_signal[v_idx-150:v_idx+150], color='red', label='Filtered')
    plt.title("Effect of Filter on Anomaly (V) Beat")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(clean_signal[n_idx-150:n_idx+150], color='blue')
    plt.title("Cleaned Normal (N) Beat")
    plt.grid(True)

    plt.tight_layout()
    # Save instead of show to avoid macOS 15 crash
    plt.savefig('phase2_results.png')
    print("\n✅ SUCCESS!")
    print("Graph saved as 'phase2_results.png' in your folder.")
    print("Open it from the VS Code sidebar to see your clean signals.")
else:
    print("Could not find N and V beats.")
