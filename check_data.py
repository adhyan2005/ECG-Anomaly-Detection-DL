import wfdb
import matplotlib.pyplot as plt
import numpy as np

# 1. Download/Read record 100 from MIT-BIH
# This will download the files to your folder automatically
record = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')

# 2. Extract the signal and the labels
signal = record.p_signal[:, 0]  # Get Lead II
labels = annotation.symbol      # The 'N', 'V', etc.
indices = annotation.sample     # The location (timestamp) of those beats

print(f"Total heartbeats found in this record: {len(labels)}")

# 3. Find a Normal (N) and an Abnormal (V) beat to compare
norm_idx = indices[labels.index('N')]
abnorm_idx = indices[labels.index('V')] if 'V' in labels else indices[1]

# 4. Plotting
plt.figure(figsize=(12, 5))

# Plot Normal Beat
plt.subplot(1, 2, 1)
plt.plot(signal[norm_idx-100 : norm_idx+100])
plt.title("Normal Beat (N)")
plt.grid(True)

# Plot Anomaly Beat
plt.subplot(1, 2, 2)
plt.plot(signal[abnorm_idx-100 : abnorm_idx+100], color='red')
plt.title("Anomaly Beat (Ventricular)")
plt.grid(True)

plt.tight_layout()
plt.show()