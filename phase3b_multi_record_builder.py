"""
Phase 3b: Multi-Record Dataset Builder (Inter-Patient Split)
--------------------------------------------------------------
Fixes the single-record limitation of phase3_data_builder.py.

Why this matters:
Training and testing on the SAME record (e.g. only Record 100) lets the
model memorize patient-specific beat morphology rather than learning
general ECG patterns. The standard fix in the literature is the
"inter-patient" DS1/DS2 split introduced by de Chazal, O'Dwyer & Reilly
(2004), used by most published MIT-BIH arrhythmia papers:

    DS1 (train) = 101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
                  122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                  223, 230
    DS2 (test)  = 100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
                  210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
                  233, 234

The autoencoder is trained ONLY on normal beats from DS1.
DS2 is held out entirely for evaluation and contains both normal and
abnormal beats, never seen during training.

Beat label simplification (documented for transparency in the paper's
Methodology section):
    NORMAL   = {'N', 'L', 'R', 'e', 'j'}
    ANOMALY  = {'A', 'a', 'J', 'S', 'V', 'r', 'F', 'E', '/', 'f', 'Q', 'n'}
    Anything else (e.g. '+', '~', rhythm/quality annotations) is skipped
    because it is not a beat label.
"""

import wfdb
import numpy as np
from scipy.signal import butter, filtfilt

DS1_TRAIN_RECORDS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
                      122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                      223, 230]

DS2_TEST_RECORDS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
                     210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
                     233, 234]

NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}
ANOMALY_SYMBOLS = {'A', 'a', 'J', 'S', 'V', 'r', 'F', 'E', '/', 'f', 'Q', 'n'}

WINDOW_SIZE = 180  # samples, centered on R-peak (matches original phase3 script)
FS = 360.0


def ecg_filter(data, lowcut=0.5, highcut=45.0, fs=FS, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def extract_beats(record_id, keep_symbols, half_window=WINDOW_SIZE // 2):
    """Returns (beats, binary_labels) for one record.
    binary_labels: 0 = normal, 1 = anomaly. Beats not in keep_symbols are skipped."""
    record = wfdb.rdrecord(str(record_id), pn_dir='mitdb')
    annotation = wfdb.rdann(str(record_id), 'atr', pn_dir='mitdb')

    signals = ecg_filter(record.p_signal[:, 0])
    indices = annotation.sample
    labels = np.array(annotation.symbol)

    beats, beat_labels = [], []
    for i in range(len(labels)):
        sym = labels[i]
        if sym not in keep_symbols:
            continue
        idx = indices[i]
        if idx <= half_window or idx >= len(signals) - half_window:
            continue
        beat = signals[idx - half_window: idx + half_window]
        denom = np.max(beat) - np.min(beat)
        if denom == 0:
            continue
        beat = (beat - np.min(beat)) / denom
        beats.append(beat)
        beat_labels.append(0 if sym in NORMAL_SYMBOLS else 1)

    return np.array(beats), np.array(beat_labels)


def build_training_set():
    print("--- Phase 3b: Building TRAIN set (DS1, normal beats only) ---")
    all_beats = []
    for rec in DS1_TRAIN_RECORDS:
        print(f"  Processing record {rec}...")
        try:
            beats, lbls = extract_beats(rec, keep_symbols=NORMAL_SYMBOLS)
            all_beats.append(beats)
            print(f"    -> {len(beats)} normal beats")
        except Exception as e:
            print(f"    !! Skipped record {rec}: {e}")

    X_train = np.concatenate(all_beats, axis=0)
    X_train = np.expand_dims(X_train, axis=2)
    np.save('X_train_normal.npy', X_train)
    print(f"✅ Train set saved: {X_train.shape[0]} normal beats -> X_train_normal.npy")


def build_test_set():
    print("\n--- Phase 3b: Building TEST set (DS2, normal + anomaly, unseen patients) ---")
    keep = NORMAL_SYMBOLS | ANOMALY_SYMBOLS
    all_beats, all_labels = [], []
    for rec in DS2_TEST_RECORDS:
        print(f"  Processing record {rec}...")
        try:
            beats, lbls = extract_beats(rec, keep_symbols=keep)
            all_beats.append(beats)
            all_labels.append(lbls)
            print(f"    -> {len(beats)} beats ({(lbls == 1).sum()} anomalous)")
        except Exception as e:
            print(f"    !! Skipped record {rec}: {e}")

    X_test = np.concatenate(all_beats, axis=0)
    y_test = np.concatenate(all_labels, axis=0)
    X_test = np.expand_dims(X_test, axis=2)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    print(f"✅ Test set saved: {X_test.shape[0]} beats "
          f"({(y_test == 1).sum()} anomalous, {(y_test == 0).sum()} normal) "
          f"-> X_test.npy / y_test.npy")


if __name__ == '__main__':
    build_training_set()
    build_test_set()
    print("\nDone. Next: run phase4_trainer.py on X_train_normal.npy, "
          "then phase5b_evaluate.py for full metrics.")
