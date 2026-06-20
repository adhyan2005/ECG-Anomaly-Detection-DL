"""
Phase 6: Baseline Comparison
--------------------------------
A paper that only reports "our deep model got X% accuracy" with nothing
to compare against is a weak paper. This script gives you a legitimate
baseline: a One-Class SVM trained on simple handcrafted features
(beat amplitude statistics + basic morphology), evaluated on the exact
same DS1/DS2 split and the exact same test beats as the autoencoder.

This lets the paper make a real claim: "the LSTM autoencoder improves
F1 by X points over a classical baseline using the same data split,"
which is a far stronger and more honest statement than an isolated
accuracy number.
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def handcrafted_features(X):
    """X shape: (n_beats, window, 1). Returns simple per-beat statistics."""
    X = X.squeeze(-1)
    feats = np.column_stack([
        X.mean(axis=1),
        X.std(axis=1),
        X.max(axis=1),
        X.min(axis=1),
        np.ptp(X, axis=1),                       # peak-to-peak amplitude
        np.mean(np.abs(np.diff(X, axis=1)), 1),  # mean absolute slope
        np.argmax(X, axis=1) / X.shape[1],       # normalized R-peak position
    ])
    return feats


def main():
    print("Loading datasets (same split used for the autoencoder)...")
    X_train = np.load('X_train_normal.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    print("Extracting handcrafted features...")
    F_train = handcrafted_features(X_train)
    F_test = handcrafted_features(X_test)

    scaler = StandardScaler().fit(F_train)
    F_train_s = scaler.transform(F_train)
    F_test_s = scaler.transform(F_test)

    print("Training One-Class SVM on normal beats only...")
    # nu ~ expected fraction of outliers in training data; small since
    # training set is curated to be normal-only
    clf = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    clf.fit(F_train_s)

    print("Scoring test set...")
    # decision_function: higher = more "normal"-like. Flip sign so higher = more anomalous,
    # matching the convention used for the autoencoder's reconstruction error.
    anomaly_score = -clf.decision_function(F_test_s)
    y_pred = (clf.predict(F_test_s) == -1).astype(int)  # -1 = outlier/anomaly in sklearn's OCSVM

    auc = roc_auc_score(y_test, anomaly_score)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- Baseline: One-Class SVM on handcrafted features ---")
    print(f"AUC:       {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print("\nCompare these numbers directly against phase5b_evaluate.py's output "
          "for the LSTM autoencoder, on the identical test set.")

    np.savez('baseline_results.npz', auc=auc, precision=prec, recall=rec, f1=f1)
    print("✅ Saved to baseline_results.npz")


if __name__ == '__main__':
    main()
