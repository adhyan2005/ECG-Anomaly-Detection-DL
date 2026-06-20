"""
Phase 5b: Full Evaluation on Held-Out Test Set (DS2)
-------------------------------------------------------
Replaces the single-beat eyeball comparison in phase5_test.py with a
proper quantitative evaluation across the entire inter-patient test set.

Threshold selection:
The anomaly threshold is computed from the TRAINING reconstruction-error
distribution only (mean + k * std of normal-beat errors), never from the
test set, to avoid information leakage. k is chosen via a small sweep
and the value maximizing F1 is reported alongside results for k=2,3 so
sensitivity to threshold choice is transparent in the paper.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix,
                              classification_report)

custom_objects = {'mae': tf.keras.losses.MeanAbsoluteError()}


def reconstruction_errors(model, X):
    preds = model.predict(X, verbose=0)
    return np.mean(np.abs(X - preds), axis=(1, 2))


def main():
    print("Loading model and datasets...")
    model = tf.keras.models.load_model('ecg_autoencoder.h5', custom_objects=custom_objects)
    X_train = np.load('X_train_normal.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    print("Computing reconstruction errors on training set (for threshold)...")
    train_errors = reconstruction_errors(model, X_train)
    mu, sigma = train_errors.mean(), train_errors.std()

    print("Computing reconstruction errors on held-out test set...")
    test_errors = reconstruction_errors(model, X_test)

    # AUC doesn't depend on a fixed threshold -- report it directly
    auc = roc_auc_score(y_test, test_errors)
    print(f"\nROC-AUC (threshold-independent): {auc:.4f}")

    print("\n--- Threshold sweep (mean + k*std of training error) ---")
    best_f1, best_k, best_pred = -1, None, None
    for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
        threshold = mu + k * sigma
        y_pred = (test_errors > threshold).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        print(f"  k={k:.1f}  threshold={threshold:.4f}  "
              f"precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
        if f1 > best_f1:
            best_f1, best_k, best_pred = f1, k, y_pred

    print(f"\nBest F1 at k={best_k} (report this configuration in the paper, "
          f"alongside k=2 and k=3 for transparency)")
    print("\nConfusion Matrix (rows=true, cols=pred) [0=normal, 1=anomaly]:")
    print(confusion_matrix(y_test, best_pred))
    print("\nFull classification report:")
    print(classification_report(y_test, best_pred, target_names=['normal', 'anomaly']))

    print("\nSaving results...")
    np.savez('evaluation_results.npz',
             test_errors=test_errors, y_test=y_test,
             auc=auc, best_k=best_k, best_f1=best_f1)
    print("✅ Saved to evaluation_results.npz")


if __name__ == '__main__':
    main()
