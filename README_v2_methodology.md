# ECG Anomaly Detection — Extended Methodology (v2)

This extends the original single-record proof-of-concept into a study with
a real evaluation protocol, suitable for a paper/preprint.

## What changed from the original repo

| Issue in original code | Fix |
|---|---|
| Trained AND "tested" on Record 100 only | Inter-patient DS1 (train) / DS2 (test) split — standard in the literature (de Chazal et al., 2004) |
| "Evaluation" = comparing 1 normal beat vs. 1 anomaly beat by eye | Full quantitative evaluation: precision, recall, F1, AUC, confusion matrix across the entire held-out test set |
| No baseline | One-Class SVM on handcrafted features, evaluated on the identical split |
| No edge-deployment angle | int8 quantization + feasibility analysis (model size, compression ratio, estimated RAM footprint) |

## Run order

```bash
pip install wfdb numpy scipy tensorflow scikit-learn matplotlib

# 1. Build train/test sets from multiple MIT-BIH records (downloads from PhysioNet)
python phase3b_multi_record_builder.py

# 2. Train the autoencoder on X_train_normal.npy (same architecture as phase4_trainer.py,
#    just point it at the new file instead of X_normal.npy)
python phase4_trainer.py

# 3. Full evaluation on the held-out DS2 test set
python phase5b_evaluate.py

# 4. Classical ML baseline for comparison
python phase6_baseline.py

# 5. Quantization feasibility analysis (no hardware required)
python phase7_quantize.py
```

Note: `phase4_trainer.py` currently loads `X_normal.npy` — change that one
line to `X_train_normal.npy` before running step 2.

## What to report in the paper

- **Methodology**: inter-patient split, beat label scheme (which symbols
  count as normal vs. anomaly — documented in `phase3b_multi_record_builder.py`),
  window size, filtering.
- **Results**: precision/recall/F1/AUC from `phase5b_evaluate.py`, with the
  full threshold sweep table (not just the best F1) so reviewers can see
  the threshold isn't cherry-picked.
- **Baseline comparison**: the One-Class SVM numbers from `phase6_baseline.py`
  next to the autoencoder's numbers, same test set.
- **Edge feasibility**: model size before/after quantization, compression
  ratio, and the estimated RAM figure from `phase7_quantize.py` — explicitly
  labeled as a *feasibility estimate*, with real ESP32 validation listed as
  future work, not claimed as measured.

## Honesty notes for the paper

- Don't claim hardware results you don't have. Frame Phase 7 as
  "pre-deployment feasibility analysis."
- Report all five threshold values (k=1.0–3.0) from Phase 5b, not just
  the best one — this is what separates a defensible result from
  threshold-shopping.
- If results turn out worse than the saturated baselines I found via
  Consensus (97–99% on this dataset), that's fine and expected — the
  paper's contribution is the edge-deployment angle, not beating SOTA
  accuracy on MIT-BIH classification, which is already a solved problem.
