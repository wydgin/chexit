EfficientNet .weights.h5 files are not in Git. Copy them here.

Holdout (final train/val split):
- eff_holdout_best.weights.h5 (preferred — deploy export from efficientnet_prog_final.py)
- eff_holdout_phase2_best.weights.h5 (usually same weights as *_best; script loads phase2 first)
- eff_holdout_phase1_best.weights.h5 (fallback only)
- eff_holdout_best_rebuilt.weights.h5 (NOT from training script; ~32MB partial — do not use for /predict)
- eff_holdout_phase1_best.weights.h5

Cross-validation folds:
- fold_0.weights.h5
- fold_1.weights.h5
