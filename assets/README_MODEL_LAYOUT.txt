Chexit model asset layout
=========================

Drop your model files under these folders:

- segmentation_models/
  - U-Net segmentation models (.keras/.h5)
  - Current backend default: segmentation_models/unet_lung_seg_best.keras
  - Training script: segmentation_models/unet_segmentation.py

- mobilenet_tb_output/
  - weights/
    - MobileNet fold weights (e.g. fold_0_weights.weights.h5)
    - Holdout split: mobile_holdout_weights.weights.h5 (or mobilenet_holdout_weights.weights.h5)
  - optuna_best_params.json (optional hyperparameter file)

- efficientnet_tb_output/
  - weights/
    - EfficientNet fold weights (e.g. fold_0.weights.h5)
    - Holdout split: eff_holdout_phase2_best.weights.h5, eff_holdout_best.weights.h5

- densenet_tb_output/
  - weights/
    - DenseNet fold weights (e.g. fold_0_phase2_best.weights.h5)
    - Holdout split: dense_holdout_phase2_best.weights.h5, dense_holdout_best.weights.h5

Optional: set CHEXIT_HOLDOUT_WEIGHTS_DIR to a folder containing all holdout .h5 files,
or CHEXIT_DENSENET_WEIGHTS / CHEXIT_MOBILENET_WEIGHTS / CHEXIT_EFFICIENTNET_WEIGHTS for explicit paths.

You can keep each architecture in its own folder; this is the intended setup for an ensemble.
