U-Net segmentation models (.keras / .h5) — not in Git.

Place trained artifacts here:
- unet_lung_seg_best.keras          (required for /predict)
- unet_lung_seg_best_weights.weights.h5
- unet_lung_seg_final.keras

Train with: python unet_segmentation.py  (writes into this folder)

Legacy path assets/models/ is still supported as a fallback.
