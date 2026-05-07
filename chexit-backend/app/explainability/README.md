# Explainability integration workspace

Use this folder to paste your model-specific explainability code before we adapt it
to the backend API pipeline.

## Paste targets

- `efficientnet_scorecam.py`
  - Paste your current EfficientNet Score-CAM implementation here.
  - Keep all helper functions/classes even if they are currently CLI-oriented.

- `explainability_config.py`
  - Put default constants (target layer name, batch size, max channels, alpha, etc.).

## Notes for pasting

- Do not worry about style/cleanup yet.
- Keep original function names if possible; we will add adapters afterward.
- If your code has CLI-only paths/args, keep them for now. I will split runtime
  API functions from CLI utilities after review.

## Next step after you paste

I will:
1. map your EfficientNet Score-CAM into the existing FastAPI inference flow,
2. make path handling compatible with `CHEXIT_ASSETS_ROOT`,
3. add a stable callable API surface for optional fused explainability later.

