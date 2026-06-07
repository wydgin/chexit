# Chexit

**Chexit (pronounced CHeckst It) - Chest X-ray Identification for Tuberculosis** is a clinical decision-support web app for TB screening using chest X-rays (CXRs). Upload a PNG, JPG, or DICOM study, get a TB probability score, risk label, per-model contributions, and an explainability heatmap.

This project was developed as a research capstone at the **University of the Philippines Diliman · Electrical and Electronics Engineering Institute (EEEI)**.

- **Web app:** [https://chexit.app](https://chexit.app) (Vercel)

> **Disclaimer:** Chexit is a research prototype for decision support only. It is not a medical device and must not replace professional diagnosis or clinical judgment.

---

## Features

- **TB risk scoring** — ensemble of three CNN architectures (MobileNetV3, EfficientNet-B2, DenseNet-121)
- **Explainable AI** — Score-CAM heatmaps overlaid on the original CXR resolution
- **Lung segmentation** — U-Net masks lungs before classification
- **Batch analysis** — up to 5 images per session in the dashboard
- **DICOM support** — `.dcm` / `.dicom` uploads in addition to standard images
- **Dark / light mode** — MUI theme with shared design tokens

---

## Architecture

```
┌─────────────────────┐         HTTPS (long /predict)        ┌──────────────────────┐
│  React + Vite + MUI │  ─────────────────────────────────►  │  FastAPI (uvicorn)   │
│  chexit.app         │         api.chexit.app               │  DigitalOcean droplet│
└─────────────────────┘                                      └──────────────────────┘
         │                                                            │
         │  Local dev: /api/* proxied to :8000                        │
         └────────────────────────────────────────────────────────────┘
```

| Layer | Stack | Location |
|-------|-------|----------|
| Frontend | React 18, TypeScript, Vite, MUI 7 | `src/` — deployed to Vercel |
| Backend | Python 3.11, FastAPI, TensorFlow | `chexit-backend/` — deployed to `/opt/chexit` on a droplet |
| Models | U-Net + ensemble weights | `assets/` (~1 GB; large `.keras`/`.h5` files are gitignored) |
| Optional | Firebase (Firestore, Storage, Analytics) | `src/firebase.ts` |

**Important:** `/predict` can run for several minutes (Score-CAM on CPU). Production builds call `https://api.chexit.app` directly from the browser. Do **not** route long inference through Vercel `/api` rewrites — edge proxies time out in seconds to ~1 minute.

---

## Repository layout

```
chexit/
├── src/                    # React UI (dashboard, about, archived sign-in)
│   ├── api/chexit.ts       # API client + env-based URL resolution
│   ├── marketing-page/     # TB analyzer dashboard (/dashboard)
│   └── about/              # Project & team page (/about)
├── chexit-backend/
│   ├── app/
│   │   ├── main.py         # FastAPI routes: /health, /predict, /upload
│   │   ├── chexit_inference.py
│   │   ├── model_defs/
│   │   └── explainability/
│   ├── tests/
│   ├── run_dev.sh
│   └── requirements.txt
├── assets/                 # ML weights & training configs (see assets/README_MODEL_LAYOUT.txt)
├── deploy/                 # systemd unit for production API
├── scripts/
│   └── deploy-to-droplet.sh
├── shared-theme/           # MUI theme shared across pages
├── public/                 # Static assets (team photos, acknowledgements)
├── vercel.json             # SPA rewrites (frontend only; avoid /api for predict)
└── .env.example
```

---

## Prerequisites

- **Node.js** 18+ and npm
- **Python 3.11** (see `chexit-backend/.python-version`)
- **~1 GB disk** for `assets/` model weights (U-Net may auto-download from Google Drive on first API start)

---

## Local development

### 1. Install frontend dependencies

```bash
npm install
```

### 2. Set up the backend

From `chexit-backend/`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Model assets live under `assets/`. MobileNet / EfficientNet / DenseNet holdout weights are tracked in git; the U-Net file `segmentation_models/unet_lung_seg_best.keras` is gitignored — copy it locally or let startup download it via `gdown` (see [chexit-backend/README.md](chexit-backend/README.md)).

### 3. Run UI + API together

```bash
npm run dev:stack
```

| Service | URL |
|---------|-----|
| Dashboard | [http://localhost:5173/dashboard](http://localhost:5173/dashboard) |
| About | [http://localhost:5173/about](http://localhost:5173/about) |
| API | [http://127.0.0.1:8000](http://127.0.0.1:8000) |
| OpenAPI | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) |

Vite proxies `/api/*` → `http://127.0.0.1:8000/*` with a 10-minute timeout (matches client-side `PREDICT_TIMEOUT_MS`).

### 4. Analyzer workflow

1. Open `/dashboard`
2. **Browse** a chest X-ray (PNG, JPG, or DICOM; max 10 MB)
3. **Analyze** — calls `POST /api/predict` (local) or `POST https://api.chexit.app/predict` (production)
4. Review diagnosis, risk score, model contributions, and heatmap overlay

### Faster local inference

```bash
CHEXIT_SKIP_SCORECAM=1 npm run dev:api
```

Uses a fast lung-mask heatmap instead of full Score-CAM (much quicker on CPU).

### Production-like preview

```bash
npm run build && npm run preview
```

On `localhost`, `/api` still proxies to port 8000.

---

## Environment variables

Copy `.env.example` → `.env` for local overrides.

### Frontend (Vite)

| Variable | Purpose |
|----------|---------|
| `VITE_CHEXIT_API_URL` | Override API origin (e.g. `https://api.chexit.app`). Leave unset locally to use Vite `/api` proxy. |
| `VITE_USE_RELATIVE_API=1` | Force same-origin `/api` (OK locally; not for long `/predict` on Vercel). |

### Backend

| Variable | Purpose |
|----------|---------|
| `CHEXIT_ASSETS_ROOT` | Path to `assets/` (production: `/opt/chexit/assets`) |
| `CHEXIT_SKIP_GDOWN=1` | Skip U-Net download from Google Drive |
| `CHEXIT_SKIP_SCORECAM=1` | Fast heatmap only |
| `CHEXIT_MAX_CXR_EDGE=2048` | Downscale large CXRs (longest side) |
| `CHEXIT_CORS_ORIGINS` | Comma-separated allowed origins |

Full list: [chexit-backend/README.md](chexit-backend/README.md) and `.env.example`.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | Upload image/DICOM → diagnosis, risk, heatmap (base64 PNG), model contributions |
| `POST` | `/upload` | Persist upload; returns download URL |
| `GET` | `/uploads/latest` | Metadata for most recent upload |
| `GET` | `/docs` | Swagger UI |

**`POST /predict` response fields:** `diagnosis`, `risk_score` (0–100), `confidence_label`, `heatmap` (base64), `model_contributions` (`mobilenet-v3-large`, `efficientnet-b2`, `densenet-121`).

---

## Inference pipeline

1. Optional downscale (`CHEXIT_MAX_CXR_EDGE`)
2. CLAHE on full-resolution grayscale
3. **U-Net** lung segmentation
4. Per-model preprocessing → ensemble classification
5. Per-model **Score-CAM** → weighted fusion into a single heatmap at original image dimensions

Details: `chexit-backend/app/chexit_inference.py`.

---

## Deployment

### Frontend (Vercel)

- Build: `npm run build` → `dist/`
- `vercel.json` serves the SPA; **do not** rely on `/api` rewrites for `/predict`
- Production builds use `https://api.chexit.app` by default (`src/api/chexit.ts`)

### Backend (DigitalOcean droplet)

```bash
# Full deploy (code + restart API)
bash scripts/deploy-to-droplet.sh

# Code only (skip ~1GB assets sync)
bash scripts/deploy-to-droplet.sh --code-only

# Assets only
bash scripts/deploy-to-droplet.sh --assets-only
```

Override target host: `CHEXIT_DROPLET=root@your-host bash scripts/deploy-to-droplet.sh`

The script rsyncs to `/opt/chexit`, installs Python deps, and restarts the `chexit-api` systemd service (`deploy/chexit-api.service`). API is served behind TLS at `https://api.chexit.app`.

---

## Scripts reference

| Command | Description |
|---------|-------------|
| `npm run dev` | Vite dev server only |
| `npm run dev:api` | FastAPI on port 8000 |
| `npm run dev:stack` | UI + API concurrently |
| `npm run build` | Typecheck + production build |
| `npm run preview` | Preview production build |
| `npm run test:backend` | pytest in `chexit-backend/` |

---

## Acknowledgements

See `/about` for team, partners, and institutional acknowledgements.
