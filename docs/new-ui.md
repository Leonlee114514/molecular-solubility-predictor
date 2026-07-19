# DisSolve — New React UI

DisSolve predicts aqueous solubility (logS), pKa, and drug-likeness properties
from molecular structure. This document covers the React frontend introduced to
replace the legacy Streamlit UI.

> The Streamlit app (`app.py` + `ui/`) is kept as the **legacy reference UI**.
> It still runs (`streamlit run app.py`) and remains the semantic reference for
> the React port. Do not delete it unless the maintainer decides to.

## Architecture

```
┌──────────────────────────┐   HTTP/JSON    ┌─────────────────────────┐
│ React frontend (Vite)    │ ─────────────► │ FastAPI backend         │
│ frontend/src             │  /api/*        │ backend/routes.py       │
│  · pages/Home            │                │  · /predict             │
│  · sections/*            │ ◄───────────── │  · /analysis            │
│  · sections/results/*    │  JSON + PNG    │  · /gnn-explain         │
│  · components/*          │                │  · /predict/batch(-file)│
└──────────────────────────┘                │  · /mol/2d|3d|info      │
                                            └───────────┬─────────────┘
                                                        │
                                            ┌───────────▼─────────────┐
                                            │ services layer          │
                                            │ services/prediction.py  │
                                            │  (framework-free; also  │
                                            │   used by Streamlit)    │
                                            └───────────┬─────────────┘
                                                        │
                                            ┌───────────▼─────────────┐
                                            │ Models (joblib / GNN)   │
                                            │ RF logS · pKa · GNN ·   │
                                            │ OOD detector · SHAP     │
                                            └─────────────────────────┘
```

- **Frontend** — React 19 + TypeScript + Vite, Tailwind (deep-space theme),
  shadcn/ui, react-i18next (zh/en), recharts, 3Dmol.js (lazy-loaded),
  react-markdown. State lives in `pages/Home.tsx`; results render through
  `sections/results/ResultsTabs.tsx` (5 tabs).
- **Backend** — FastAPI (`backend/routes.py`), thin HTTP layer over the
  framework-free `services/prediction.py`. Batch tasks run in daemon threads
  (`backend/tasks.py`, in-memory registry).
- **i18n** — `core/i18n.py` is the single source of truth
  (`scripts/dump_i18n.py` exports flat dot-key JSON to
  `frontend/src/i18n/{zh,en}.json`; run it after adding keys). Backend prose
  (analysis, SHAP insight) is translated server-side; the frontend refetches on
  language switch.

## Run (development)

```bash
# terminal 1 — backend on :8000
venv/Scripts/python.exe -m uvicorn backend.main:app --port 8000

# terminal 2 — frontend dev server (proxies /api → :8000)
cd frontend
npm run dev            # http://localhost:5173
```

## Build

```bash
cd frontend
npm run build          # tsc -b && vite build → dist/
```

## Tests

```bash
venv/Scripts/python.exe -m pytest tests/ -q
```

## API summary

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Service status + model availability |
| POST | `/api/predict` | Single prediction `{smiles, mode, lang}` → logS / pKa / OOD / SHAP |
| POST | `/api/analysis` | pKa factors, Lipinski, drug-likeness, ADME/Tox, ionization, linkage |
| POST | `/api/explain` | LLM (Kimi) markdown explanation; 503 without API key |
| POST | `/api/gnn-explain` | GNNExplainer bond + feature importance |
| GET | `/api/mol/2d` | 2D PNG (`&bonds=i-j:w,...` highlights bonds) |
| GET | `/api/mol/3d` | molblock for 3Dmol.js |
| GET | `/api/mol/info` | `{formula, mw}` |
| GET | `/api/molecules` | Local molecule library (quick select) |
| GET | `/api/molecules/search` | Name search (local + optional PubChem) |
| POST | `/api/mol/parse-file` | mol/sdf/mol2/pdb/xyz → SMILES |
| POST | `/api/predict/batch` | Start batch from JSON list → `{task_id}` |
| POST | `/api/predict/batch-file` | Start batch from CSV upload (SMILES column auto-detected) |
| GET | `/api/predict/batch/{task_id}` | Poll status/progress/results |

## Feature notes

- **Batch prediction** — CSV upload with client-side header preview; SMILES
  column auto-detected with the same rules as the backend (user can override;
  an override extracts the column client-side and uses the JSON endpoint).
  Progress polled every 1.5 s; results table + client-side CSV download.
- **Prediction history** — last 15 predictions in `localStorage`
  (`dissolve_history_v1`), full `PredictionResult` cached so "reuse" restores
  the result tabs without re-predicting. Quota guard drops SHAP arrays first,
  then oldest entries.
- **3D viewer** — `React.lazy` + Suspense so the ~1.7 MB 3dmol chunk loads
  only when the Preview tab renders.
