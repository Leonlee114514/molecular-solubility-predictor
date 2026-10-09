"""DisSolve - the single source of truth for model / data artifacts.

Every artifact path, filename and load-priority order is declared here once.
Readers (services/prediction.py, model.py, app.py, scripts/, tests/) import from
this module instead of hard-coding "output_v2/..." themselves, so promoting a
new model version is a one-file edit rather than a seven-file hunt.

Paths resolve relative to this file, never to the CWD, so Streamlit, uvicorn,
pytest and the training scripts all point at the same files regardless of the
directory they were started from.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output_v2"

# ── RF solubility model: (model_path, descriptor_names_path), priority order ──
# V6 clean (seed=2026 hold-out split) is production; V5 is the legacy fallback
# for older checkouts.
RF_CANDIDATES: tuple[tuple[Path, Path], ...] = (
    (OUTPUT_DIR / "solubility_model_v6_clean.pkl.gz", OUTPUT_DIR / "descriptor_names_v6_clean.pkl"),
    (OUTPUT_DIR / "solubility_model_v5.pkl.gz", OUTPUT_DIR / "descriptor_names_v5.pkl"),
)

# ── pKa: separate acidic/basic regressors, legacy mixed model as last resort ──
PKA_ACIDIC_MODEL = OUTPUT_DIR / "pka_acidic_model.pkl"
PKA_BASIC_MODEL = OUTPUT_DIR / "pka_basic_model.pkl"
PKA_LEGACY_MODEL = OUTPUT_DIR / "pka_model.pkl"

# ── OOD detector (training-set descriptor ranges + fingerprint references) ──
OOD_DETECTOR = OUTPUT_DIR / "ood_detector.pkl.gz"

# ── GNN: (filename, hidden_dim), priority order ──
# gnn_solubility_model_v5_clean.pt is the promoted clean seed=2026 checkpoint.
# Its byte-identical duplicate gnn_solubility_model_v5.pt was removed on
# 2026-10-09 (same SHA-256, written twice at promotion time), so the clean name
# is now the only entry carrying those weights.
GNN_CANDIDATES: tuple[tuple[str, int], ...] = (
    ("gnn_solubility_model_v5_clean.pt", 256),
    ("gnn_solubility_model_v4.pt", 256),
    ("gnn_solubility_model_v3.pt", 128),
    ("gnn_solubility_model.pt", 128),
)

# The clean seed=2026 candidate. scripts/evaluate_models.py loads this one FIRST
# (comparing it against the shipped weights is that script's whole purpose), so
# it is exported separately from the production load order above.
GNN_CLEAN_FILE = OUTPUT_DIR / "gnn_solubility_model_v5_clean.pt"


def gnn_paths() -> tuple[Path, ...]:
    """Every GNN candidate path in priority order, whether it exists or not."""
    return tuple(OUTPUT_DIR / name for name, _ in GNN_CANDIDATES)


def existing_gnn() -> tuple[Path | None, int]:
    """First GNN checkpoint present on disk, with its hidden_dim.

    Returns (None, 0) when no checkpoint is available.
    """
    for name, hidden_dim in GNN_CANDIDATES:
        candidate = OUTPUT_DIR / name
        if candidate.exists():
            return candidate, hidden_dim
    return None, 0


def gnn_available() -> bool:
    """Cheap GNN availability check (file existence only, no model loading)."""
    return any(path.exists() for path in gnn_paths())
