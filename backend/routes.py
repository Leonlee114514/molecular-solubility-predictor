"""DisSolve FastAPI routes — all /api endpoints.

Design notes:
  * No Streamlit caching: raw functions (features.compute_features,
    core.analysis.analyze_*, services.prediction.*) are called directly, with
    cachetools.TTLCache where the Streamlit layer used @st.cache_data.
  * Endpoints returning translated PROSE accept `lang` and run inside
    core.i18n.language_context so t() resolves per-request.
  * Heavy/blocking endpoints are plain `def` (FastAPI runs them in a
    threadpool); only the file-upload endpoint is async (await file.read()).
"""

import io
import logging
import threading

from cachetools import TTLCache
from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel

from backend import prediction_result_to_dict, to_jsonable
from backend.tasks import registry
from core.i18n import language_context

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Backend-local caches (TTLs mirror the Streamlit layer) ──
_cache_lock = threading.Lock()
_features_cache: TTLCache = TTLCache(maxsize=512, ttl=3600)   # smiles -> (features, fp)
_analysis_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)   # (smiles, pka, lang) -> dict
_kimi_cache: TTLCache = TTLCache(maxsize=128, ttl=86400)      # request key -> markdown
_mol2d_cache: TTLCache = TTLCache(maxsize=256, ttl=600)       # smiles -> PNG bytes (short TTL)


def _cached_features(smiles):
    """compute_features with a 1h TTL cache (mirrors cached_compute_features)."""
    from features import compute_features

    with _cache_lock:
        hit = _features_cache.get(smiles)
    if hit is not None:
        return hit
    result = compute_features(smiles)
    if result is not None:
        with _cache_lock:
            _features_cache[smiles] = result
    return result


# ── Request models ──

class PredictRequest(BaseModel):
    smiles: str
    mode: str = "auto"
    lang: str = "zh"


class AnalysisRequest(BaseModel):
    smiles: str
    pka: float | None = None
    logs: float | None = None  # predicted logS, for the solubility × pKa linkage prose
    lang: str = "zh"


class ExplainRequest(BaseModel):
    smiles: str
    prediction: float
    features: dict[str, float]
    shap_features: list[str] | None = None
    shap_values: list[float] | None = None
    pka_value: float | None = None
    pka_type: str | None = None
    lang: str = "zh"


class GnnExplainRequest(BaseModel):
    smiles: str


class BatchRequest(BaseModel):
    smiles_list: list[str]
    mode: str = "auto"


# ── Health ──

@router.get("/health")
def health():
    """Service status + model availability (file existence) and load state."""
    from services import prediction as svc

    def _flag(path_attr, loaded_attr):
        path = getattr(svc, path_attr, None)
        available = bool(path.exists()) if path is not None else False
        return {"available": available, "loaded": getattr(svc, loaded_attr, None) is not None}

    return {
        "status": "ok",
        "models": {
            "rf": _flag("_SOLUBILITY_MODEL_PATH", "_solubility_model"),
            "pka": _flag("_PKA_MODEL_PATH", "_pka_model"),
            "gnn": {"available": svc.gnn_files_exist(),
                    "loaded": getattr(svc, "_gnn_model", None) is not None},
            "ood": _flag("_OOD_DETECTOR_PATH", "_ood_detector"),
        },
    }


# ── Prediction ──

@router.post("/predict")
def predict(req: PredictRequest):
    """Single-molecule prediction via the framework-free service layer."""
    from services.prediction import run_prediction

    try:
        with language_context(req.lang):
            result = run_prediction(req.smiles, mode=req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Prediction failed for SMILES %r", req.smiles)
        raise HTTPException(status_code=500, detail="Prediction failed")
    return prediction_result_to_dict(result)


# ── Analysis (pKa factors, Lipinski, drug-likeness, ADME/Tox) ──

def _pka_type_of(pka: float) -> str:
    """Same thresholds as model.get_pka_type."""
    if pka < 6:
        return "acid"
    if pka > 8:
        return "base"
    return "amphoteric"


# Physiological environments for the ionization profile (mirrors ui/results.py).
_IONIZATION_ENVS = [
    ("stomach", 1.5),
    ("duodenum", 4.5),
    ("intestine", 6.8),
    ("blood", 7.4),
]


def _ionization_profile(pka: float, pka_type: str) -> list[dict]:
    """Henderson-Hasselbalch unionized fractions, ported from ui/results.py.

    acid:      f = 1 / (1 + 10^(pH - pKa))
    base/amphoteric: f = 1 / (1 + 10^(pKa - pH))
    """
    out = []
    for env, ph in _IONIZATION_ENVS:
        if pka_type == "acid":
            frac = 1 / (1 + 10 ** (ph - pka))
        else:
            frac = 1 / (1 + 10 ** (pka - ph))
        out.append({"env": env, "ph": ph, "pct": frac * 100})
    return out


def _pharma_analysis_key(pka: float, pka_type: str) -> str:
    """Structured key for the pharmacological-analysis box (frontend translates)."""
    if pka_type == "acid":
        return "strong_acid" if pka < 4 else "mid_acid"
    if pka_type == "base":
        return "strong_base" if pka > 9 else "weak_base"
    return "amphoteric"


def _linkage_prose(logs: float, pka: float, pka_type: str) -> str:
    """Solubility × pKa linkage sentences, ported from ui/results.py (translated)."""
    from core.i18n import t

    parts = []
    if logs > 0:
        parts.append(t("result.pharma.linkage.soluble"))
    elif logs > -2:
        parts.append(t("result.pharma.linkage.moderate"))
    else:
        parts.append(t("result.pharma.linkage.poor"))
    if pka_type == "acid":
        parts.append(
            t("result.pharma.linkage.pka_weak_acid", val=pka)
            if pka < 4
            else t("result.pharma.linkage.pka_mid_acid", val=pka)
        )
    elif pka_type == "base":
        parts.append(
            t("result.pharma.linkage.pka_strong_base", val=pka)
            if pka > 9
            else t("result.pharma.linkage.pka_weak_base", val=pka)
        )
    else:
        parts.append(t("result.pharma.linkage.pka_neutral", val=pka))
    if logs > 0 and pka_type == "acid" and pka < 4:
        parts.append(t("result.pharma.linkage.combo_good"))
    elif logs < -2 and pka_type == "base" and pka > 9:
        parts.append(t("result.pharma.linkage.combo_challenging"))
    elif logs > 0 and pka_type == "base" and pka > 9:
        parts.append(t("result.pharma.linkage.combo_acceptable"))
    return " | ".join(parts)


@router.post("/analysis")
def analysis(req: AnalysisRequest):
    """Analysis bundle with backend-side translated prose."""
    cache_key = (req.smiles.strip(), req.pka, req.logs, req.lang)
    with _cache_lock:
        hit = _analysis_cache.get(cache_key)
    if hit is not None:
        return hit

    from core.analysis import (
        analyze_admet,
        analyze_druglikeness,
        analyze_lipinski,
        analyze_pka_chemistry,
    )

    result = _cached_features(req.smiles.strip())
    if result is None:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {req.smiles!r}")
    features, _ = result

    pka_type = _pka_type_of(req.pka) if req.pka is not None else None

    try:
        with language_context(req.lang):
            payload = {
                "pka_factors": (
                    analyze_pka_chemistry(req.smiles.strip(), req.pka)
                    if req.pka is not None else None
                ),
                "lipinski": analyze_lipinski(features),
                "druglikeness": analyze_druglikeness(req.smiles.strip()),
                "admet": analyze_admet(req.smiles.strip(), features, req.pka),
                "ionization": (
                    _ionization_profile(req.pka, pka_type)
                    if req.pka is not None else None
                ),
                "pharma_analysis": (
                    _pharma_analysis_key(req.pka, pka_type)
                    if req.pka is not None else None
                ),
                "linkage": (
                    _linkage_prose(req.logs, req.pka, pka_type)
                    if req.logs is not None and req.pka is not None else None
                ),
            }
    except Exception:
        logger.exception("Analysis failed for SMILES %r", req.smiles)
        raise HTTPException(status_code=500, detail="Analysis failed")

    payload = to_jsonable(payload)
    with _cache_lock:
        _analysis_cache[cache_key] = payload
    return payload


# ── AI explanation (Kimi) ──

@router.post("/explain")
def explain(req: ExplainRequest):
    """Kimi AI chemistry explanation (markdown). 24h TTL cache.

    503 when no KIMI_API_KEY is configured; 502 on upstream API errors.
    """
    from core.ai_client import _get_api_key, call_kimi_explain

    if not _get_api_key():
        raise HTTPException(status_code=503, detail="KIMI_API_KEY is not configured")

    cache_key = (
        req.smiles.strip(),
        round(float(req.prediction), 4),
        tuple(sorted((req.features or {}).items())),
        tuple(req.shap_features or ()),
        tuple(req.shap_values or ()),
        req.pka_value,
        req.pka_type,
        req.lang,
    )
    with _cache_lock:
        hit = _kimi_cache.get(cache_key)
    if hit is not None:
        return {"markdown": hit, "cached": True}

    try:
        with language_context(req.lang):
            markdown = call_kimi_explain(
                req.smiles.strip(),
                req.prediction,
                req.features,
                shap_features=req.shap_features,
                shap_values=req.shap_values,
                pka_value=req.pka_value,
                pka_type=req.pka_type,
            )
    except Exception as e:
        logger.exception("Kimi explanation failed")
        raise HTTPException(status_code=502, detail=f"Upstream AI service error: {type(e).__name__}")

    if markdown is None:  # defensive: key disappeared between check and call
        raise HTTPException(status_code=503, detail="KIMI_API_KEY is not configured")

    with _cache_lock:
        _kimi_cache[cache_key] = markdown
    return {"markdown": markdown, "cached": False}


# ── 2D / 3D structures ──

class _UploadAdapter:
    """Give an uploaded file's bytes the interface features.smiles_from_file expects."""

    def __init__(self, data: bytes, name: str):
        self._data = data
        self.name = name

    def getvalue(self):
        return self._data


@router.post("/mol/parse-file")
async def mol_parse_file(file: UploadFile = File(...)):
    """Parse an uploaded molecular file (.mol/.sdf/.mol2/.pdb/.xyz).

    Returns {smiles, formula, mw} (canonical SMILES + formula + molecular weight).
    """
    from features import smiles_from_file

    data = await file.read()
    try:
        parsed = smiles_from_file(_UploadAdapter(data, file.filename or ""))
    except Exception:
        logger.exception("Failed to parse molecular file %r", file.filename)
        parsed = None
    if parsed is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse molecular file: {file.filename!r}",
        )
    smiles, formula, mw = parsed
    return {"smiles": smiles, "formula": formula, "mw": float(mw)}


@router.get("/mol/2d")
def mol_2d(smiles: str = Query(...), bonds: str | None = Query(None)):
    """Dark-theme 2D structure PNG (reuses ui.plots.mol_to_dark_image).

    Optional `bonds` highlights GNN-important bonds via
    ui.plots.mol_to_dark_image_with_importance. Format: "i-j:w,i-j:w,..."
    (atom index pairs, optional :weight in 0-1, default 1.0). Unparseable or
    non-bonded pairs are ignored; when nothing valid remains the plain
    rendering is returned.
    """
    from rdkit import Chem

    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {smiles!r}")

    bond_weights: dict[int, float] = {}
    if bonds:
        for part in bonds.split(","):
            part = part.strip()
            if not part:
                continue
            pair, _, weight_s = part.partition(":")
            ij = pair.split("-")
            if len(ij) != 2:
                continue
            try:
                a, b = int(ij[0]), int(ij[1])
                weight = float(weight_s) if weight_s else 1.0
            except ValueError:
                continue
            if a < 0 or b < 0 or a >= mol.GetNumAtoms() or b >= mol.GetNumAtoms():
                continue
            try:
                bond = mol.GetBondBetweenAtoms(a, b)
            except RuntimeError:
                continue
            if bond is not None:
                bond_weights[bond.GetIdx()] = max(0.0, min(1.0, weight))

    cache_key = (smiles, bonds if bond_weights else None)
    with _cache_lock:
        png = _mol2d_cache.get(cache_key)
    if png is None:
        if bond_weights:
            from ui.plots import mol_to_dark_image_with_importance

            img = mol_to_dark_image_with_importance(mol, bond_weights)
        else:
            from ui.plots import mol_to_dark_image

            img = mol_to_dark_image(mol)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        with _cache_lock:
            _mol2d_cache[cache_key] = png
    return Response(content=png, media_type="image/png")


@router.get("/mol/info")
def mol_info(smiles: str = Query(...)):
    """Basic molecular identity: {formula, mw} via RDKit."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {smiles!r}")
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": float(Descriptors.MolWt(mol)),
    }


@router.get("/mol/3d")
def mol_3d(smiles: str = Query(...)):
    """3D mol block (embed + MMFF optimize) for client-side 3Dmol.js rendering."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smiles = smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            raise ValueError("unparseable SMILES")
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
            raise ValueError("3D embedding failed")
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        # Center on the coordinate centroid (same as ui.plots.show_3d_molecule)
        from rdkit.Geometry import Point3D
        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        cx = sum(conf.GetAtomPosition(i).x for i in range(n_atoms)) / n_atoms
        cy = sum(conf.GetAtomPosition(i).y for i in range(n_atoms)) / n_atoms
        cz = sum(conf.GetAtomPosition(i).z for i in range(n_atoms)) / n_atoms
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D(pos.x - cx, pos.y - cy, pos.z - cz))

        mol = Chem.RemoveHs(mol)
        molblock = Chem.MolToMolBlock(mol)
    except Exception:
        raise HTTPException(
            status_code=400, detail=f"Could not build 3D structure for SMILES: {smiles!r}"
        )
    return {"molblock": molblock}


# ── GNN explanation ──

@router.post("/gnn-explain")
def gnn_explain(req: GnnExplainRequest):
    """GNNExplainer bond + atom-feature attribution for one SMILES.

    Uses the prediction service's GNN singleton (no Streamlit caching).
    """
    from rdkit import Chem

    from services import prediction as svc

    if not svc.gnn_files_exist():
        raise HTTPException(status_code=503, detail="GNN model is not available")
    model, encoder = svc._get_gnn()
    if model is None:
        raise HTTPException(status_code=503, detail="GNN model is not available")

    smiles = req.smiles.strip()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {smiles!r}")
    graph = encoder.mol_to_graph(mol)
    if graph is None:
        raise HTTPException(status_code=400, detail=f"Could not encode SMILES: {smiles!r}")
    edge_index = graph["edge_index"]
    if edge_index.size(1) == 0:
        raise HTTPException(status_code=400, detail="Single-atom molecule: no bonds to explain")

    try:
        from gnn_explainer import GNNExplainer

        explainer = GNNExplainer(model, lr=0.01, epochs=300)
        result = explainer.explain(graph["x"], edge_index)
    except Exception:
        logger.exception("GNN explanation failed for SMILES %r", smiles)
        raise HTTPException(status_code=500, detail="GNN explanation failed")

    bond_importance = result["bond_importance"]
    # Atom-symbol labels for the UI ("C0—O1 (SINGLE)"), same as the old app.
    bond_details = []
    for a, b, imp in bond_importance:
        symbol_a = mol.GetAtomWithIdx(int(a)).GetSymbol()
        symbol_b = mol.GetAtomWithIdx(int(b)).GetSymbol()
        bond = mol.GetBondBetweenAtoms(int(a), int(b))
        bond_type = str(bond.GetBondType()) if bond is not None else "?"
        bond_details.append([int(a), int(b), float(imp), f"{symbol_a}{a}—{symbol_b}{b} ({bond_type})"])

    return to_jsonable({
        "bond_importance": bond_importance,        # [(atom_i, atom_j, importance)]
        "bond_details": bond_details,              # [(atom_i, atom_j, importance, label)]
        "feature_importance": result["feature_importance"],  # 37-dim atom feature weights
        "elapsed": result["elapsed"],
    })


# ── Molecule search (local DB + optional PubChem) ──

@router.get("/molecules")
def molecules_list():
    """Full local molecule database listing (name + SMILES), for quick-select UI."""
    from molecules import MOLECULE_DB

    return {
        "molecules": [
            {"name": name, "smiles": smiles}
            for name, smiles in MOLECULE_DB.items()
            if smiles and name != "(自定义输入)"
        ]
    }


def _display_name(smiles: str, fallback: str) -> str:
    """Reverse-lookup the display name for a SMILES in MOLECULE_DB."""
    from molecules import MOLECULE_DB

    return next(
        (k for k, v in MOLECULE_DB.items() if v == smiles and k != "(自定义输入)"),
        fallback,
    )


@router.get("/molecules/search")
def molecules_search(q: str = Query(...), lang: str = "zh", pubchem: bool = False):
    """Local exact + fuzzy search (mirrors ui/components.py), optional PubChem.

    Runs in the threadpool (def) because PubChem is a blocking HTTP call.
    """
    from molecules import SEARCH_INDEX, search_pubchem

    query = q.strip().lower()
    exact: list[dict] = []
    fuzzy: list[dict] = []
    if query:
        if query in SEARCH_INDEX:
            smiles = SEARCH_INDEX[query]
            exact.append({"name": _display_name(smiles, query), "smiles": smiles})
        else:
            matches = [k for k in SEARCH_INDEX.keys() if query in k or k in query]
            matches.sort(key=lambda x: (0 if x.startswith(query) else 1, len(x)))
            seen_smiles = set()
            for key in matches[:8]:
                smiles = SEARCH_INDEX[key]
                if smiles in seen_smiles:
                    continue
                seen_smiles.add(smiles)
                fuzzy.append({"name": _display_name(smiles, key), "smiles": smiles})

    pubchem_result = None
    if pubchem and query:
        with language_context(lang):
            smiles, status = search_pubchem(q.strip())
        pubchem_result = {"smiles": smiles, "status": status}

    return {"exact": exact, "fuzzy": fuzzy, "pubchem": pubchem_result}


# ── Batch prediction ──

def _validate_batch(smiles_list, mode):
    """Shared validation for both batch endpoints. Returns normalized mode."""
    from services.prediction import _normalize_mode

    if not smiles_list:
        raise HTTPException(status_code=400, detail="Empty SMILES list")
    try:
        return _normalize_mode(mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/batch")
def predict_batch_endpoint(req: BatchRequest):
    """Start a background batch prediction task."""
    normalized = _validate_batch(req.smiles_list, req.mode)
    task_id = registry.create([str(s) for s in req.smiles_list], normalized)
    return {"task_id": task_id, "count": len(req.smiles_list)}


@router.get("/predict/batch/{task_id}")
def predict_batch_status(task_id: str):
    """Poll a batch task: status, progress, results (when done)."""
    task = registry.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    return task


@router.post("/predict/batch-file")
async def predict_batch_file(file: UploadFile = File(...), mode: str = Form("auto")):
    """Batch prediction from a CSV upload; SMILES column auto-detected.

    Column sniffing mirrors app.py's batch flow exactly.
    """
    raw = (await file.read()).decode("utf-8-sig")
    header_line = raw.split("\n", 1)[0]
    header = [c.strip().strip('"').strip("'") for c in header_line.split(",")]
    header_lower = [h.lower() for h in header]

    smiles_col = None
    for keyword in ("smiles", "smile", "smi", "canonical_smiles", "isomeric_smiles"):
        for i, h in enumerate(header_lower):
            if keyword in h:
                smiles_col = i
                break
        if smiles_col is not None:
            break
    if smiles_col is None:
        for i, h in enumerate(header_lower):
            if "mol" in h or "structure" in h or "compound" in h:
                smiles_col = i
                break
    if smiles_col is None:
        smiles_col = 0

    import pandas as pd

    try:
        df = pd.read_csv(io.StringIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    if header[smiles_col] not in df.columns:
        raise HTTPException(
            status_code=400, detail=f"SMILES column {header[smiles_col]!r} not found in CSV"
        )
    smiles_list = df[header[smiles_col]].dropna().astype(str).tolist()

    normalized = _validate_batch(smiles_list, mode)
    task_id = registry.create(smiles_list, normalized)
    return {
        "task_id": task_id,
        "count": len(smiles_list),
        "smiles_column": header[smiles_col],
    }
