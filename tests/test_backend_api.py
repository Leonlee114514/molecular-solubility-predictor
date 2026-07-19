"""Tests for the FastAPI backend (backend/) using fastapi.testclient.TestClient.

Covers: health, single prediction (all modes), analysis with language
switching, 2D/3D structures, GNN explanation, molecule search, batch task
lifecycle, CSV batch upload, and the AI-explanation no-key error path
(monkeypatched — never calls the real API).
"""

import os
import time

import pytest
from fastapi.testclient import TestClient

from backend.main import app

# ── Well-known SMILES ──
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

DESCRIPTOR_KEYS = [
    "MolWt", "LogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "FractionCSP3", "NumSaturatedRings", "HallKierAlpha", "Chi0v", "Chi1v",
]

_GNN_FILES = [
    os.path.join("output_v2", f)
    for f in ("gnn_solubility_model_v4.pt", "gnn_solubility_model_v3.pt", "gnn_solubility_model.pt")
]
_GNN_AVAILABLE = any(os.path.exists(p) for p in _GNN_FILES)
requires_gnn = pytest.mark.skipif(not _GNN_AVAILABLE, reason="GNN model files not found")


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _poll_batch(client, task_id, timeout=240.0):
    """Poll a batch task until it leaves 'running' (or the deadline hits)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/predict/batch/{task_id}")
        assert resp.status_code == 200
        payload = resp.json()
        if payload["status"] != "running":
            return payload
        time.sleep(0.5)
    raise TimeoutError(f"Batch task {task_id} still running after {timeout}s")


class TestHealth:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert set(data["models"].keys()) == {"rf", "pka", "gnn", "ood"}
        assert data["models"]["rf"]["available"] is True
        assert data["models"]["pka"]["available"] is True
        assert data["models"]["ood"]["available"] is True


class TestPredict:
    def test_predict_rf_caffeine(self, client):
        resp = client.post("/api/predict", json={"smiles": CAFFEINE, "mode": "rf"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_used"] == "RF"
        assert data["model_selected"] == "rf"
        assert -13.0 <= data["logS_final"] <= 3.0
        assert data["logS_gnn"] is None
        assert list(data["features"].keys()) == DESCRIPTOR_KEYS
        # SHAP names are RAW English: 13 descriptor keys + MorganFP
        assert data["shap_names"][:13] == DESCRIPTOR_KEYS
        assert data["shap_names"][-1] == "MorganFP"
        assert len(data["shap_values"]) == 14
        assert data["ood_risk"] in ("LOW", "MEDIUM", "HIGH", "UNKNOWN")
        assert data["pka"] is None or isinstance(data["pka"], float)
        if data["pka"] is not None:
            assert data["pka_kind"] in ("acid", "base", "amphoteric")
        # JSON round-trip already proved numpy types were converted
        assert isinstance(data["logS_rf"], float)

    def test_predict_auto(self, client):
        resp = client.post("/api/predict", json={"smiles": ASPIRIN, "mode": "auto"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_used"] in ("RF", "GNN", "Ensemble", "Ensemble(W)")
        assert -13.0 <= data["logS_final"] <= 3.0

    @requires_gnn
    def test_predict_gnn(self, client):
        resp = client.post("/api/predict", json={"smiles": ETHANOL, "mode": "gnn"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_used"] == "GNN"
        assert data["logS_gnn"] is not None
        assert data["logS_final"] == pytest.approx(data["logS_gnn"])
        assert data["shap_values"] is None  # SHAP skipped for GNN-only

    @requires_gnn
    def test_predict_ensemble(self, client):
        resp = client.post("/api/predict", json={"smiles": ETHANOL, "mode": "ensemble"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_used"] == "Ensemble"
        expected = 0.45 * data["logS_rf"] + 0.55 * data["logS_gnn"]
        assert data["logS_final"] == pytest.approx(expected)

    def test_predict_invalid_smiles_400(self, client):
        resp = client.post("/api/predict", json={"smiles": "not_a_smiles", "mode": "rf"})
        assert resp.status_code == 400
        assert "detail" in resp.json()

    def test_predict_invalid_mode_400(self, client):
        resp = client.post("/api/predict", json={"smiles": ETHANOL, "mode": "xgboost"})
        assert resp.status_code == 400


class TestAnalysis:
    def test_analysis_en(self, client):
        resp = client.post("/api/analysis", json={"smiles": ASPIRIN, "pka": 3.5, "lang": "en"})
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {
            "pka_factors", "lipinski", "druglikeness", "admet",
            "ionization", "pharma_analysis", "linkage",
        }
        # pKa factors: 5 translated factor names -> scores
        assert data["pka_factors"] is not None
        assert len(data["pka_factors"]) == 5
        # Lipinski: aspirin passes Ro5 -> English prose
        assert data["lipinski"]["is_druglike"] is True
        assert "bioavailability" in data["lipinski"]["interpretation"]
        # Drug-likeness metrics present
        assert 0.0 <= data["druglikeness"]["qed"] <= 1.0
        # ADMET sections present
        for section in ("absorption", "distribution", "metabolism", "excretion", "toxicity"):
            assert section in data["admet"]

    def test_analysis_zh(self, client):
        resp = client.post("/api/analysis", json={"smiles": ASPIRIN, "pka": 3.5, "lang": "zh"})
        assert resp.status_code == 200
        data = resp.json()
        assert "口服" in data["lipinski"]["interpretation"]

    def test_analysis_without_pka(self, client):
        resp = client.post("/api/analysis", json={"smiles": ETHANOL})
        assert resp.status_code == 200
        data = resp.json()
        assert data["pka_factors"] is None
        assert data["lipinski"] is not None

    def test_analysis_invalid_smiles_400(self, client):
        resp = client.post("/api/analysis", json={"smiles": "not_a_smiles"})
        assert resp.status_code == 400

    def test_analysis_ionization_acid(self, client):
        """Aspirin (pKa 3.5, acid): ionization profile + linkage prose (en)."""
        resp = client.post("/api/analysis", json={
            "smiles": ASPIRIN, "pka": 3.5, "logs": -1.2, "lang": "en",
        })
        assert resp.status_code == 200
        data = resp.json()
        ion = data["ionization"]
        assert [e["env"] for e in ion] == ["stomach", "duodenum", "intestine", "blood"]
        assert [e["ph"] for e in ion] == [1.5, 4.5, 6.8, 7.4]
        # acid: f = 1 / (1 + 10^(pH - pKa)); stomach pH 1.5 -> ~99% unionized
        assert ion[0]["pct"] == pytest.approx(1 / (1 + 10 ** (1.5 - 3.5)) * 100, abs=0.01)
        # blood pH 7.4 -> almost fully ionized
        assert ion[3]["pct"] < 0.1
        assert data["pharma_analysis"] == "strong_acid"
        assert "empty stomach" in data["linkage"]

    def test_analysis_ionization_base(self, client):
        """Base with pKa 9.8 uses the base formula and the strong-base combo."""
        resp = client.post("/api/analysis", json={
            "smiles": ETHANOL, "pka": 9.8, "logs": 0.5, "lang": "en",
        })
        assert resp.status_code == 200
        data = resp.json()
        ion = data["ionization"]
        # base: f = 1 / (1 + 10^(pKa - pH)); blood pH 7.4 -> ~0.4% unionized
        assert ion[3]["pct"] == pytest.approx(1 / (1 + 10 ** (9.8 - 7.4)) * 100, abs=0.01)
        assert data["pharma_analysis"] == "strong_base"
        # logS > 0 + strong base -> combo_acceptable
        assert "acceptable bioavailability" in data["linkage"]

    def test_analysis_no_linkage_without_logs(self, client):
        """pKa present but logS missing -> ionization yes, linkage no."""
        resp = client.post("/api/analysis", json={"smiles": ASPIRIN, "pka": 3.5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ionization"] is not None
        assert data["linkage"] is None

    def test_analysis_no_ionization_without_pka(self, client):
        resp = client.post("/api/analysis", json={"smiles": ETHANOL, "logs": 0.5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ionization"] is None
        assert data["pharma_analysis"] is None
        assert data["linkage"] is None


class TestExplain:
    def test_explain_no_key_503(self, client, monkeypatch):
        """No API key -> clean 503 JSON (monkeypatched; never calls the real API)."""
        monkeypatch.setattr("core.ai_client._get_api_key", lambda: None)
        resp = client.post("/api/explain", json={
            "smiles": ETHANOL,
            "prediction": -0.3,
            "features": {"MolWt": 46.07, "LogP": -0.18},
        })
        assert resp.status_code == 503
        assert "detail" in resp.json()


class TestMolStructures:
    def test_mol_2d_png(self, client):
        resp = client.get("/api/mol/2d", params={"smiles": ETHANOL})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert len(resp.content) > 100
        assert resp.content[:4] == b"\x89PNG"

    def test_mol_2d_invalid_400(self, client):
        resp = client.get("/api/mol/2d", params={"smiles": "not_a_smiles"})
        assert resp.status_code == 400

    def test_mol_3d_molblock(self, client):
        from rdkit import Chem

        resp = client.get("/api/mol/3d", params={"smiles": ETHANOL})
        assert resp.status_code == 200
        molblock = resp.json()["molblock"]
        assert isinstance(molblock, str) and "\n" in molblock
        # Mol block re-parses; ethanol has 3 heavy atoms
        mol = Chem.MolFromMolBlock(molblock)
        assert mol is not None
        assert mol.GetNumAtoms() == 3
        # Atom lines present in V2000 format
        atom_lines = [ln for ln in molblock.splitlines()[4:] if ln.strip()]
        assert len(atom_lines) >= 3

    def test_mol_3d_invalid_400(self, client):
        resp = client.get("/api/mol/3d", params={"smiles": "not_a_smiles"})
        assert resp.status_code == 400

    def test_mol_info(self, client):
        resp = client.get("/api/mol/info", params={"smiles": ETHANOL})
        assert resp.status_code == 200
        data = resp.json()
        assert data["formula"] == "C2H6O"
        assert 45.0 < data["mw"] < 48.0

    def test_mol_info_invalid_400(self, client):
        resp = client.get("/api/mol/info", params={"smiles": "not_a_smiles"})
        assert resp.status_code == 400

    def test_mol_2d_with_bonds_highlight(self, client):
        """bonds=i-j:w highlights valid atom pairs; garbage pairs are ignored."""
        resp = client.get("/api/mol/2d", params={"smiles": ETHANOL, "bonds": "0-1:0.9,1-2:0.5"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content[:4] == b"\x89PNG"

    def test_mol_2d_bonds_garbage_falls_back(self, client):
        """Unparseable/out-of-range bond specs degrade to the plain rendering."""
        resp = client.get("/api/mol/2d", params={"smiles": ETHANOL, "bonds": "xyz,7-9"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"


class TestMolParseFile:
    def test_parse_mol_file(self, client):
        """A valid .mol file yields canonical SMILES + formula + MW."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles(ETHANOL))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        molblock = Chem.MolToMolBlock(mol)

        resp = client.post(
            "/api/mol/parse-file",
            files={"file": ("ethanol.mol", molblock, "chemical/x-mdl-molfile")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["formula"] == "C2H6O"
        assert 45.0 < data["mw"] < 48.0
        assert Chem.MolFromSmiles(data["smiles"]) is not None
        # Round-trip: parsed SMILES is the same molecule
        assert Chem.MolToInchiKey(Chem.MolFromSmiles(data["smiles"])) == \
            Chem.MolToInchiKey(Chem.MolFromSmiles(ETHANOL))

    def test_parse_file_invalid_400(self, client):
        resp = client.post(
            "/api/mol/parse-file",
            files={"file": ("junk.mol", b"this is not a mol file", "text/plain")},
        )
        assert resp.status_code == 400
        assert "detail" in resp.json()


class TestMoleculesList:
    def test_molecules_list(self, client):
        resp = client.get("/api/molecules")
        assert resp.status_code == 200
        molecules = resp.json()["molecules"]
        assert len(molecules) > 300
        assert all("name" in m and "smiles" in m for m in molecules)
        assert all(m["smiles"] for m in molecules)
        names = [m["name"] for m in molecules]
        assert "(自定义输入)" not in names
        # Ethanol is present with its bilingual display name
        ethanol = next(m for m in molecules if m["smiles"] == "CCO")
        assert "Ethanol" in ethanol["name"]


class TestGnnExplain:
    @requires_gnn
    def test_gnn_explain(self, client):
        resp = client.post("/api/gnn-explain", json={"smiles": CAFFEINE})
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"bond_importance", "bond_details", "feature_importance", "elapsed"}
        assert len(data["bond_importance"]) > 0
        # Each entry: (atom_i, atom_j, importance 0..1)
        a, b, imp = data["bond_importance"][0]
        assert isinstance(a, int) and isinstance(b, int)
        assert 0.0 <= imp <= 1.0
        # bond_details adds a human-readable label like "C0—N1 (SINGLE)"
        da, db, dimp, label = data["bond_details"][0]
        assert (da, db) == (a, b) and dimp == pytest.approx(imp)
        assert "—" in label and "(" in label
        assert len(data["feature_importance"]) == 37
        assert data["elapsed"] > 0
        # RDKit Mol object must not leak into the payload
        assert "mol" not in data

    @requires_gnn
    def test_gnn_explain_invalid_400(self, client):
        resp = client.post("/api/gnn-explain", json={"smiles": "not_a_smiles"})
        assert resp.status_code == 400


class TestMoleculeSearch:
    def test_search_exact(self, client):
        resp = client.get("/api/molecules/search", params={"q": "ethanol"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["exact"], "expected an exact match for 'ethanol'"
        assert data["exact"][0]["smiles"] == "CCO"
        assert "Ethanol" in data["exact"][0]["name"]
        assert data["pubchem"] is None  # not requested

    def test_search_fuzzy(self, client):
        resp = client.get("/api/molecules/search", params={"q": "ethano"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["exact"] == []
        assert any(m["smiles"] == "CCO" for m in data["fuzzy"])

    def test_search_no_match(self, client):
        resp = client.get("/api/molecules/search", params={"q": "zzzqqqxxx"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["exact"] == []
        assert data["fuzzy"] == []
        assert data["pubchem"] is None


class TestBatch:
    def test_batch_lifecycle(self, client):
        resp = client.post("/api/predict/batch", json={
            "smiles_list": [ETHANOL, "not_a_smiles"],
            "mode": "rf",
        })
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        payload = _poll_batch(client, task_id)
        assert payload["status"] == "done"
        assert payload["progress"] == {"done": 2, "total": 2}
        results = payload["results"]
        assert len(results) == 2
        # Row 1: valid result
        assert results[0]["smiles"] == ETHANOL
        assert -13.0 <= results[0]["logS_final"] <= 3.0
        assert results[0]["model_used"] == "RF"
        # Row 2: error dict, not an exception
        assert results[1]["smiles"] == "not_a_smiles"
        assert "error" in results[1]

    def test_batch_empty_400(self, client):
        resp = client.post("/api/predict/batch", json={"smiles_list": [], "mode": "rf"})
        assert resp.status_code == 400

    def test_batch_unknown_task_404(self, client):
        resp = client.get("/api/predict/batch/does-not-exist")
        assert resp.status_code == 404

    def test_batch_file_csv(self, client):
        csv_content = "smiles\nCCO\nCC(=O)Oc1ccccc1C(=O)O\n"
        resp = client.post(
            "/api/predict/batch-file",
            files={"file": ("batch.csv", csv_content, "text/csv")},
            data={"mode": "rf"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["smiles_column"] == "smiles"
        assert body["count"] == 2

        payload = _poll_batch(client, body["task_id"])
        assert payload["status"] == "done"
        assert len(payload["results"]) == 2
        assert all("logS_final" in row for row in payload["results"])

    def test_batch_file_custom_column(self, client):
        """SMILES column auto-detection works for non-'smiles' headers too."""
        csv_content = "id,canonical_smiles\n1,CCO\n"
        resp = client.post(
            "/api/predict/batch-file",
            files={"file": ("batch.csv", csv_content, "text/csv")},
            data={"mode": "rf"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["smiles_column"] == "canonical_smiles"
        payload = _poll_batch(client, body["task_id"])
        assert payload["status"] == "done"
        assert payload["results"][0]["smiles"] == "CCO"
