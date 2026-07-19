"""Tests for services/prediction.py — framework-free unified prediction service."""

import os

import pytest

from services.prediction import PredictionResult, predict_batch, run_prediction

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

MODEL_USED_LABELS = {"RF", "GNN", "Ensemble", "Ensemble(W)"}
OOD_RISKS = {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}
PKA_KINDS = {"acid", "base", "amphoteric"}


def _assert_common(result, mode):
    """Shared invariants for any successful prediction."""
    assert isinstance(result, PredictionResult)
    assert result.model_selected == mode
    assert result.model_used in MODEL_USED_LABELS
    # Plausible logS range for drug-like molecules
    assert -13.0 <= result.logS_final <= 3.0
    assert list(result.features.keys()) == DESCRIPTOR_KEYS
    assert all(isinstance(v, float) for v in result.features.values())
    assert result.ood_risk in OOD_RISKS
    assert result.pka is None or isinstance(result.pka, float)
    if result.pka is not None:
        assert result.pka_kind in PKA_KINDS
    else:
        assert result.pka_kind is None


class TestRunPredictionModes:
    """run_prediction works for all 4 modes."""

    @pytest.mark.parametrize("mode", ["auto", "rf", "gnn", "ensemble"])
    def test_caffeine_all_modes(self, mode):
        result = run_prediction(CAFFEINE, mode=mode)
        _assert_common(result, mode)

    @pytest.mark.parametrize("mode", ["AUTO", "Rf", "GNN", "Ensemble"])
    def test_mode_case_insensitive(self, mode):
        result = run_prediction(ETHANOL, mode=mode)
        assert result.model_selected == mode.lower()

    def test_rf_mode_pure(self):
        """RF mode: no GNN involved, prediction equals RF output."""
        result = run_prediction(CAFFEINE, mode="rf")
        assert result.model_used == "RF"
        assert result.logS_gnn is None
        assert result.model_disagreement is None
        assert result.logS_final == pytest.approx(result.logS_rf)

    @requires_gnn
    def test_gnn_mode(self):
        """GNN mode: prediction equals GNN output; SHAP skipped (GNN-only)."""
        result = run_prediction(ETHANOL, mode="gnn")
        assert result.model_used == "GNN"
        assert result.logS_gnn is not None
        assert result.logS_final == pytest.approx(result.logS_gnn)
        assert result.shap_values is None
        assert result.shap_names is None
        assert result.shap_base_value is None

    @requires_gnn
    def test_ensemble_mode(self):
        """Ensemble mode: 0.45*RF + 0.55*GNN."""
        result = run_prediction(ETHANOL, mode="ensemble")
        assert result.model_used == "Ensemble"
        assert result.logS_gnn is not None
        expected = 0.45 * result.logS_rf + 0.55 * result.logS_gnn
        assert result.logS_final == pytest.approx(expected)
        assert result.model_disagreement == pytest.approx(abs(result.logS_rf - result.logS_gnn))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            run_prediction(CAFFEINE, mode="xgboost")


class TestShap:
    """SHAP output uses raw English names: 13 descriptor keys + MorganFP."""

    def test_shap_names_raw_english_rf(self):
        result = run_prediction(ASPIRIN, mode="rf")
        assert result.shap_names is not None
        assert len(result.shap_names) == 14
        assert result.shap_names[:13] == DESCRIPTOR_KEYS
        assert result.shap_names[-1] == "MorganFP"

    def test_shap_values_and_base(self):
        result = run_prediction(ASPIRIN, mode="rf")
        assert result.shap_values is not None
        assert len(result.shap_values) == 14
        assert all(isinstance(v, float) for v in result.shap_values)
        assert isinstance(result.shap_base_value, float)
        # SHAP additivity: base + sum(contributions) == RF prediction
        assert result.shap_base_value + sum(result.shap_values) == pytest.approx(
            result.logS_rf, abs=1e-4
        )


class TestPka:
    """pKa prediction (model file present in output_v2)."""

    def test_caffeine_pka(self):
        result = run_prediction(CAFFEINE, mode="rf")
        assert result.pka is not None
        assert isinstance(result.pka, float)
        assert result.pka_kind in PKA_KINDS

    def test_pka_kind_thresholds(self):
        """pKa kind enum: <6 acid, >8 base, else amphoteric (from model.get_pka_type)."""
        from services.prediction import _pka_kind
        assert _pka_kind(4.76) == "acid"
        assert _pka_kind(5.99) == "acid"
        assert _pka_kind(9.5) == "base"
        assert _pka_kind(8.01) == "base"
        assert _pka_kind(7.0) == "amphoteric"


class TestOod:
    """OOD fields populated for auto mode."""

    def test_ood_fields(self):
        result = run_prediction(CAFFEINE, mode="auto")
        assert result.ood_risk in OOD_RISKS
        if result.ood_risk != "UNKNOWN":
            assert 0.0 <= result.ood_score <= 1.0
            assert 0.0 <= result.ood_max_tanimoto <= 1.0
            assert isinstance(result.ood_out_of_range, list)
            assert isinstance(result.ood_extreme, list)

    def test_ood_runs_for_non_auto_modes(self):
        result = run_prediction(ETHANOL, mode="rf")
        assert result.ood_risk in OOD_RISKS


class TestInvalidSmiles:
    """Invalid SMILES handling in single and batch paths."""

    def test_run_prediction_raises(self):
        with pytest.raises(ValueError):
            run_prediction("not_a_smiles")

    def test_empty_smiles_raises(self):
        with pytest.raises(ValueError):
            run_prediction("   ")

    def test_batch_captures_error_rows(self):
        """One invalid row yields an error dict; the rest still succeed."""
        results = predict_batch([CAFFEINE, "not_a_smiles", ETHANOL], mode="rf")
        assert len(results) == 3
        assert isinstance(results[0], PredictionResult)
        assert isinstance(results[1], dict)
        assert results[1]["smiles"] == "not_a_smiles"
        assert "error" in results[1]
        assert isinstance(results[2], PredictionResult)


class TestBatchParity:
    """Batch shares the single-prediction code path: results must match."""

    def test_batch_matches_single_rf(self):
        single = run_prediction(CAFFEINE, mode="rf")
        (batch_result,) = predict_batch([CAFFEINE], mode="rf")
        assert isinstance(batch_result, PredictionResult)
        assert batch_result.logS_final == pytest.approx(single.logS_final)
        assert batch_result.logS_rf == pytest.approx(single.logS_rf)
        assert batch_result.model_used == single.model_used
        assert batch_result.pka == pytest.approx(single.pka)

    @requires_gnn
    def test_batch_matches_single_auto(self):
        single = run_prediction(ASPIRIN, mode="auto")
        (batch_result,) = predict_batch([ASPIRIN], mode="auto")
        assert isinstance(batch_result, PredictionResult)
        assert batch_result.logS_final == pytest.approx(single.logS_final)
        assert batch_result.model_used == single.model_used

    def test_batch_multiple_molecules(self):
        results = predict_batch([CAFFEINE, ETHANOL, ASPIRIN], mode="rf")
        assert len(results) == 3
        assert all(isinstance(r, PredictionResult) for r in results)
        assert [r.smiles for r in results] == [CAFFEINE, ETHANOL, ASPIRIN]


class TestKnownMolecules:
    """Sanity checks on chemically well-understood molecules."""

    def test_ethanol_more_soluble_than_aspirin(self):
        """Ethanol (miscible with water) must be far more soluble than aspirin."""
        ethanol = run_prediction(ETHANOL, mode="rf")
        aspirin = run_prediction(ASPIRIN, mode="rf")
        assert ethanol.logS_final > aspirin.logS_final

    def test_result_smiles_echoes_input(self):
        result = run_prediction(f"  {ETHANOL}  ", mode="rf")
        assert result.smiles == ETHANOL
