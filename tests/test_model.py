"""Tests for model.py — prediction utilities, pKa type, solubility levels, ensemble."""

import pytest
from model import (
    get_pka_type,
    get_solubility_level,
    predict_solubility_ensemble,
    predict_solubility_weighted,
    predict_solubility_auto,
)


class TestGetPkaType:
    """Tests for get_pka_type()."""

    def test_acidic_pka(self):
        """pKa < 6 should be classified as acidic."""
        result = get_pka_type(4.76)
        assert result[0] == "acid"
        assert "酸性" in result[1]

    def test_basic_pka(self):
        """pKa > 8 should be classified as basic."""
        result = get_pka_type(9.5)
        assert result[0] == "base"
        assert "碱性" in result[1]

    def test_amphoteric_pka(self):
        """pKa between 6 and 8 should be amphoteric."""
        result = get_pka_type(7.0)
        assert result[0] == "amphoteric"
        assert "两性" in result[1] or "中性" in result[1]

    def test_edge_case_acidic_boundary(self):
        """pKa = 5.99 should still be acidic."""
        result = get_pka_type(5.99)
        assert result[0] == "acid"

    def test_edge_case_basic_boundary(self):
        """pKa = 8.01 should still be basic."""
        result = get_pka_type(8.01)
        assert result[0] == "base"

    def test_returns_4_elements(self):
        """Should return 4 elements: type, label, css_class, color, description."""
        result = get_pka_type(4.76)
        assert len(result) == 5

    def test_color_strings(self):
        """Color strings should be valid hex colors."""
        for pka in [3.0, 7.0, 10.0]:
            result = get_pka_type(pka)
            color = result[3]
            assert color.startswith("#")
            assert len(color) == 7


class TestGetSolubilityLevel:
    """Tests for get_solubility_level()."""

    def test_highly_soluble(self):
        """logS > 0 should be highly soluble."""
        level, color, css = get_solubility_level(1.5)
        assert "Highly" in level or "易溶" in level

    def test_moderately_soluble(self):
        """-2 < logS <= 0 should be moderately soluble."""
        level, color, css = get_solubility_level(-1.0)
        assert "Moderately" in level or "中等" in level

    def test_poorly_soluble(self):
        """logS <= -2 should be poorly soluble."""
        level, color, css = get_solubility_level(-3.0)
        assert "Poorly" in level or "难溶" in level

    def test_boundary_zero(self):
        """logS = 0 should be moderately soluble (since 0 > 0 is False)."""
        level, _, _ = get_solubility_level(0)
        assert "Moderately" in level or "中等" in level

    def test_boundary_negative_two(self):
        """logS = -2 should be poorly soluble."""
        level1, _, _ = get_solubility_level(-2)
        assert "Poorly" in level1 or "难溶" in level1
        level2, _, _ = get_solubility_level(-1.999)
        assert "Moderately" in level2 or "中等" in level2

    def test_returns_3_elements(self):
        """Should return (label, color, css_class)."""
        result = get_solubility_level(-1.0)
        assert len(result) == 3
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)
        assert isinstance(result[2], str)


class TestPredictSolubilityEnsemble:
    """Tests for predict_solubility_ensemble()."""

    def test_ensemble_mean_weighted(self):
        """Ensemble should return weighted average (0.45/0.55)."""
        ensemble, rf_p, gnn_p = predict_solubility_ensemble(-1.0, -2.0)
        assert rf_p == -1.0
        assert gnn_p == -2.0
        # 0.45 * (-1.0) + 0.55 * (-2.0) = -0.45 - 1.10 = -1.55
        assert ensemble == pytest.approx(-1.55, abs=0.01)

    def test_identical_models(self):
        """When RF and GNN agree, ensemble should equal that value."""
        ensemble, _, _ = predict_solubility_ensemble(-1.5, -1.5)
        assert ensemble == pytest.approx(-1.5, abs=0.01)

    def test_large_disagreement(self):
        """Even with large disagreement, formula should hold."""
        ensemble, _, _ = predict_solubility_ensemble(2.0, -2.0)
        assert ensemble == pytest.approx(0.45 * 2.0 + 0.55 * (-2.0), abs=0.01)


class TestPredictSolubilityWeighted:
    """Tests for predict_solubility_weighted()."""

    def test_default_weight(self):
        """Default weight should be 0.45."""
        result = predict_solubility_weighted(-1.0, -2.0)
        assert result == pytest.approx(-1.55, abs=0.01)

    def test_custom_weight(self):
        """Custom RF weight should work."""
        result = predict_solubility_weighted(-1.0, -2.0, rf_weight=0.5)
        assert result == pytest.approx(-1.5, abs=0.01)

    def test_rf_only(self):
        """rf_weight=1.0 should use RF only."""
        result = predict_solubility_weighted(-1.0, -2.0, rf_weight=1.0)
        assert result == -1.0

    def test_gnn_only(self):
        """rf_weight=0.0 should use GNN only."""
        result = predict_solubility_weighted(-1.0, -2.0, rf_weight=0.0)
        assert result == -2.0


class TestPredictSolubilityAuto:
    """Tests for predict_solubility_auto() — Auto+ strategy."""

    def test_no_gnn_uses_rf(self):
        """When GNN is None, should fall back to RF."""
        pred, label, disc = predict_solubility_auto("LOW", -1.0, None)
        assert pred == -1.0
        assert label == "RF"
        assert disc == 0.0

    def test_high_disagreement_uses_gnn(self):
        """Disagreement > 1.0 should select GNN."""
        pred, label, disc = predict_solubility_auto("LOW", -1.0, -2.5)
        assert pred == -2.5
        assert label == "GNN"
        assert disc > 1.0

    def test_ood_medium_uses_gnn(self):
        """MEDIUM OOD should select GNN."""
        pred, label, _ = predict_solubility_auto("MEDIUM", -1.0, -1.5)
        assert label == "GNN"

    def test_ood_high_uses_gnn(self):
        """HIGH OOD should select GNN."""
        pred, label, _ = predict_solubility_auto("HIGH", -1.0, -1.2)
        assert label == "GNN"

    def test_low_ood_ensemble(self):
        """LOW OOD + low disagreement should use weighted ensemble."""
        pred, label, disc = predict_solubility_auto("LOW", -1.0, -1.2)
        assert label == "Ensemble(W)"
        assert disc < 1.0
        # 0.45 * (-1.0) + 0.55 * (-1.2) = -0.45 - 0.66 = -1.11
        assert pred == pytest.approx(-1.11, abs=0.01)

    def test_disagreement_value_returned(self):
        """Should return correct disagreement."""
        _, _, disc = predict_solubility_auto("LOW", -1.0, -0.5)
        assert disc == pytest.approx(0.5, abs=0.01)

    def test_barely_under_disagreement_threshold(self):
        """Disagreement exactly 1.0 should still use ensemble."""
        pred, label, _ = predict_solubility_auto("LOW", -1.0, 0.0)
        assert label == "Ensemble(W)"
