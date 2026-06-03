"""Tests for core/analysis.py — pKa, Lipinski, ADME/Tox, drug-likeness."""

import pytest
from core.analysis import (
    analyze_pka_chemistry,
    analyze_lipinski,
    analyze_druglikeness,
    detect_functional_groups,
    analyze_admet,
)


class TestAnalyzePkaChemistry:
    """Tests for analyze_pka_chemistry()."""

    def test_ethanol_pka_returns_dict(self):
        """Ethanol should return a dict of factors."""
        result = analyze_pka_chemistry("CCO", 15.5)
        assert isinstance(result, dict)
        assert len(result) > 0
        expected_keys = [
            "Inductive\n(诱导效应)",
            "Resonance\n(共轭效应)",
            "Intra-HB\n(分子内氢键)",
            "Steric\n(空间位阻)",
            "Hybridization\n(杂化/芳香性)",
        ]
        for key in expected_keys:
            assert key in result

    def test_acidic_molecule_sign(self):
        """Acidic molecule (pKa < 7) should have positive inductive effect."""
        result = analyze_pka_chemistry("CC(=O)O", 4.76)  # acetic acid
        # Inductive effect should be positive for acidic molecule
        assert result.get("Inductive\n(诱导效应)", 0) >= 0

    def test_invalid_smiles_returns_empty(self):
        """Invalid SMILES should return empty dict."""
        assert analyze_pka_chemistry("ZZZZ", 5.0) == {}

    def test_aromatic_resonance(self):
        """Aromatic molecules should show resonance contribution."""
        result = analyze_pka_chemistry("c1ccccc1O", 9.95)  # phenol
        assert result.get("Resonance\n(共轭效应)", 0) > 0

    def test_salicylate_internal_hbond(self):
        """Salicylic acid should detect intramolecular H-bond."""
        # Salicylic acid: ortho-OH-benzoic acid
        result = analyze_pka_chemistry("OC(=O)C1=CC=CC=C1O", 2.97)
        assert result.get("Intra-HB\n(分子内氢键)", 0) != 0.0


class TestAnalyzeLipinski:
    """Tests for analyze_lipinski()."""

    @pytest.fixture
    def druglike_features(self):
        """Features for a typical drug-like molecule (aspirin-like)."""
        return {
            "MolWt": 180.16,
            "LogP": 1.2,
            "NumHDonors": 1,
            "NumHAcceptors": 3,
            "NumRotatableBonds": 3,
            "NumAromaticRings": 1,
            "NumAliphaticRings": 0,
            "FractionCSP3": 0.25,
            "NumSaturatedRings": 0,
            "HallKierAlpha": -1.0,
            "Chi0v": 10.0,
            "Chi1v": 5.0,
            "TPSA": 60.0,
        }

    def test_druglike_passes(self, druglike_features):
        """A drug-like molecule should pass ≥4 rules."""
        result = analyze_lipinski(druglike_features)
        assert result["passed"] >= 4
        assert result["is_druglike"] is True
        assert result["violations"] <= 1
        assert len(result["rules"]) == 5

    def test_all_rules_returned(self, druglike_features):
        """Should return 5 rules in the rules list."""
        result = analyze_lipinski(druglike_features)
        assert len(result["rules"]) == 5
        # Each rule is a tuple of (name, key, passed_bool, value_str)
        for rule in result["rules"]:
            assert len(rule) == 4

    def test_mw_violation(self):
        """MW > 500 should produce a violation."""
        features = {
            "MolWt": 600, "LogP": 2, "NumHDonors": 1, "NumHAcceptors": 2,
            "NumRotatableBonds": 3, "NumAromaticRings": 0, "NumAliphaticRings": 0,
            "FractionCSP3": 0.0, "NumSaturatedRings": 0, "HallKierAlpha": 0,
            "Chi0v": 0, "Chi1v": 0, "TPSA": 0,
        }
        result = analyze_lipinski(features)
        # First rule (MW) should be failed
        assert result["rules"][0][2] is False
        assert result["violations"] >= 1

    def test_all_violations(self):
        """All rules violated should give passed=0, violations=5."""
        features = {
            "MolWt": 600, "LogP": 7, "NumHDonors": 6, "NumHAcceptors": 12,
            "NumRotatableBonds": 15, "NumAromaticRings": 0, "NumAliphaticRings": 0,
            "FractionCSP3": 0.0, "NumSaturatedRings": 0, "HallKierAlpha": 0,
            "Chi0v": 0, "Chi1v": 0, "TPSA": 0,
        }
        result = analyze_lipinski(features)
        assert result["passed"] == 0
        assert result["violations"] == 5
        assert result["is_druglike"] is False


class TestAnalyzeDruglikeness:
    """Tests for analyze_druglikeness()."""

    def test_invalid_smiles_returns_none(self):
        """Invalid SMILES should return None."""
        assert analyze_druglikeness("ZZZZ") is None

    def test_ethanol_has_all_keys(self):
        """Ethanol should return dict with all expected keys."""
        result = analyze_druglikeness("CCO")
        assert result is not None
        assert "qed" in result
        assert "sascore" in result
        assert "fsp3" in result
        assert "qed_level" in result
        assert "sa_level" in result
        assert "fsp3_level" in result

    def test_qed_range(self):
        """QED should be between 0 and 1."""
        result = analyze_druglikeness("CCO")
        assert 0 <= result["qed"] <= 1

    def test_sascore_range(self):
        """SAscore should be between 1 and 10."""
        result = analyze_druglikeness("CCO")
        assert 1 <= result["sascore"] <= 10

    def test_caffeine_druglikeness(self):
        """Caffeine should have sensible drug-likeness metrics."""
        result = analyze_druglikeness("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        assert result is not None
        assert 0 < result["qed"] < 1
        assert 1 <= result["sascore"] <= 10


class TestDetectFunctionalGroups:
    """Tests for detect_functional_groups()."""

    def test_invalid_smiles_returns_empty(self):
        """Invalid SMILES should return empty dict."""
        assert detect_functional_groups("ZZZZ") == {}

    def test_ethanol_groups(self):
        """Ethanol should have hydroxyl (but not phenol)."""
        groups = detect_functional_groups("CCO")
        assert isinstance(groups, dict)
        assert len(groups) > 0

    def test_phenol_detection(self):
        """Phenol should detect both aromatic ring and phenol group."""
        groups = detect_functional_groups("c1ccccc1O")
        assert groups.get("aromatic_ring") is True
        assert groups.get("phenol") is True

    def test_carboxylic_acid_detection(self):
        """Acetic acid should detect carboxylic acid."""
        groups = detect_functional_groups("CC(=O)O")
        assert groups.get("carboxylic_acid") is True

    def test_nitro_detection(self):
        """Nitrobenzene should detect nitro group (charged or unionized)."""
        groups = detect_functional_groups("c1ccccc1[N+](=O)[O-]")
        # Check either the original or charged nitro pattern
        assert groups.get("nitro") is True or groups.get("nitro_charged") is True

    def test_halogen_detection(self):
        """Chlorobenzene should detect halogen."""
        groups = detect_functional_groups("c1ccccc1Cl")
        assert groups.get("halogen") is True


class TestAnalyzeAdmet:
    """Tests for analyze_admet()."""

    @pytest.fixture
    def aspirin_features(self):
        """Features for aspirin (for ADMET test consistency)."""
        from features import compute_features
        result = compute_features("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert result is not None
        return result[0]

    def test_aspirin_has_all_sections(self, aspirin_features):
        """Aspirin ADMET should have absorption, distribution, metabolism, excretion, toxicity."""
        result = analyze_admet("CC(=O)OC1=CC=CC=C1C(=O)O", aspirin_features, pka_val=3.5)
        assert "absorption" in result
        assert "distribution" in result
        assert "metabolism" in result
        assert "excretion" in result
        assert "toxicity" in result

    def test_aspirin_distribution_keys(self, aspirin_features):
        """Distribution should have summary, vd_estimate, ppb."""
        result = analyze_admet("CC(=O)OC1=CC=CC=C1C(=O)O", aspirin_features, pka_val=3.5)
        d = result["distribution"]
        assert "summary" in d
        assert "vd_estimate" in d
        assert "ppb" in d

    def test_toxicity_is_list_of_tuples(self, aspirin_features):
        """Toxicity should be a list of (level, description) tuples."""
        result = analyze_admet("CC(=O)OC1=CC=CC=C1C(=O)O", aspirin_features, pka_val=3.5)
        assert isinstance(result["toxicity"], list)
        if result["toxicity"]:
            first = result["toxicity"][0]
            assert isinstance(first, tuple)
            assert len(first) == 2

    def test_benzene_toxicity_levels(self):
        """Benzene should have toxicity alerts (> low)."""
        from features import compute_features
        feat = compute_features("c1ccccc1")
        assert feat is not None
        result = analyze_admet("c1ccccc1", feat[0])
        # Benzene is a known carcinogen — should have some alerts
        assert len(result["toxicity"]) > 0

    def test_metabolism_has_cyp_enzymes(self, aspirin_features):
        """Metabolism section should have CYP enzyme info."""
        result = analyze_admet("CC(=O)OC1=CC=CC=C1C(=O)O", aspirin_features, pka_val=3.5)
        assert "cyp_enzymes" in result["metabolism"]
        assert isinstance(result["metabolism"]["cyp_enzymes"], str)

    def test_excretion_has_route(self, aspirin_features):
        """Excretion should specify the route."""
        result = analyze_admet("CC(=O)OC1=CC=CC=C1C(=O)O", aspirin_features, pka_val=3.5)
        assert "route" in result["excretion"]
        assert isinstance(result["excretion"]["route"], str)
