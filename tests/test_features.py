"""Tests for features.py — molecular descriptor computation."""

import numpy as np
import pytest
from features import compute_features, smiles_from_file


def _make_valid_mol_block():
    """Generate a valid MOL block for methane using RDKit."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("C")
    if mol:
        return Chem.MolToMolBlock(mol)
    return None


class TestComputeFeatures:
    """Tests for compute_features()."""

    def test_ethanol_features(self):
        """Ethanol should produce valid features with correct keys."""
        result = compute_features("CCO")
        assert result is not None
        features, fp = result
        assert isinstance(features, dict)
        # All 13 descriptor keys present
        expected_keys = {
            "MolWt", "LogP", "NumHDonors", "NumHAcceptors",
            "TPSA", "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
            "FractionCSP3", "NumSaturatedRings", "HallKierAlpha", "Chi0v", "Chi1v",
        }
        assert set(features.keys()) == expected_keys

    def test_ethanol_descriptor_values(self):
        """Ethanol descriptor values should be physically plausible."""
        features, _ = compute_features("CCO")
        # MolWt of ethanol = 46.07
        assert 44 < features["MolWt"] < 48
        # LogP should be low (ethanol is hydrophilic)
        assert -1 < features["LogP"] < 0
        # 1 hydroxyl H-donor
        assert features["NumHDonors"] == 1
        # 1 oxygen H-acceptor
        assert features["NumHAcceptors"] == 1
        # TPSA of ethanol ~ 20.23 Å²
        assert 18 < features["TPSA"] < 22
        # 0 rings
        assert features["NumAromaticRings"] == 0
        assert features["NumAliphaticRings"] == 0
        assert features["NumSaturatedRings"] == 0
        # 0-1 rotatable bonds
        assert features["NumRotatableBonds"] <= 1
        # FractionCSP3: all 2 carbons are sp3
        assert features["FractionCSP3"] == 1.0

    def test_aspirin_features(self):
        """Aspirin should have ring-containing features."""
        result = compute_features("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert result is not None
        features, _ = result
        # Aspirin has one aromatic ring
        assert features["NumAromaticRings"] == 1
        # Molecular weight ~ 180
        assert 170 < features["MolWt"] < 190
        # LogP ~ 1.2
        assert 0.5 < features["LogP"] < 2.5

    def test_caffeine_features(self):
        """Caffeine should have multiple aromatic rings."""
        result = compute_features("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        assert result is not None
        features, _ = result
        # Caffeine has 2 aromatic rings (purine)
        assert features["NumAromaticRings"] >= 2
        # MW ~ 194
        assert 190 < features["MolWt"] < 200
        # 0 H-donors, multiple acceptors
        assert features["NumHDonors"] == 0
        assert features["NumHAcceptors"] >= 3

    def test_water_features(self):
        """Water should give valid minimal features."""
        result = compute_features("O")
        assert result is not None
        features, _ = result
        assert features["MolWt"] == pytest.approx(18.015, rel=0.01)
        # RDKit reports 0 H-donors and 0 H-acceptors for isolated water
        assert features["NumRotatableBonds"] == 0

    def test_invalid_smiles(self):
        """Invalid SMILES should return None."""
        result = compute_features("CCOOORRRZZZ###")
        assert result is None

    def test_empty_string(self):
        """Empty string should return None."""
        assert compute_features("") is None
        assert compute_features(None) is None

    def test_whitespace_only(self):
        """Whitespace-only string should return None."""
        assert compute_features("   ") is None

    def test_fingerprint_shape(self):
        """Fingerprint should be 1024-bit numpy array of ints."""
        _, fp = compute_features("CCO")
        assert isinstance(fp, np.ndarray)
        assert fp.shape == (1024,)
        assert fp.dtype.kind in ("i", "u")  # integer type
        assert fp.sum() > 0  # at least some bits set

    def test_deterministic_fingerprint(self):
        """Same SMILES should produce identical fingerprint."""
        _, fp1 = compute_features("CCO")
        _, fp2 = compute_features("CCO")
        np.testing.assert_array_equal(fp1, fp2)

    def test_different_molecules_different_fingerprints(self):
        """Different SMILES should yield different fingerprints."""
        _, fp_eth = compute_features("CCO")
        _, fp_asp = compute_features("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert not np.array_equal(fp_eth, fp_asp)

    def test_aromatic_rings_consistency(self):
        """Benzene-like aromatic count should match expectation."""
        # C6H12 (cyclohexane) — aliphatic
        feat_cyclo, _ = compute_features("C1CCCCC1")
        # Benzene — aromatic
        feat_bz, _ = compute_features("c1ccccc1")
        assert feat_bz["NumAromaticRings"] == 1
        assert feat_cyclo["NumAromaticRings"] == 0
        assert feat_cyclo["NumAliphaticRings"] >= 1


class TestSmilesFromFile:
    """Tests for smiles_from_file()."""

    def test_none_input(self):
        """None input should return None."""
        assert smiles_from_file(None) is None

    def test_mol_block_parsing(self):
        """A valid .mol block string should parse correctly."""
        mol_block = _make_valid_mol_block()
        if mol_block is None:
            pytest.skip("RDKit could not generate MOL block")

        class FakeFile:
            name = "test.mol"
            def __init__(self, data):
                self._data = data
            def getvalue(self):
                return self._data.encode("utf-8")

        result = smiles_from_file(FakeFile(mol_block))
        assert result is not None
        smiles, formula, mw = result
        assert len(smiles) > 0
        assert isinstance(formula, str)
        assert mw > 0

    def test_unknown_extension_fallback(self):
        """Files with unknown extensions should try all parsers."""
        mol_block = _make_valid_mol_block()
        if mol_block is None:
            pytest.skip("RDKit could not generate MOL block")

        class FakeFile:
            name = "test.unknown"
            def __init__(self, data):
                self._data = data
            def getvalue(self):
                return self._data.encode("utf-8")

        result = smiles_from_file(FakeFile(mol_block))
        # Should parse fine via fallback
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])
