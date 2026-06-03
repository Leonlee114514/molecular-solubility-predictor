"""Shared test fixtures and test molecules for DisSolve."""

import pytest
import sys
import os

# Ensure project root is on sys.path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Well-known SMILES strings used across tests ──

@pytest.fixture
def smiles_ethanol():
    """Ethanol: CH3CH2OH, highly soluble."""
    return "CCO"


@pytest.fixture
def smiles_aspirin():
    """Aspirin (acetylsalicylic acid)."""
    return "CC(=O)OC1=CC=CC=C1C(=O)O"


@pytest.fixture
def smiles_caffeine():
    """Caffeine."""
    return "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


@pytest.fixture
def smiles_water():
    """Water — simplest oxygen-containing molecule."""
    return "O"


@pytest.fixture
def smiles_invalid():
    """Invalid SMILES string."""
    return "CCOOORRRZZZ###"


@pytest.fixture
def features_ethanol():
    """Expected feature dict keys for ethanol computed by features.compute_features."""
    from features import compute_features
    result = compute_features("CCO")
    assert result is not None, "compute_features should work for ethanol"
    return result[0]  # (features_dict, fp_array)


@pytest.fixture
def fp_ethanol():
    """Expected fingerprint array for ethanol."""
    from features import compute_features
    result = compute_features("CCO")
    assert result is not None
    return result[1]
