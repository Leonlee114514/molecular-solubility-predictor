"""Regression tests for pka_resolve - the shared pKa pair resolution.

Two classes of behaviour are pinned here:

  * Structural evidence decides the kind when a SMILES is available. The
    motivating regression is aniline, a textbook weak base whose predictions
    (acidic ~12, basic ~4) cross neither the < 7 nor the > 7 threshold and were
    therefore previously mislabelled "amphoteric".
  * Numeric fallback (no SMILES, or no recognizable ionizable group) keeps the
    legacy thresholds but resolves a same-side pair toward physiological pH.
"""

import pytest

from pka_resolve import ionizable_groups, resolve_pka_pair

# (name, SMILES, expect_acidic_group, expect_basic_group)
GROUP_CASES = [
    ("glycine", "NCC(=O)O", True, True),
    ("aniline", "Nc1ccccc1", False, True),
    ("pyridine", "c1ccncc1", False, True),
    ("triethylamine", "CCN(CC)CC", False, True),
    ("imidazole", "c1cnc[nH]1", False, True),
    ("acetic acid", "CC(=O)O", True, False),
    ("benzoic acid", "O=C(O)c1ccccc1", True, False),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O", True, False),
    ("tetrazole", "c1nnn[nH]1", True, False),
    # Deliberately inert: phenol (pKa ~10, unionized at pH 7.4), amides, nitro,
    # hydrocarbons and ethers must not be reported as ionizable.
    ("phenol", "Oc1ccccc1", False, False),
    ("acetamide", "CC(N)=O", False, False),
    ("urea", "NC(N)=O", False, False),
    ("nitrobenzene", "[O-][N+](=O)c1ccccc1", False, False),
    ("toluene", "Cc1ccccc1", False, False),
    ("diethyl ether", "CCOCC", False, False),
    ("glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", False, False),
    ("indole", "c1ccc2[nH]ccc2c1", False, False),
]


class TestIonizableGroups:
    @pytest.mark.parametrize("name,smiles,expect_acid,expect_base", GROUP_CASES)
    def test_group_detection(self, name, smiles, expect_acid, expect_base):
        has_acid, has_base = ionizable_groups(smiles)
        assert has_acid is expect_acid, f"{name}: acidic group"
        assert has_base is expect_base, f"{name}: basic group"

    def test_invalid_smiles_is_inert(self):
        assert ionizable_groups("not_a_smiles") == (False, False)

    def test_empty_smiles_is_inert(self):
        assert ionizable_groups("") == (False, False)
        assert ionizable_groups(None) == (False, False)


class TestResolveWithStructure:
    """Structural evidence wins over the raw numbers."""

    def test_aniline_is_a_base_not_amphoteric(self):
        """Regression: (12.22, 4.28) is a weak base, not an amphoteric species."""
        primary, kind = resolve_pka_pair(12.22, 4.28, smiles="Nc1ccccc1")
        assert kind == "base"
        assert primary == pytest.approx(4.28)

    def test_pyridine_is_a_base(self):
        primary, kind = resolve_pka_pair(10.75, 4.55, smiles="c1ccncc1")
        assert kind == "base"
        assert primary == pytest.approx(4.55)

    def test_carboxylic_acid_is_an_acid(self):
        """The basic head still emits a value for a pure acid - ignore it."""
        primary, kind = resolve_pka_pair(3.95, 6.42, smiles="CC(=O)O")
        assert kind == "acid"
        assert primary == pytest.approx(3.95)

    def test_amino_acid_is_amphoteric(self):
        primary, kind = resolve_pka_pair(3.79, 8.32, smiles="NCC(=O)O")
        assert kind == "amphoteric"
        assert primary == pytest.approx(8.32)  # nearer physiological pH

    def test_structure_only_fixes_kind_not_values(self):
        """An absurd numeric pair is still labelled by the real structure."""
        primary, kind = resolve_pka_pair(14.0, 1.0, smiles="Nc1ccccc1")
        assert kind == "base"
        assert primary == pytest.approx(1.0)


class TestNumericFallback:
    """No SMILES, or no ionizable group -> legacy thresholds."""

    def test_both_none(self):
        assert resolve_pka_pair(None, None) == (None, None)

    def test_only_basic(self):
        primary, kind = resolve_pka_pair(None, 9.5)
        assert kind == "base"
        assert primary == pytest.approx(9.5)

    def test_only_acidic(self):
        primary, kind = resolve_pka_pair(4.8, None)
        assert kind == "acid"
        assert primary == pytest.approx(4.8)

    def test_straddling_pair_is_amphoteric(self):
        primary, kind = resolve_pka_pair(2.34, 9.60)
        assert kind == "amphoteric"
        assert primary == pytest.approx(9.60)

    def test_acid_dominant(self):
        primary, kind = resolve_pka_pair(4.20, 2.00)
        assert kind == "acid"
        assert primary == pytest.approx(4.20)

    def test_base_dominant(self):
        primary, kind = resolve_pka_pair(14.00, 9.25)
        assert kind == "base"
        assert primary == pytest.approx(9.25)

    def test_same_side_pair_prefers_nearer_ph(self):
        primary, kind = resolve_pka_pair(14.00, 0.60)
        assert kind == "base"
        assert primary == pytest.approx(0.60)

    def test_inert_molecule_falls_back_to_numbers(self):
        """Toluene has no ionizable group, so the numeric rule still applies."""
        primary, kind = resolve_pka_pair(2.34, 9.60, smiles="Cc1ccccc1")
        assert kind == "amphoteric"
        assert primary == pytest.approx(9.60)


class TestBandConsistency:
    """One pKa value must never be classified two different ways.

    The numeric fallback used physiological pH (7.0) while the API's
    classification used the band thresholds (6.0 / 8.0), so pKa 6.5 came back
    "acid" from one entry point and "amphoteric" from the other.
    """

    @pytest.mark.parametrize("pka", [1.0, 5.99, 6.0, 6.5, 7.0, 7.99, 8.0, 8.01, 12.0])
    def test_single_prediction_matches_the_shared_band(self, pka):
        from core.chemistry import classify_pka

        assert resolve_pka_pair(pka, None)[1] == classify_pka(pka)
        assert resolve_pka_pair(None, pka)[1] == classify_pka(pka)
