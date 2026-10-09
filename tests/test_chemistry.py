"""Tests for core.chemistry - the single source of the pKa band thresholds.

Before this module existed, the acid / neutral / base split was hard-coded in
six places across Python and TypeScript with three different threshold pairs
(5/9, 6/8 and 7). These tests pin the boundaries and assert that every caller
routes through the shared implementation instead of re-deriving them.
"""

import pytest

from core.chemistry import (
    ACID_PKA_MAX,
    BASE_PKA_MIN,
    PHYSIOLOGICAL_PH,
    PKA_KINDS,
    STRONG_ACID_PKA_MAX,
    STRONG_BASE_PKA_MIN,
    classify_pka,
    pharma_strength,
)
from features import compute_features


class TestClassifyPka:
    @pytest.mark.parametrize(
        "pka,expected",
        [
            (-2.0, "acid"),
            (0.0, "acid"),
            (5.99, "acid"),
            (ACID_PKA_MAX, "amphoteric"),
            (6.01, "amphoteric"),
            (7.0, "amphoteric"),
            (7.99, "amphoteric"),
            (BASE_PKA_MIN, "amphoteric"),
            (8.01, "base"),
            (12.22, "base"),
        ],
    )
    def test_band_placement(self, pka, expected):
        assert classify_pka(pka) == expected

    def test_band_edges_land_in_the_neutral_band(self):
        """The thresholds themselves are neutral, not acid/base: the split is
        '< ACID_PKA_MAX' and '> BASE_PKA_MIN'."""
        assert classify_pka(ACID_PKA_MAX) == "amphoteric"
        assert classify_pka(BASE_PKA_MIN) == "amphoteric"

    def test_physiological_ph_is_a_separate_constant(self):
        """The ionization reference pH (7.0) and the label thresholds (6.0/8.0)
        answer different questions; merging them is the mistake this module
        exists to prevent."""
        assert PHYSIOLOGICAL_PH == 7.0
        assert PHYSIOLOGICAL_PH not in (ACID_PKA_MAX, BASE_PKA_MIN)

    def test_output_is_always_a_declared_kind(self):
        assert set(PKA_KINDS) == {"acid", "base", "amphoteric"}
        for pka in (-5.0, 3.0, 6.5, 7.0, 8.5, 20.0):
            assert classify_pka(pka) in PKA_KINDS


class TestCallersShareOneSource:
    """Every band classification in the app must come back to classify_pka.

    The prediction service is covered on the other side of the same invariant:
    tests/test_pka_resolve.py::TestBandConsistency asserts that a single pKa
    value resolves to the same band classify_pka gives it.
    """

    PKA_SAMPLES = (-1.0, 3.0, 5.99, 6.0, 7.0, 8.0, 8.01, 11.5)

    def test_backend_api(self):
        from backend.routes import _pka_type_of

        for pka in self.PKA_SAMPLES:
            assert _pka_type_of(pka) == classify_pka(pka)

    def test_admet_delegates_when_no_kind_is_given(self, monkeypatch):
        """analyze_admet asks core.chemistry instead of re-deriving the band,
        and leaves the decision alone when the caller already resolved a kind."""
        import core.analysis as analysis

        seen = []
        real = analysis.classify_pka
        monkeypatch.setattr(
            analysis,
            "classify_pka",
            lambda pka: (seen.append(pka), real(pka))[1],
        )
        features, _ = compute_features("CCO")

        analysis.analyze_admet("CCO", features, pka_val=6.9)
        assert seen == [6.9]

        seen.clear()
        analysis.analyze_admet("CCO", features, pka_val=6.9, pka_kind="acid")
        assert seen == []


class TestPharmaStrength:
    """The strength split inside a band (4.0 / 9.0) is shared too."""

    @pytest.mark.parametrize(
        "pka,kind,expected",
        [
            (2.0, "acid", "strong_acid"),
            (3.99, "acid", "strong_acid"),
            (STRONG_ACID_PKA_MAX, "acid", "mid_acid"),
            (5.0, "acid", "mid_acid"),
            (8.0, "base", "weak_base"),
            (STRONG_BASE_PKA_MIN, "base", "weak_base"),
            (9.01, "base", "strong_base"),
            (12.0, "base", "strong_base"),
            (7.0, "amphoteric", "amphoteric"),
        ],
    )
    def test_strength_placement(self, pka, kind, expected):
        assert pharma_strength(pka, kind) == expected

    def test_any_other_kind_reads_as_amphoteric(self):
        """Both call sites fell through to amphoteric for anything that is not
        acid/base, including None; that behaviour is preserved."""
        for kind in (None, "", "amphoteric", "unknown"):
            assert pharma_strength(3.0, kind) == "amphoteric"

    def test_backend_delegates_to_the_same_decision(self):
        from backend.routes import _pharma_analysis_key

        for kind in ("acid", "base", "amphoteric"):
            for pka in (2.0, 4.0, 7.0, 9.0, 12.0):
                assert _pharma_analysis_key(pka, kind) == pharma_strength(pka, kind)
