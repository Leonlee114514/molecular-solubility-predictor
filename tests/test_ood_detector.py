"""Tests for ood_detector.py — OOD detection logic."""

import numpy as np
import pytest
from ood_detector import (
    OODDetector,
    OODResult,
    load_ood_detector,
    DESCRIPTOR_ORDER,
    DESCRIPTOR_NAMES_CN,
)


@pytest.fixture
def sample_detector():
    """Create an OODDetector with synthetic training statistics that match in-distribution features."""
    rng = np.random.RandomState(42)
    desc_stats = {}
    for name in DESCRIPTOR_ORDER:
        # Use stats that make in_distribution_features look in-distribution
        if name == "MolWt":
            desc_stats[name] = {"mean": 200.0, "std": 80.0, "min": 18.0, "max": 500.0}
        elif name == "LogP":
            desc_stats[name] = {"mean": 2.0, "std": 2.0, "min": -3.0, "max": 6.0}
        elif name == "TPSA":
            desc_stats[name] = {"mean": 60.0, "std": 40.0, "min": 0.0, "max": 200.0}
        elif name in ("Chi0v", "Chi1v"):
            desc_stats[name] = {"mean": 10.0, "std": 8.0, "min": 0.0, "max": 50.0}
        elif name in ("NumRotatableBonds",):
            desc_stats[name] = {"mean": 5.0, "std": 4.0, "min": 0.0, "max": 20.0}
        elif name in ("NumHDonors", "NumHAcceptors"):
            desc_stats[name] = {"mean": 3.0, "std": 3.0, "min": 0.0, "max": 15.0}
        elif "Ring" in name or "ring" in name:
            desc_stats[name] = {"mean": 2.0, "std": 2.0, "min": 0.0, "max": 10.0}
        elif name == "FractionCSP3":
            desc_stats[name] = {"mean": 0.5, "std": 0.3, "min": 0.0, "max": 1.0}
        elif name == "HallKierAlpha":
            desc_stats[name] = {"mean": -1.0, "std": 1.5, "min": -5.0, "max": 5.0}
        else:
            desc_stats[name] = {"mean": 2.0, "std": 1.0, "min": -5.0, "max": 10.0}

    # Generate 50 random reference fingerprints
    fp_samples = rng.randint(0, 2, size=(50, 1024)).astype(np.int8)
    return OODDetector(desc_stats=desc_stats, fp_samples=fp_samples)


@pytest.fixture
def in_distribution_features():
    """Features that should be well within training distribution."""
    return {
        "MolWt": 180.0,
        "LogP": 2.0,
        "NumHDonors": 1,
        "NumHAcceptors": 3,
        "TPSA": 60.0,
        "NumRotatableBonds": 3,
        "NumAromaticRings": 1,
        "NumAliphaticRings": 0,
        "FractionCSP3": 0.25,
        "NumSaturatedRings": 0,
        "HallKierAlpha": -1.0,
        "Chi0v": 10.0,
        "Chi1v": 5.0,
    }


@pytest.fixture
def out_of_distribution_features():
    """Features far outside training range."""
    return {
        "MolWt": 2000.0,   # way above training max
        "LogP": 20.0,       # way above training max
        "NumHDonors": 50,   # way above training max
        "NumHAcceptors": 100,
        "TPSA": 500.0,
        "NumRotatableBonds": 100,
        "NumAromaticRings": 20,
        "NumAliphaticRings": 30,
        "FractionCSP3": 1.0,
        "NumSaturatedRings": 25,
        "HallKierAlpha": 20.0,
        "Chi0v": 200.0,
        "Chi1v": 150.0,
    }


class TestOODDetectorInit:
    """Tests for OODDetector initialization."""

    def test_initialization(self):
        """OODDetector should store stats and compute popcounts."""
        desc_stats = {"MolWt": {"mean": 200.0, "std": 50.0, "min": 20.0, "max": 500.0}}
        fp = np.zeros((10, 1024), dtype=np.int8)
        detector = OODDetector(desc_stats=desc_stats, fp_samples=fp)
        assert detector.desc_stats is desc_stats
        assert detector.fp_samples is fp
        assert detector._fp_popcounts.shape == (10,)

    def test_popcounts_computed_correctly(self):
        """_fp_popcounts should be the sum of bits in each fingerprint."""
        fp = np.zeros((3, 1024), dtype=np.int8)
        fp[0, :10] = 1   # popcount = 10
        fp[1, :20] = 1   # popcount = 20
        fp[2, :] = 0     # popcount = 0
        detector = OODDetector(desc_stats={}, fp_samples=fp)
        np.testing.assert_array_equal(detector._fp_popcounts, [10, 20, 0])


class TestOODDetectorCheck:
    """Tests for OODDetector.check()."""

    @pytest.fixture
    def fp_matching_reference(self, sample_detector):
        """Return a fingerprint that matches one of the reference fingerprints."""
        return sample_detector.fp_samples[0].copy()

    def test_returns_ood_result(self, sample_detector, in_distribution_features, fp_matching_reference):
        """Check should return an OODResult."""
        result = sample_detector.check(in_distribution_features, fp_matching_reference)
        assert isinstance(result, OODResult)

    def test_ood_result_has_correct_keys(self, sample_detector, in_distribution_features, fp_matching_reference):
        """OODResult should have all expected attributes."""
        result = sample_detector.check(in_distribution_features, fp_matching_reference)
        assert hasattr(result, "risk_level")
        assert hasattr(result, "overall_score")
        assert hasattr(result, "max_tanimoto")
        assert hasattr(result, "desc_z_scores")
        assert hasattr(result, "desc_out_of_range")
        assert hasattr(result, "desc_extreme")
        assert hasattr(result, "warnings")

    def test_in_distribution_low_risk(self, sample_detector, in_distribution_features, fp_matching_reference):
        """In-distribution features should give LOW risk."""
        result = sample_detector.check(in_distribution_features, fp_matching_reference)
        assert result.risk_level == "LOW"

    def test_out_of_distribution_high_risk(self, sample_detector, out_of_distribution_features):
        """Far OOD features should give HIGH risk."""
        fp = np.zeros(1024, dtype=np.int8)
        result = sample_detector.check(out_of_distribution_features, fp)
        assert result.risk_level == "HIGH"

    def test_out_of_range_detected(self, sample_detector, out_of_distribution_features):
        """Extreme values should appear in desc_out_of_range."""
        fp = np.zeros(1024, dtype=np.int8)
        result = sample_detector.check(out_of_distribution_features, fp)
        assert len(result.desc_out_of_range) > 0
        assert "MolWt" in result.desc_out_of_range

    def test_extreme_values_detected(self, sample_detector, out_of_distribution_features):
        """|z| > 3 descriptors should appear in desc_extreme."""
        fp = np.zeros(1024, dtype=np.int8)
        result = sample_detector.check(out_of_distribution_features, fp)
        assert len(result.desc_extreme) > 0

    def test_z_scores_are_computed(self, sample_detector, in_distribution_features, fp_matching_reference):
        """desc_z_scores should contain entries for all descriptors."""
        result = sample_detector.check(in_distribution_features, fp_matching_reference)
        assert len(result.desc_z_scores) == len(DESCRIPTOR_ORDER)
        for name in DESCRIPTOR_ORDER:
            assert name in result.desc_z_scores

    def test_tanimoto_similarity_zero_fp(self, sample_detector, in_distribution_features):
        """All-zero fingerprint should give max_tanimoto=0.0."""
        fp = np.zeros(1024, dtype=np.int8)
        result = sample_detector.check(in_distribution_features, fp)
        assert result.max_tanimoto == 0.0

    def test_overall_score_range(self, sample_detector, in_distribution_features, fp_matching_reference):
        """overall_score should be between 0.0 and 1.0."""
        result = sample_detector.check(in_distribution_features, fp_matching_reference)
        assert 0.0 <= result.overall_score <= 1.0

    def test_ood_overall_score_higher(self, sample_detector, in_distribution_features, out_of_distribution_features):
        """OOD should have higher overall_score than in-distribution."""
        fp = np.zeros(1024, dtype=np.int8)
        low_result = sample_detector.check(in_distribution_features, fp)
        high_result = sample_detector.check(out_of_distribution_features, fp)
        assert high_result.overall_score > low_result.overall_score

    def test_high_risk_warnings(self, sample_detector, out_of_distribution_features):
        """HIGH risk should produce warnings."""
        fp = np.zeros(1024, dtype=np.int8)
        result = sample_detector.check(out_of_distribution_features, fp)
        assert len(result.warnings) > 0

    def test_low_risk_no_warnings(self, sample_detector, in_distribution_features, fp_matching_reference):
        """LOW risk should have no warnings."""
        result = sample_detector.check(in_distribution_features, fp_matching_reference)
        assert len(result.warnings) == 0


class TestMaxTanimoto:
    """Tests for the internal _max_tanimoto method."""

    def test_identical_fingerprint(self):
        """An fp identical to a reference should give tanimoto=1.0."""
        fp = np.zeros(1024, dtype=np.int8)
        fp[:10] = 1
        ref = fp.copy().reshape(1, 1024)
        detector = OODDetector(desc_stats={}, fp_samples=ref)
        result = detector._max_tanimoto(fp)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_fingerprint(self):
        """An fp with no overlapping bits should give tanimoto=0.0."""
        ref = np.zeros((1, 1024), dtype=np.int8)
        ref[0, :10] = 1
        fp = np.zeros(1024, dtype=np.int8)
        fp[10:20] = 1  # non-overlapping
        detector = OODDetector(desc_stats={}, fp_samples=ref)
        result = detector._max_tanimoto(fp)
        assert result == 0.0

    def test_zero_fp(self):
        """An all-zero fingerprint should return 0.0."""
        ref = np.zeros((1, 1024), dtype=np.int8)
        ref[0, :10] = 1
        fp = np.zeros(1024, dtype=np.int8)
        detector = OODDetector(desc_stats={}, fp_samples=ref)
        result = detector._max_tanimoto(fp)
        assert result == 0.0

    def test_zero_ref_fp(self):
        """An all-zero reference fingerprint should still compute safely."""
        ref = np.zeros((1, 1024), dtype=np.int8)
        fp = np.zeros(1024, dtype=np.int8)
        fp[:10] = 1
        detector = OODDetector(desc_stats={}, fp_samples=ref)
        result = detector._max_tanimoto(fp)
        assert result == 0.0

    def test_multiple_references(self):
        """Should find max over all reference fingerprints."""
        # Use simple deterministic fingerprints to avoid int8 dot-product overflow
        refs = np.zeros((3, 1024), dtype=np.int8)
        refs[0, :50] = 1
        refs[1, 50:100] = 1
        refs[2, 100:200] = 1
        fp = refs[1].copy()
        detector = OODDetector(desc_stats={}, fp_samples=refs)
        result = detector._max_tanimoto(fp)
        assert result == pytest.approx(1.0, abs=0.01)


class TestDESCRIPTOR_NAMES_CN:
    """Tests for Chinese descriptor names mapping."""

    def test_all_descriptors_have_names(self):
        """All descriptors in DESCRIPTOR_ORDER should have Chinese names."""
        for name in DESCRIPTOR_ORDER:
            assert name in DESCRIPTOR_NAMES_CN, f"Missing Chinese name for {name}"

    def test_names_are_strings(self):
        """All Chinese names should be strings."""
        for cn_name in DESCRIPTOR_NAMES_CN.values():
            assert isinstance(cn_name, str)
            assert len(cn_name) > 0


class TestLoadOODDetector:
    """Tests for load_ood_detector()."""

    def test_file_not_found(self):
        """Loading non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_ood_detector("/nonexistent/path.pkl")


if __name__ == "__main__":
    pytest.main([__file__])
