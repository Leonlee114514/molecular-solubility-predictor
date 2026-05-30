"""
DisSolve - Out-of-Distribution (OOD) Detector.

Automatically flags molecules that fall outside the training data's chemical space,
so users know when predictions may be unreliable.

Detection uses two complementary signals:
  1. Descriptor statistics — how many std deviations each of the 8 features is from training mean
  2. Fingerprint similarity — max Tanimoto similarity to a reference set of training fingerprints

Based on the OOD_TEST_GUIDE.md framework:
  - Category 1: Molecular size OOD (MW far beyond training range) → caught by descriptors
  - Category 2: Inorganic/organometallic OOD → caught by fingerprint similarity
  - Category 3: Structural novelty OOD (cage molecules, fullerenes) → caught by fingerprint similarity
"""

import numpy as np
import joblib
from dataclasses import dataclass, field

DESCRIPTOR_NAMES_CN = {
    "MolWt":              "分子量 (MolWt)",
    "LogP":               "脂水分配系数 (LogP)",
    "NumHDonors":         "氢键供体 (H-Donors)",
    "NumHAcceptors":      "氢键受体 (H-Acceptors)",
    "TPSA":               "极性表面积 (TPSA)",
    "NumRotatableBonds":  "可旋转键 (Rotatable Bonds)",
    "NumAromaticRings":   "芳香环 (Aromatic Rings)",
    "NumAliphaticRings":  "脂肪环 (Aliphatic Rings)",
    "FractionCSP3":       "碳饱和比例 (FractionCSP3)",
    "NumSaturatedRings":  "饱和环数 (Saturated Rings)",
    "HallKierAlpha":      "分子柔性 (Hall-Kier α)",
    "Chi0v":              "连接性指数 χ0v (Chi0)",
    "Chi1v":              "连接性指数 χ1v (Chi1)",
}

DESCRIPTOR_ORDER = [
    "MolWt", "LogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "FractionCSP3", "NumSaturatedRings", "HallKierAlpha", "Chi0v", "Chi1v",
]


@dataclass
class OODResult:
    risk_level: str       # "LOW", "MEDIUM", "HIGH"
    overall_score: float  # 0.0–1.0, higher = more out-of-distribution
    max_tanimoto: float   # max fingerprint similarity to training reference set
    desc_z_scores: dict   # descriptor_name → z_score
    desc_out_of_range: list[str]  # names of descriptors outside training min/max
    desc_extreme: list[str]       # names of descriptors with |z| > 3
    warnings: list[str] = field(default_factory=list)


class OODDetector:
    """Detects whether a molecule is out-of-distribution relative to training data."""

    def __init__(self, desc_stats: dict, fp_samples: np.ndarray):
        self.desc_stats = desc_stats
        self.fp_samples = fp_samples  # (n_ref, 1024) binary array
        self._fp_popcounts = fp_samples.sum(axis=1)  # precompute for speed

    def check(self, features_dict: dict, fp_array: np.ndarray) -> OODResult:
        desc_z_scores = {}
        desc_out_of_range = []
        desc_extreme = []

        for name in DESCRIPTOR_ORDER:
            value = features_dict.get(name, 0)
            stats = self.desc_stats[name]
            std = stats["std"]
            z = float((value - stats["mean"]) / std) if std > 1e-9 else 0.0
            desc_z_scores[name] = z

            if value < stats["min"] or value > stats["max"]:
                desc_out_of_range.append(name)
            if abs(z) > 3.0:
                desc_extreme.append(name)

        max_tanimoto = self._max_tanimoto(fp_array)

        n_out_of_range = len(desc_out_of_range)
        n_extreme = len(desc_extreme)

        # Composite score (0=in-distribution, 1=far OOD)
        desc_score = min(1.0, n_extreme / 3.0 + n_out_of_range * 0.15)
        fp_score = max(0.0, 1.0 - max_tanimoto / 0.30)  # 0 at sim≥0.30, 1 at sim=0
        overall_score = 0.5 * desc_score + 0.5 * fp_score

        # Risk classification (calibrated for ECFP4 1024-bit fingerprints)
        if max_tanimoto < 0.15 or n_extreme >= 3 or n_out_of_range >= 3:
            risk = "HIGH"
        elif max_tanimoto < 0.25 or n_extreme >= 1 or n_out_of_range >= 1:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        warnings = self._build_warnings(
            risk, desc_out_of_range, desc_extreme, max_tanimoto, features_dict
        )

        return OODResult(
            risk_level=risk,
            overall_score=round(overall_score, 3),
            max_tanimoto=round(max_tanimoto, 3),
            desc_z_scores=desc_z_scores,
            desc_out_of_range=desc_out_of_range,
            desc_extreme=desc_extreme,
            warnings=warnings,
        )

    def _max_tanimoto(self, fp: np.ndarray) -> float:
        """Compute max Tanimoto similarity of fp to the reference set."""
        fp_flat = fp.ravel()
        fp_pop = int(fp_flat.sum())
        if fp_pop == 0:
            return 0.0

        ref_pops = self._fp_popcounts  # (n_ref,)
        intersection = np.dot(self.fp_samples, fp_flat)  # (n_ref,) integer
        union = ref_pops + fp_pop - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            tanimotos = np.where(union > 0, intersection / union, 0.0)
        return float(tanimotos.max())

    def _build_warnings(self, risk, out_of_range, extreme, max_sim, features):
        w = []
        if risk == "LOW":
            return w

        if risk == "HIGH":
            w.append(
                "该分子严重偏离训练数据分布，预测值可能极不可靠。"
                "建议仅用作定性参考，不要依赖具体数值。"
            )
        else:
            w.append(
                "该分子部分偏离训练数据分布，预测值可能存在较大误差。"
                "建议谨慎解读结果。"
            )

        for name in extreme:
            cn = DESCRIPTOR_NAMES_CN.get(name, name)
            val = features.get(name, "?")
            mean = self.desc_stats[name]["mean"]
            std = self.desc_stats[name]["std"]
            direction = "高于" if val > mean else "低于"
            w.append(f"{cn} 为 {val:.1f}，{direction}训练均值 ({mean:.1f} ± {std:.1f}) 超过 3 个标准差")

        for name in out_of_range:
            if name in extreme:
                continue  # already covered
            cn = DESCRIPTOR_NAMES_CN.get(name, name)
            val = features.get(name, "?")
            tmin = self.desc_stats[name]["min"]
            tmax = self.desc_stats[name]["max"]
            w.append(f"{cn} 为 {val:.1f}，超出训练集范围 [{tmin:.1f}, {tmax:.1f}]")

        if max_sim < 0.15:
            w.append(
                f"分子指纹最大相似度仅 {max_sim:.2f}，"
                "与训练集中任何分子的结构差异都很大"
            )
        elif max_sim < 0.25:
            w.append(
                f"分子指纹最大相似度仅 {max_sim:.2f}，"
                "与训练集分子的结构相似度偏低"
            )

        return w


def load_ood_detector(path: str = "output_v2/ood_detector.pkl") -> OODDetector:
    """Load a saved OOD detector from disk."""
    data = joblib.load(path)
    return OODDetector(desc_stats=data["desc_stats"], fp_samples=data["fp_samples"])


def save_ood_detector(detector: OODDetector, path: str = "output_v2/ood_detector.pkl"):
    """Save an OOD detector to disk."""
    data = {
        "desc_stats": detector.desc_stats,
        "fp_samples": detector.fp_samples,
    }
    joblib.dump(data, path)
