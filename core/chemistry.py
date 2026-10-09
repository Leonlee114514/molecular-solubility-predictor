"""DisSolve - shared chemistry constants and classification.

Two *different* questions about a pKa value were answered with hard-coded
numbers in six places, in two languages, with three diverging threshold sets
(5/9, 6/8 and 7 all appeared). This module is now the single place both
questions are answered:

* **pKa band** - where a value sits on the acid / neutral / base scale used for
  labels and colours. Thresholds: 6.0 and 8.0.
* **Physiological pH** - the reference point for "is this group ionized in the
  body", used to pick the primary value out of an acidic/basic prediction pair
  and to drive the Henderson-Hasselbalch direction. That number is 7.0.

The two are deliberately different numbers answering different questions, so
they carry different names instead of one shared magic value.
"""

from __future__ import annotations

# Reference pH of the body: ionization relevance and direction hinge on this.
PHYSIOLOGICAL_PH = 7.0

# Band thresholds for the acid / neutral / base label scale.
ACID_PKA_MAX = 6.0
BASE_PKA_MIN = 8.0

PKA_KINDS = ("acid", "base", "amphoteric")


def classify_pka(pka: float) -> str:
    """Place a pKa value on the band scale: "acid" | "amphoteric" | "base"."""
    if pka < ACID_PKA_MAX:
        return "acid"
    if pka > BASE_PKA_MIN:
        return "base"
    return "amphoteric"


# Strength thresholds *within* a band: an acid counts as strong below this pKa,
# a base as strong above it. Shared by backend/routes.py (/analysis) and
# ui/results.py (pharmacology tab), which used to carry separate copies.
STRONG_ACID_PKA_MAX = 4.0
STRONG_BASE_PKA_MIN = 9.0


def pharma_strength(pka: float, kind: str) -> str:
    """Strength label for a pKa inside its band.

    Returns "strong_acid" | "mid_acid" | "strong_base" | "weak_base" |
    "amphoteric" - the keys both UIs branch on to choose their wording. Any
    kind other than "acid"/"base" (including None) reads as amphoteric, which
    is what both call sites did before.
    """
    if kind == "acid":
        return "strong_acid" if pka < STRONG_ACID_PKA_MAX else "mid_acid"
    if kind == "base":
        return "strong_base" if pka > STRONG_BASE_PKA_MIN else "weak_base"
    return "amphoteric"
