"""DisSolve - pKa pair resolution (framework-free, single shared implementation).

Turns the two independent pKa predictions (acidic / basic) into one primary
value plus a chemical kind: "acid" | "base" | "amphoteric" | None.

Why this module exists
----------------------
`model.py` and `services/prediction.py` used to carry two near-identical copies
of this logic, and the numeric-only rule mislabelled every molecule whose two
predictions fall on the SAME side of physiological pH. Aniline is a textbook
weak base, but its predictions (acidic 12.2, basic 4.3) cross neither threshold
(< 7 / > 7), so the old code fell through to "amphoteric" - a value that is
wrong as chemistry and misleading in the UI.

Resolution order
----------------
1. Structural evidence - when a SMILES is supplied, ask which ionizable groups
   the molecule actually contains. This is the primary criterion, and the
   reason the `smiles` argument exists at all.
2. Numeric fallback - no SMILES, or no recognizable ionizable group. Each
   prediction is classified with core.chemistry's band scale (6.0 / 8.0), the
   same rule the rest of the app uses, so a value is never labelled differently
   here and in the API. A pair that lands on the same side of the scale is
   resolved toward the value nearer physiological pH instead of being forced to
   "amphoteric".

The primary value is always the prediction for the resolved kind (acidic for
"acid", basic for "base"); for "amphoteric" it is whichever lies nearer pH 7.
"""

from __future__ import annotations

from core.chemistry import PHYSIOLOGICAL_PH, classify_pka

# Ionizable acidic groups: deprotonate near physiological pH.
# Phenols are deliberately excluded - pKa ~10, essentially unionized at pH 7.4.
_ACIDIC_SMARTS = (
    "[CX3](=O)[OX2H1]",        # carboxylic acid
    "[SX4](=O)(=O)[OX2H1]",    # sulfonic acid
    "[PX4](=O)([OX2H1])",      # phosphonic / phosphoric acid
    "c1nnn[nH]1",              # tetrazole
)

# Ionizable basic groups: protonate near physiological pH.
_BASIC_SMARTS = (
    # Aliphatic sp3 amine - excludes positively charged nitrogen (nitro,
    # quaternary ammonium) plus amide / sulfonamide / imine / nitrile / N-O and
    # N-aryl (the aniline-type amine is handled separately below).
    "[NX3;!$([NX3+]);!$(N-[#6]=[OX1]);!$(N-[#16](=[OX1])=[OX1]);!$(N=*);"
    "!$(N#*);!$(N[N+](=O)[O-]);!$(N[#8]);!$(N-c)]",
    "[NX3;!$([NX3+])][c]",     # aniline-type aromatic amine
    # Pyridine-type aromatic nitrogen. Excludes azoles whose nitrogen is bonded
    # to a second aromatic nitrogen (tetrazole, triazole, tetrazine) - those are
    # strongly acidic and essentially non-basic.
    "[nX2;!$([nX2]~[nX2])]",
)

# The 7.0 reference comes from core.chemistry, where the physiological pH and
# the acid/base band thresholds (6.0/8.0) are deliberately separate constants:
# a prediction counts as "relevant" when it falls on the ionized side of the
# physiological pH, which is a different question from which band label it gets.
_PHYSIOLOGICAL_PH = PHYSIOLOGICAL_PH

_compiled = None


def _compiled_patterns():
    """Lazily compile the SMARTS patterns; unparseable ones are skipped."""
    global _compiled
    if _compiled is None:
        from rdkit import Chem

        acid, base = [], []
        for pattern in _ACIDIC_SMARTS:
            query = Chem.MolFromSmarts(pattern)
            if query is not None:
                acid.append(query)
        for pattern in _BASIC_SMARTS:
            query = Chem.MolFromSmarts(pattern)
            if query is not None:
                base.append(query)
        _compiled = (tuple(acid), tuple(base))
    return _compiled


def ionizable_groups(smiles):
    """Return (has_acidic_group, has_basic_group) for a SMILES string.

    Returns (False, False) when the SMILES cannot be parsed or carries no
    recognizable ionizable group - the caller then falls back to numeric rules.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return False, False
    acid_queries, base_queries = _compiled_patterns()
    has_acid = any(mol.HasSubstructMatch(q) for q in acid_queries)
    has_base = any(mol.HasSubstructMatch(q) for q in base_queries)
    return has_acid, has_base


def _pick_primary(kind, pka_acidic, pka_basic):
    """Choose the primary value for a resolved kind."""
    values = [v for v in (pka_acidic, pka_basic) if v is not None]
    if not values:
        return None
    if kind == "acid" and pka_acidic is not None:
        return float(pka_acidic)
    if kind == "base" and pka_basic is not None:
        return float(pka_basic)
    return float(min(values, key=lambda v: abs(v - _PHYSIOLOGICAL_PH)))


def resolve_pka_pair(pka_acidic, pka_basic, smiles=None):
    """Resolve separate acidic/basic pKa predictions into (primary_pka, kind).

    Args:
        pka_acidic: acidic pKa prediction, or None.
        pka_basic:  basic pKa prediction (of the conjugate acid), or None.
        smiles:     optional SMILES; when given, the molecule's actual
                    ionizable groups decide the kind.

    Returns:
        (primary_pka, kind) where kind is "acid" | "base" | "amphoteric" | None.
        (None, None) when both predictions are missing.
    """
    if pka_acidic is None and pka_basic is None:
        return None, None

    # ── 1. Structural evidence ──
    if smiles:
        has_acid, has_base = ionizable_groups(smiles)
        if has_acid or has_base:
            if has_acid and has_base:
                kind = "amphoteric"
            elif has_acid:
                kind = "acid"
            else:
                kind = "base"
            return _pick_primary(kind, pka_acidic, pka_basic), kind

    # ── 2. Numeric fallback ──
    # The band comes from core.chemistry - the same rule every other caller uses
    # - so one pKa value can never be "acid" here and "amphoteric" in the API.
    # PHYSIOLOGICAL_PH only decides which of two predictions sits nearer the
    # body's pH; it is not a classification threshold.
    acidic_band = classify_pka(pka_acidic) if pka_acidic is not None else None
    basic_band = classify_pka(pka_basic) if pka_basic is not None else None

    if acidic_band == "acid" and basic_band == "base":
        kind = "amphoteric"
    elif acidic_band == "acid":
        kind = "acid"
    elif basic_band == "base":
        kind = "base"
    elif pka_acidic is not None and pka_basic is not None:
        # Both predictions land on the same side of the band scale: prefer the
        # one nearer physiological pH. (Previously unconditionally
        # "amphoteric", which is what mislabelled aniline-type weak bases.)
        kind = "base" if abs(pka_basic - _PHYSIOLOGICAL_PH) < abs(
            pka_acidic - _PHYSIOLOGICAL_PH
        ) else "acid"
    elif acidic_band is not None:
        kind = acidic_band
    else:
        kind = basic_band

    return _pick_primary(kind, pka_acidic, pka_basic), kind
