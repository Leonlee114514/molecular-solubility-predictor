"""
DisSolve - Chemistry analysis functions (pKa, Lipinski, ADMET, drug-likeness).
Split from features.py to keep module responsibilities clean.
"""

import math
import gzip
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from core.i18n import t, get_lang


# ========== pKa Chemistry Analysis ==========

def _join_sentences(parts):
    """Join translated sentence fragments with language-appropriate separator."""
    sep = "; " if get_lang() == "en" else "；"
    return sep.join(parts) if parts else ""

def analyze_pka_chemistry(smiles, pka_val):
    """Analyze chemical factors contributing to pKa value.
    Returns dict of factor_name -> contribution_score.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    is_acidic = pka_val < 7
    factors = {}

    # Inductive effect
    en_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [7, 8, 9, 17, 35])
    inductive = min(en_atoms * 0.4, 3.0)
    factors[t('analysis.pka.inductive')] = inductive if is_acidic else -inductive * 0.6

    # Resonance effect
    aromatic = Descriptors.NumAromaticRings(mol)
    resonance = min(aromatic * 1.2, 3.0)
    factors[t('analysis.pka.resonance')] = resonance if is_acidic else resonance * 0.5

    # Intramolecular H-bond
    hbond_pat1 = Chem.MolFromSmarts('[OH]c1ccccc1C(=O)[OH]')
    hbond_pat2 = Chem.MolFromSmarts('[OH]c1ccccc1[OH]')
    has_hbond = False
    if hbond_pat1 and mol.HasSubstructMatch(hbond_pat1):
        has_hbond = True
    if hbond_pat2 and mol.HasSubstructMatch(hbond_pat2):
        has_hbond = True
    hbond_score = 1.5 if has_hbond else 0.0
    factors[t('analysis.pka.intra_hb')] = hbond_score if is_acidic else -hbond_score * 0.5

    # Steric hindrance
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    steric = -min(rot_bonds * 0.25, 2.0)
    factors[t('analysis.pka.steric')] = steric if is_acidic else -steric

    # Hybridization/aromaticity
    sp2_score = 1.0 if aromatic > 0 else -0.5
    factors[t('analysis.pka.hybridization')] = sp2_score if is_acidic else -sp2_score

    return factors


# ========== Lipinski Rule of Five ==========

def analyze_lipinski(features):
    """Evaluate Lipinski's Rule of Five for oral drug-likeness.

    Returns dict with per-rule pass/fail and overall score (0-5).
    Lipinski rules:
      1. Molecular Weight ≤ 500 Da
      2. LogP ≤ 5
      3. H-Bond Donors ≤ 5
      4. H-Bond Acceptors ≤ 10
      5. Rotatable Bonds ≤ 10 (extended rule)
    """
    rules = [
        ("Molecular Weight ≤ 500", "MolWt", features["MolWt"] <= 500, f'{features["MolWt"]:.0f} Da'),
        ("LogP ≤ 5", "LogP", features["LogP"] <= 5, f'{features["LogP"]:.2f}'),
        ("H-Bond Donors ≤ 5", "NumHDonors", features["NumHDonors"] <= 5, str(features["NumHDonors"])),
        ("H-Bond Acceptors ≤ 10", "NumHAcceptors", features["NumHAcceptors"] <= 10, str(features["NumHAcceptors"])),
        ("Rotatable Bonds ≤ 10", "NumRotatableBonds", features["NumRotatableBonds"] <= 10, str(features["NumRotatableBonds"])),
    ]
    violations = sum(1 for _, _, passed, _ in rules if not passed)
    passed_count = 5 - violations
    return {
        "rules": rules,
        "passed": passed_count,
        "violations": violations,
        "is_druglike": passed_count >= 4,
        "interpretation": (
            t("analysis.lipinski.good")
            if passed_count >= 4 else
            t("analysis.lipinski.violated", n=violations)
        ),
    }


# ========== Inline SAscore (for Streamlit Cloud compatibility) ==========

_fscores = None
_mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def _sa_load_fragment_scores():
    global _fscores
    data = pickle.load(gzip.open("data/sa_fpscores.pkl.gz"))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def _sa_calculate_score(mol):
    if not mol.GetNumAtoms():
        return None
    if _fscores is None:
        _sa_load_fragment_scores()

    sfp = _mfpgen.GetSparseCountFingerprint(mol)
    score1 = 0.
    nf = 0
    nze = sfp.GetNonzeroElements()
    for fid, count in nze.items():
        nf += count
        score1 += _fscores.get(fid, -4) * count
    score1 /= nf

    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    nBridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    score3 = 0.
    numBits = len(nze)
    if nAtoms > numBits:
        score3 = math.log(float(nAtoms) / numBits) * .5

    sascore = score1 + score2 + score3

    min_ = -4.0
    max_ = 2.5
    sascore = 11. - (sascore - min_ + 1) / (max_ - min_) * 9.
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
    return sascore


# ========== Drug-likeness Metrics ==========

def analyze_druglikeness(smiles):
    """Compute QED, SAscore, and Fsp³ drug-likeness metrics.

    QED (Quantitative Estimate of Drug-likeness): 0-1, higher = more drug-like.
    SAscore (Synthetic Accessibility): typically 1-10, lower = easier to synthesize.
    Fsp³ (Fraction of sp3 carbons): 0-1, higher = more 3D character.
    """
    from rdkit.Chem import QED

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    qed = QED.default(mol)
    sa = _sa_calculate_score(mol)
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    n_carbons = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    n_sp3 = int(round(fsp3 * n_carbons)) if n_carbons > 0 else 0

    # QED interpretation
    if qed >= 0.67:
        qed_level = t("analysis.qed.attractive")
        qed_color = "#34d399"
    elif qed >= 0.49:
        qed_level = t("analysis.qed.moderate")
        qed_color = "#fbbf24"
    else:
        qed_level = t("analysis.qed.low")
        qed_color = "#f87171"

    # SAscore interpretation
    if sa <= 3.0:
        sa_level = t("analysis.sa.easy")
        sa_color = "#34d399"
    elif sa <= 6.0:
        sa_level = t("analysis.sa.moderate")
        sa_color = "#fbbf24"
    else:
        sa_level = t("analysis.sa.hard")
        sa_color = "#f87171"

    # Fsp³ interpretation
    if fsp3 >= 0.45:
        fsp3_level = t("analysis.fsp3.high")
        fsp3_color = "#34d399"
    elif fsp3 >= 0.25:
        fsp3_level = t("analysis.fsp3.moderate")
        fsp3_color = "#fbbf24"
    else:
        fsp3_level = t("analysis.fsp3.planar")
        fsp3_color = "#a78bfa"

    return {
        "qed": qed,
        "qed_color": qed_color,
        "qed_level": qed_level,
        "sascore": sa,
        "sa_color": sa_color,
        "sa_level": sa_level,
        "fsp3": fsp3,
        "fsp3_color": fsp3_color,
        "fsp3_level": fsp3_level,
        "n_carbons": n_carbons,
        "n_sp3": n_sp3,
    }


# ========== Functional Group Detection ==========

def detect_functional_groups(smiles):
    """Detect common functional groups and structural motifs relevant to ADME/Tox.
    Returns a dict of group_name -> bool.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    patterns = {
        "aromatic_ring": "c1ccccc1",
        "phenol": "c1ccccc1O",
        "aniline": "c1ccccc1N",
        "ester": "[CX3](=O)[OX2H0]",
        "amide": "[CX3](=O)[NX3]",
        "carboxylic_acid": "[CX3](=O)[OX2H1]",
        "primary_amine": "[NX3H2]",
        "secondary_amine": "[NX3H1][CX4]",
        "tertiary_amine": "[NX3]([CX4])([CX4])[CX4]",
        "methyl": "[CX4H3]",
        "ether": "[OX2H0]([CX4])[CX4]",
        "ketone": "[CX3](=O)[CX4]",
        "aldehyde": "[CX3H1](=O)",
        "nitro": "[NX3](=O)=O",
        "nitro_charged": "[N+X3](=O)[O-]",
        "sulfonamide": "[SX4](=O)(=O)[NX3]",
        "thiol": "[SX2H]",
        "halogen": "[F,Cl,Br,I]",
        "nitrile": "[CX2]#[NX1]",
        "nitro_aromatic": "c1ccccc1[NX3](=O)=O",
        "aniline_alert": "[cR][NH2]",
        "epoxide": "C1OC1",
        "hydrazine": "[NX3][NX3]",
        "alkyl_halide": "[CX4][F,Cl,Br,I]",
        "michael_acceptor": "[CX3]=[CX3][CX3]=O",
    }

    detected = {}
    for name, smarts in patterns.items():
        try:
            pat = Chem.MolFromSmarts(smarts)
            detected[name] = mol.HasSubstructMatch(pat) if pat else False
        except Exception:
            detected[name] = False

    return detected


# ========== ADME/Tox Analysis ==========

def analyze_admet(smiles, features, pka_val=None):
    """Analyze ADME/Tox properties based on molecular descriptors and functional groups.
    Returns a dict with A, D, M, E, T analysis sections.
    """
    fg = detect_functional_groups(smiles)
    mw = features["MolWt"]
    logp = features["LogP"]
    tpsa = features["TPSA"]
    hbd = features["NumHDonors"]
    hba = features["NumHAcceptors"]
    rot_bonds = features["NumRotatableBonds"]

    # Absorption
    absorption_factors = []
    if tpsa < 60:
        absorption_factors.append(t("analysis.admet.absorption.tpsa_low"))
    elif tpsa < 140:
        absorption_factors.append(t("analysis.admet.absorption.tpsa_mid"))
    else:
        absorption_factors.append(t("analysis.admet.absorption.tpsa_high"))

    if pka_val is not None:
        if pka_val < 6:
            absorption_factors.append(t("analysis.admet.absorption.pka_acid", val=pka_val))
        elif pka_val > 8:
            absorption_factors.append(t("analysis.admet.absorption.pka_base", val=pka_val))
        else:
            absorption_factors.append(t("analysis.admet.absorption.pka_neutral", val=pka_val))

    if hbd > 5:
        absorption_factors.append(t("analysis.admet.absorption.hbd_high"))
    if hba > 10:
        absorption_factors.append(t("analysis.admet.absorption.hba_high"))

    absorption_summary = _join_sentences(absorption_factors) if absorption_factors else t("analysis.admet.absorption.fallback")

    # Distribution
    distribution_factors = []
    vd_estimate = t("analysis.druglikeness.vd")
    if logp > 3 and mw < 500:
        vd_estimate = t("analysis.admet.distribution.vd_high")
        distribution_factors.append(t("analysis.admet.distribution.lipophilic"))
    elif logp < 0:
        vd_estimate = t("analysis.admet.distribution.vd_low")
        distribution_factors.append(t("analysis.admet.distribution.hydrophilic"))
    else:
        distribution_factors.append(t("analysis.admet.distribution.moderate"))

    if tpsa < 70 and logp > 1:
        distribution_factors.append(t("analysis.admet.distribution.bbb"))
    if mw > 500:
        distribution_factors.append(t("analysis.admet.distribution.high_mw"))

    ppb = t("analysis.druglikeness.ppb")
    if logp > 4:
        ppb = t("analysis.admet.distribution.ppb_high")
    elif logp < 0:
        ppb = t("analysis.admet.distribution.ppb_low")

    distribution_summary = _join_sentences(distribution_factors) if distribution_factors else t("analysis.admet.distribution.fallback")

    # Metabolism
    metabolism_sites = []
    cyp_notes = []
    if fg.get("aromatic_ring"):
        metabolism_sites.append(t("analysis.admet.metabolism.aromatic"))
        cyp_notes.append("CYP2C9, CYP2E1")
    if fg.get("phenol"):
        metabolism_sites.append(t("analysis.admet.metabolism.phenol"))
    if fg.get("carboxylic_acid"):
        metabolism_sites.append(t("analysis.admet.metabolism.carboxylic"))
    if fg.get("ester"):
        metabolism_sites.append(t("analysis.admet.metabolism.ester"))
        cyp_notes.append("Esterases (non-CYP)")
    if fg.get("amide"):
        metabolism_sites.append(t("analysis.admet.metabolism.amide"))
    if fg.get("methyl"):
        metabolism_sites.append(t("analysis.admet.metabolism.methyl"))
        cyp_notes.append("CYP3A4")
    if fg.get("primary_amine") or fg.get("secondary_amine"):
        metabolism_sites.append(t("analysis.admet.metabolism.amine_sec"))
        cyp_notes.append("CYP2D6, CYP3A4")
    if fg.get("tertiary_amine"):
        metabolism_sites.append(t("analysis.admet.metabolism.amine_tert"))
        cyp_notes.append("CYP3A4, CYP2D6")

    unique_cyps = list(set(cyp_notes))
    metabolism_summary = _join_sentences(metabolism_sites) if metabolism_sites else t("analysis.admet.metabolism.fallback")
    cyp_summary = ", ".join(unique_cyps) if unique_cyps else t("analysis.admet.metabolism.cyp_fallback")

    # Excretion
    excretion_factors = []
    if mw < 350 and logp < 2:
        excretion_factors.append(t("analysis.admet.excretion.renal"))
        excretion_route = t("analysis.admet.excretion.route_renal")
    elif mw > 500:
        excretion_factors.append(t("analysis.admet.excretion.biliary"))
        excretion_route = t("analysis.admet.excretion.route_biliary")
    elif logp > 3:
        excretion_factors.append(t("analysis.admet.excretion.hepato"))
        excretion_route = t("analysis.admet.excretion.route_dual")
    else:
        excretion_route = t("analysis.admet.excretion.route_dual")

    if tpsa > 100:
        excretion_factors.append(t("analysis.admet.excretion.tpsa"))
    if fg.get("carboxylic_acid") or fg.get("phenol"):
        excretion_factors.append(t("analysis.admet.excretion.conjugate"))

    excretion_summary = _join_sentences(excretion_factors) if excretion_factors else t("analysis.admet.excretion.fallback")

    # Toxicity
    toxicity_alerts = []
    if fg.get("nitro_aromatic"):
        toxicity_alerts.append((t("analysis.admet.toxicity.level_high"), t("analysis.admet.toxicity.nitro")))
    if fg.get("aniline_alert"):
        toxicity_alerts.append((t("analysis.admet.toxicity.level_medium"), t("analysis.admet.toxicity.aniline")))
    if fg.get("epoxide"):
        toxicity_alerts.append((t("analysis.admet.toxicity.level_high"), t("analysis.admet.toxicity.epoxide")))
    if fg.get("hydrazine"):
        toxicity_alerts.append((t("analysis.admet.toxicity.level_high"), t("analysis.admet.toxicity.hydrazine")))
    if fg.get("alkyl_halide"):
        toxicity_alerts.append((t("analysis.admet.toxicity.level_medium"), t("analysis.admet.toxicity.alkyl_halide")))
    if fg.get("michael_acceptor"):
        toxicity_alerts.append((t("analysis.admet.toxicity.level_medium"), t("analysis.admet.toxicity.michael")))

    if fg.get("halogen") and not fg.get("alkyl_halide"):
        toxicity_alerts.append((t("analysis.admet.toxicity.level_low"), t("analysis.admet.toxicity.halogen")))

    if not toxicity_alerts:
        toxicity_alerts.append((t("analysis.admet.toxicity.level_low"), t("analysis.admet.toxicity.none")))

    return {
        "absorption": absorption_summary,
        "distribution": {
            "summary": distribution_summary,
            "vd_estimate": vd_estimate,
            "ppb": ppb,
        },
        "metabolism": {
            "summary": metabolism_summary,
            "cyp_enzymes": cyp_summary,
        },
        "excretion": {
            "summary": excretion_summary,
            "route": excretion_route,
        },
        "toxicity": toxicity_alerts,
    }
