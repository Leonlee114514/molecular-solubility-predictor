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


# ========== pKa Chemistry Analysis ==========

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
    factors['Inductive\n(诱导效应)'] = inductive if is_acidic else -inductive * 0.6

    # Resonance effect
    aromatic = Descriptors.NumAromaticRings(mol)
    resonance = min(aromatic * 1.2, 3.0)
    factors['Resonance\n(共轭效应)'] = resonance if is_acidic else resonance * 0.5

    # Intramolecular H-bond
    hbond_pat1 = Chem.MolFromSmarts('[OH]c1ccccc1C(=O)[OH]')
    hbond_pat2 = Chem.MolFromSmarts('[OH]c1ccccc1[OH]')
    has_hbond = False
    if hbond_pat1 and mol.HasSubstructMatch(hbond_pat1):
        has_hbond = True
    if hbond_pat2 and mol.HasSubstructMatch(hbond_pat2):
        has_hbond = True
    hbond_score = 1.5 if has_hbond else 0.0
    factors['Intra-HB\n(分子内氢键)'] = hbond_score if is_acidic else -hbond_score * 0.5

    # Steric hindrance
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    steric = -min(rot_bonds * 0.25, 2.0)
    factors['Steric\n(空间位阻)'] = steric if is_acidic else -steric

    # Hybridization/aromaticity
    sp2_score = 1.0 if aromatic > 0 else -0.5
    factors['Hybridization\n(杂化/芳香性)'] = sp2_score if is_acidic else -sp2_score

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
            "符合 Lipinski 五规则，具有良好口服生物利用度潜力"
            if passed_count >= 4 else
            f"违反 {violations} 条规则，口服吸收可能受限，但仍有成为药物的可能（许多成功药物也违反五规则）"
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
        qed_level = "Attractive (有吸引力)"
        qed_color = "#34d399"
    elif qed >= 0.49:
        qed_level = "Moderate (中等)"
        qed_color = "#fbbf24"
    else:
        qed_level = "Low (偏低)"
        qed_color = "#f87171"

    # SAscore interpretation
    if sa <= 3.0:
        sa_level = "Easy (容易合成)"
        sa_color = "#34d399"
    elif sa <= 6.0:
        sa_level = "Moderate (中等难度)"
        sa_color = "#fbbf24"
    else:
        sa_level = "Difficult (难以合成)"
        sa_color = "#f87171"

    # Fsp³ interpretation
    if fsp3 >= 0.45:
        fsp3_level = "High 3D complexity (高三维复杂度)"
        fsp3_color = "#34d399"
    elif fsp3 >= 0.25:
        fsp3_level = "Moderate (中等)"
        fsp3_color = "#fbbf24"
    else:
        fsp3_level = "Mostly planar (偏平面)"
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
        absorption_factors.append("TPSA < 60 Å²，易于穿过肠道细胞膜")
    elif tpsa < 140:
        absorption_factors.append("TPSA 中等 (60-140 Å²)，吸收尚可")
    else:
        absorption_factors.append("TPSA > 140 Å²，极性表面积较大，可能限制被动跨膜吸收")

    if pka_val is not None:
        if pka_val < 4:
            absorption_factors.append(f"pKa = {pka_val:.1f}（酸性），胃中(pH 1.5)以分子态为主，胃吸收良好")
        elif pka_val > 9:
            absorption_factors.append(f"pKa = {pka_val:.1f}（碱性），胃中离子化，主要在小肠(pH 6.8)吸收")
        else:
            absorption_factors.append(f"pKa = {pka_val:.1f}（近中性），全肠道均有吸收")

    if hbd > 5:
        absorption_factors.append("H-Bond Donors > 5，跨膜需要脱溶剂化，可能降低吸收")
    if hba > 10:
        absorption_factors.append("H-Bond Acceptors > 10，过多的氢键受体降低膜通透性")

    absorption_summary = "；".join(absorption_factors) if absorption_factors else "吸收特性待评估"

    # Distribution
    distribution_factors = []
    vd_estimate = "中等"
    if logp > 3 and mw < 500:
        vd_estimate = "较高（亲脂性强，易分布至组织）"
        distribution_factors.append("分子亲脂性强 (LogP > 3)，倾向于分布到脂肪组织和通过血脑屏障")
    elif logp < 0:
        vd_estimate = "较低（亲水性强，主要留在血浆中）"
        distribution_factors.append("分子亲水性强 (LogP < 0)，主要分布在血浆和细胞外液")
    else:
        distribution_factors.append("LogP 适中，组织分布较均衡")

    if tpsa < 70 and logp > 1:
        distribution_factors.append("低 TPSA + 中等 LogP = 可能通过血脑屏障 (BBB)")
    if mw > 500:
        distribution_factors.append("分子量 > 500，组织渗透能力下降")

    ppb = "中等"
    if logp > 4:
        ppb = "较高（亲脂性强，与血浆蛋白结合率高）"
    elif logp < 0:
        ppb = "较低（亲水性强，游离药物比例高）"

    distribution_summary = "；".join(distribution_factors) if distribution_factors else "分布特性待评估"

    # Metabolism
    metabolism_sites = []
    cyp_notes = []
    if fg.get("aromatic_ring"):
        metabolism_sites.append("芳香环（可能被 CYP450 氧化为环氧化物/酚类）")
        cyp_notes.append("CYP2C9, CYP2E1")
    if fg.get("phenol"):
        metabolism_sites.append("酚羟基（易被 II 相代谢：葡萄糖醛酸化/硫酸化）")
    if fg.get("carboxylic_acid"):
        metabolism_sites.append("羧基（可能与氨基酸结合或形成酰基葡萄糖醛酸）")
    if fg.get("ester"):
        metabolism_sites.append("酯键（被酯酶水解，可能首过效应显著）")
        cyp_notes.append("酯酶（非 CYP）")
    if fg.get("amide"):
        metabolism_sites.append("酰胺键（代谢较稳定，但可被酰胺酶水解）")
    if fg.get("methyl"):
        metabolism_sites.append("甲基（可被 CYP450 氧化为羟甲基 -> 醛 -> 羧酸）")
        cyp_notes.append("CYP3A4")
    if fg.get("primary_amine") or fg.get("secondary_amine"):
        metabolism_sites.append("伯/仲胺基（可能发生 N-脱烷基或 N-氧化）")
        cyp_notes.append("CYP2D6, CYP3A4")
    if fg.get("tertiary_amine"):
        metabolism_sites.append("叔胺基（易发生 N-脱甲基化）")
        cyp_notes.append("CYP3A4, CYP2D6")

    unique_cyps = list(set(cyp_notes))
    metabolism_summary = "；".join(metabolism_sites) if metabolism_sites else "结构较简单，主要代谢途径待实验验证"
    cyp_summary = ", ".join(unique_cyps) if unique_cyps else "待实验确定"

    # Excretion
    excretion_factors = []
    if mw < 350 and logp < 2:
        excretion_factors.append("分子量小 + 亲水性适中 → 倾向于肾脏排泄（肾小球滤过）")
        excretion_route = "主要经肾脏排泄"
    elif mw > 500:
        excretion_factors.append("分子量 > 500 → 倾向于肝胆排泄（胆汁）")
        excretion_route = "倾向于肝胆排泄"
    elif logp > 3:
        excretion_factors.append("LogP > 3 → 在肾小管中易被重吸收，肝胆排泄比例增加")
        excretion_route = "肝肾双途径排泄"
    else:
        excretion_route = "肝肾双途径排泄"

    if tpsa > 100:
        excretion_factors.append("高 TPSA 有利于肾脏排泄（水溶性代谢物）")
    if fg.get("carboxylic_acid") or fg.get("phenol"):
        excretion_factors.append("含羧基/酚羟基 → 易形成 II 相代谢物经肾脏排出")

    excretion_summary = "；".join(excretion_factors) if excretion_factors else "排泄途径待评估"

    # Toxicity
    toxicity_alerts = []
    if fg.get("nitro_aromatic"):
        toxicity_alerts.append(("高", "硝基芳香族 → 可能经 CYP450 还原为有毒的亚硝基/羟胺中间体，有致突变风险"))
    if fg.get("aniline_alert"):
        toxicity_alerts.append(("中", "芳香胺 → 可能经 N-氧化生成致癌性 N-羟基代谢物"))
    if fg.get("epoxide"):
        toxicity_alerts.append(("高", "环氧化物 → 高反应活性，可与 DNA/蛋白质共价结合，可能致癌"))
    if fg.get("hydrazine"):
        toxicity_alerts.append(("高", "肼类结构 → 已知的肝毒性警报结构"))
    if fg.get("alkyl_halide"):
        toxicity_alerts.append(("中", "卤代烷基 → 可能是烷化剂，与谷胱甘肽结合消耗肝脏保护物质"))
    if fg.get("michael_acceptor"):
        toxicity_alerts.append(("中", "Michael 受体 (α,β-不饱和羰基) → 可与亲核基团非特异性结合，可能引起肝毒性/皮肤致敏"))

    if fg.get("halogen") and not fg.get("alkyl_halide"):
        toxicity_alerts.append(("低", "含卤素取代基 → 代谢较稳定，但可能生物积累"))

    if not toxicity_alerts:
        toxicity_alerts.append(("低", "未检出常见结构警报，常规毒性风险较低"))

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
