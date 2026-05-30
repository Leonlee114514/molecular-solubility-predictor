"""
DisSolve - Shared molecular feature computation.
Used by app.py, train_model.py, and train_model_v2.py.
"""

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, AllChem


def compute_features(smiles_string):
    """Extract 13 molecular descriptors + 1024-bit Morgan fingerprint from a SMILES string.
    Returns (features_dict, fingerprint_array) or None if SMILES is invalid.
    """
    if not smiles_string:
        return None
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    features = {}
    # ── Core 8 descriptors ──
    features['MolWt'] = Descriptors.MolWt(mol)
    features['LogP'] = Descriptors.MolLogP(mol)
    features['NumHDonors'] = Descriptors.NumHDonors(mol)
    features['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    features['TPSA'] = Descriptors.TPSA(mol)
    features['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    features['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    features['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)

    # ── Extended 5 descriptors ──
    features['FractionCSP3'] = Descriptors.FractionCSP3(mol)       # carbon saturation ratio
    features['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)  # saturated ring count
    features['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)     # molecular flexibility
    features['Chi0v'] = Descriptors.Chi0v(mol)                     # connectivity (order 0)
    features['Chi1v'] = Descriptors.Chi1v(mol)                     # connectivity (order 1)

    rdBase.DisableLog("rdApp.warning")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_array = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)
    rdBase.EnableLog("rdApp.warning")

    return features, fp_array


def smiles_from_file(uploaded_file):
    """Parse an uploaded molecular file (.mol, .sdf, .mol2, .pdb, .xyz) and return
    (canonical_smiles, mol_formula, mol_weight) or None if parsing fails.

    Returns
    -------
    tuple or None
        (smiles: str, formula: str, mw: float) on success, or None on failure.
    """
    from rdkit.Chem import rdMolDescriptors

    if uploaded_file is None:
        return None

    raw_bytes = uploaded_file.getvalue()
    name = (uploaded_file.name or "").lower()

    mol = None
    if name.endswith(".mol") or name.endswith(".sdf"):
        mol = Chem.MolFromMolBlock(raw_bytes.decode("utf-8", errors="replace"))
    elif name.endswith(".mol2"):
        mol = Chem.MolFromMol2Block(raw_bytes.decode("utf-8", errors="replace"))
    elif name.endswith(".pdb"):
        mol = Chem.MolFromPDBBlock(raw_bytes.decode("utf-8", errors="replace"))
    elif name.endswith(".xyz"):
        mol = Chem.MolFromXYZBlock(raw_bytes.decode("utf-8", errors="replace"))
    else:
        # Try each parser in order
        text = raw_bytes.decode("utf-8", errors="replace")
        for parser in (Chem.MolFromMolBlock, Chem.MolFromPDBBlock, Chem.MolFromMol2Block, Chem.MolFromXYZBlock):
            mol = parser(text)
            if mol is not None:
                break

    if mol is None:
        return None

    smiles = Chem.MolToSmiles(mol)
    if not smiles:
        return None

    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.MolWt(mol)
    return smiles, formula, mw
