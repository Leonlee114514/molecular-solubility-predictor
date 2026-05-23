"""
DisSolve - Shared molecular feature computation.
Used by app.py, train_model.py, and train_model_v2.py.
"""

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, AllChem


def compute_features(smiles_string):
    """Extract 8 molecular descriptors + 1024-bit Morgan fingerprint from a SMILES string.
    Returns (features_dict, fingerprint_array) or None if SMILES is invalid.
    """
    if not smiles_string:
        return None
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    features = {}
    features['MolWt'] = Descriptors.MolWt(mol)
    features['LogP'] = Descriptors.MolLogP(mol)
    features['NumHDonors'] = Descriptors.NumHDonors(mol)
    features['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    features['TPSA'] = Descriptors.TPSA(mol)
    features['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    features['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    features['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)

    rdBase.DisableLog("rdApp.warning")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_array = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)
    rdBase.EnableLog("rdApp.warning")

    return features, fp_array
