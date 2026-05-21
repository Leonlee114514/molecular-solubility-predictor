"""
DisSolve - Shared molecular feature computation and chemistry analysis.
Used by app.py, train_model.py, and train_model_v2.py.
"""

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, AllChem, Draw


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


def show_3d_molecule(smiles):
    """Generate interactive 3D ball-and-stick model HTML using py3Dmol."""
    try:
        import py3Dmol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        from rdkit.Geometry import Point3D
        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        cx = sum(conf.GetAtomPosition(i).x for i in range(n_atoms)) / n_atoms
        cy = sum(conf.GetAtomPosition(i).y for i in range(n_atoms)) / n_atoms
        cz = sum(conf.GetAtomPosition(i).z for i in range(n_atoms)) / n_atoms
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D(pos.x - cx, pos.y - cy, pos.z - cz))

        mol = Chem.RemoveHs(mol)
        mb = Chem.MolToMolBlock(mol)
        view = py3Dmol.view(width=480, height=420)
        view.addModel(mb, 'mol')
        view.setStyle({'stick': {'radius': 0.18}, 'sphere': {'scale': 0.3}})
        view.setBackgroundColor('#1a1a2e')
        view.zoomTo()
        html = view._make_html()
        return f'<div style="width:100%;max-width:100%;display:flex;justify-content:center;border-radius:12px;">{html}</div>'
    except Exception:
        return None


def mol_to_dark_image(mol, size=(500, 400)):
    """Render a 2D molecular structure image with dark theme."""
    from io import BytesIO
    from PIL import Image, ImageFilter
    from rdkit.Chem.Draw import rdMolDraw2D

    w, h = size
    BG = np.array([26, 26, 46], dtype=np.uint8)

    draw = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = draw.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 3
    opts.multipleBondOffset = 0.18
    opts.padding = 0.08
    opts.legendFontSize = 22

    opts.updateAtomPalette({
        6:  (0.82, 0.82, 0.92),
        7:  (0.35, 0.65, 1.00),
        8:  (1.00, 0.40, 0.40),
        9:  (0.35, 0.90, 0.55),
        16: (1.00, 0.85, 0.30),
        17: (0.35, 0.90, 0.55),
        15: (1.00, 0.65, 0.20),
    })

    draw.DrawMolecule(mol)
    draw.FinishDrawing()

    png_data = draw.GetDrawingText()
    img = Image.open(BytesIO(png_data)).convert("RGBA")
    arr = np.array(img, dtype=np.float32)

    alpha = arr[:, :, 3:4] / 255.0
    bg_layer = np.full((h, w, 4), np.append(BG, [255]), dtype=np.float32)
    composed = arr * alpha + bg_layer * (1 - alpha)

    glow = img.filter(ImageFilter.GaussianBlur(radius=2))
    glow_arr = np.array(glow, dtype=np.float32) * alpha * 0.2
    composed = np.clip(composed + glow_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(composed, "RGBA")
