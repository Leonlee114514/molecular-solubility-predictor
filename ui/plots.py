"""
DisSolve - Shared plotting and molecular visualization utilities.
Centralises dark theme configuration, CJK font setup, and molecule rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
from ui.components import get_cjk_font

DARK_THEME = {
    "figure.facecolor": "#0a0a0f",
    "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#2a2a3a",
    "axes.labelcolor": "#a0a0b0",
    "xtick.color": "#a0a0b0",
    "ytick.color": "#a0a0b0",
    "text.color": "#f0f0f5",
}


def setup_plt_dark():
    """Apply the DisSolve dark theme and CJK font to matplotlib globals."""
    plt.rcParams.update(DARK_THEME)
    plt.rcParams["axes.unicode_minus"] = False
    cjk = get_cjk_font()
    if cjk:
        plt.rcParams["font.family"] = cjk


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
    BG = np.array([42, 42, 60], dtype=np.uint8)

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

    fg_mask = alpha[:, :, 0] > 0.3
    dark_bond = fg_mask & (composed[:, :, :3].max(axis=2) < 70)
    composed[dark_bond, 0] = np.clip(composed[dark_bond, 0] + 110, 0, 255)
    composed[dark_bond, 1] = np.clip(composed[dark_bond, 1] + 95, 0, 255)
    composed[dark_bond, 2] = np.clip(composed[dark_bond, 2] + 120, 0, 255)

    glow = img.filter(ImageFilter.GaussianBlur(radius=2))
    glow_arr = np.array(glow, dtype=np.float32) * alpha * 0.2
    composed = np.clip(composed + glow_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(composed, "RGBA")


def mol_to_dark_image_with_importance(mol, bond_weights, size=(500, 400)):
    """Render a 2D molecular structure with bonds highlighted by GNN importance.

    Important bonds are drawn in brighter/warmer colors (purple → yellow).
    Less important bonds appear dimmer.

    Args:
        mol: RDKit Mol object.
        bond_weights: dict mapping bond_idx -> importance (0~1).
                      Bonds not in the dict get a subtle default colour.
        size: (width, height) in pixels.

    Returns:
        PIL Image (RGBA) with highlighted bonds.
    """
    from io import BytesIO
    from PIL import Image, ImageFilter
    from rdkit.Chem.Draw import rdMolDraw2D

    w, h = size
    BG = np.array([42, 42, 60], dtype=np.uint8)

    if not bond_weights:
        # Fall back to standard rendering
        return mol_to_dark_image(mol, size)

    # Compute highlight colours per bond using a purple→orange→yellow gradient
    max_w = max(bond_weights.values()) if bond_weights else 1.0
    # RDKit bond highlight colours: dict bondIdx -> (r, g, b) in 0-1
    highlight_colours = {}
    bond_line_widths = {}
    for bidx, wgt in bond_weights.items():
        norm = wgt / max_w if max_w > 0 else 0.0
        # Colour gradient: dim purple (0.3→0.6 intensity) → bright yellow-gold
        r = 0.55 + 0.45 * norm
        g = 0.25 + 0.65 * norm
        b = 0.90 - 0.80 * norm
        highlight_colours[bidx] = (r, g, b)
        # Line width: thicker for more important bonds
        bond_line_widths[bidx] = 3 + 5 * norm

    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
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

    # Use RDKit's highlight bonds API
    highlight_bond_list = list(highlight_colours.keys())
    drawer.DrawMolecule(mol, highlightBonds=highlight_bond_list,
                        highlightBondColors=highlight_colours)
    drawer.FinishDrawing()

    png_data = drawer.GetDrawingText()
    img = Image.open(BytesIO(png_data)).convert("RGBA")
    arr = np.array(img, dtype=np.float32)

    alpha = arr[:, :, 3:4] / 255.0
    bg_layer = np.full((h, w, 4), np.append(BG, [255]), dtype=np.float32)
    composed = arr * alpha + bg_layer * (1 - alpha)

    # Enhance bright bonds with a subtle glow
    glow = img.filter(ImageFilter.GaussianBlur(radius=2))
    glow_arr = np.array(glow, dtype=np.float32) * alpha * 0.2
    composed = np.clip(composed + glow_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(composed, "RGBA")
