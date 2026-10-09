"""DisSolve - 2D molecular structure rendering.

Used by the FastAPI backend (/api/mol/2d). Two palettes share one code path:
"paper" is what the React UI requests, "dark" is the palette of the Streamlit UI
that was retired in favour of the React port - it is kept as a rendering
baseline (and for a future dark mode), not because anything calls it today.
"""

import numpy as np


# 2D structure rendering themes. "dark" is the Streamlit app's palette, "paper"
# is the light React UI's: paper background, ink-coloured bonds, no bloom. Both
# go through the same code path so the two UIs cannot drift apart structurally.
_THEMES = {
    "dark": {
        "bg": (42, 42, 60),
        "bond_line_width": 3,
        "atom_palette": {
            6:  (0.82, 0.82, 0.92),
            7:  (0.35, 0.65, 1.00),
            8:  (1.00, 0.40, 0.40),
            9:  (0.35, 0.90, 0.55),
            16: (1.00, 0.85, 0.30),
            17: (0.35, 0.90, 0.55),
            15: (1.00, 0.65, 0.20),
        },
        "lift_dark_bonds": True,
        "glow": True,
        "legend_bg": (42, 42, 60, 255),
        "legend_fg": (205, 205, 220, 255),
        "legend_frame": (125, 125, 150, 255),
    },
    "paper": {
        "bg": (255, 253, 249),
        "bond_line_width": 2,
        "atom_palette": {
            6:  (0.11, 0.10, 0.09),
            7:  (0.12, 0.32, 0.51),
            8:  (0.72, 0.11, 0.11),
            9:  (0.08, 0.50, 0.24),
            16: (0.54, 0.43, 0.23),
            17: (0.08, 0.50, 0.24),
            15: (0.63, 0.40, 0.13),
        },
        "lift_dark_bonds": False,
        "glow": False,
        "legend_bg": (255, 253, 249, 255),
        "legend_fg": (92, 87, 79, 255),
        "legend_frame": (200, 194, 184, 255),
    },
}


def _compose_on_theme(img, alpha, size, cfg):
    """Composite a transparent RDKit render onto the theme's background."""
    from PIL import Image, ImageFilter

    w, h = size
    arr = np.array(img, dtype=np.float32)
    bg = np.full((h, w, 4), np.append(np.array(cfg["bg"]), [255]), dtype=np.float32)
    composed = arr * alpha + bg * (1 - alpha)

    if cfg["lift_dark_bonds"]:
        # RDKit draws bonds near-black; against a dark background they have to
        # be lifted. On paper they are already correct, so this is skipped.
        fg_mask = alpha[:, :, 0] > 0.3
        dark_bond = fg_mask & (composed[:, :, :3].max(axis=2) < 70)
        composed[dark_bond, 0] = np.clip(composed[dark_bond, 0] + 110, 0, 255)
        composed[dark_bond, 1] = np.clip(composed[dark_bond, 1] + 95, 0, 255)
        composed[dark_bond, 2] = np.clip(composed[dark_bond, 2] + 120, 0, 255)

    if cfg["glow"]:
        glow = img.filter(ImageFilter.GaussianBlur(radius=2))
        composed = composed + np.array(glow, dtype=np.float32) * alpha * 0.2

    return Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8), "RGBA")


def mol_to_dark_image(mol, size=(500, 400), theme="dark"):
    """Render a 2D molecular structure.

    theme="dark" (the default) matches the Streamlit app; theme="paper" is the
    light React UI - paper background, ink-coloured bonds, no glow.
    """
    from io import BytesIO
    from PIL import Image
    from rdkit.Chem.Draw import rdMolDraw2D

    cfg = _THEMES[theme]
    w, h = size

    draw = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = draw.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = cfg["bond_line_width"]
    opts.multipleBondOffset = 0.18
    opts.padding = 0.08
    opts.legendFontSize = 22
    opts.updateAtomPalette(cfg["atom_palette"])

    draw.DrawMolecule(mol)
    draw.FinishDrawing()

    img = Image.open(BytesIO(draw.GetDrawingText())).convert("RGBA")
    alpha = np.array(img, dtype=np.float32)[:, :, 3:4] / 255.0
    return _compose_on_theme(img, alpha, size, cfg)


def _importance_color(norm, theme="dark"):
    """Map normalized importance (0..1) to the highlight colour (0-1 floats).

    The single source for both the per-bond highlight and the legend bar, so
    the two can no longer disagree.
    """
    n = max(0.0, min(1.0, norm))
    if theme == "paper":
        # Ink blue -> deep amber: legible against a paper background.
        return (0.12 + 0.51 * n, 0.32 + 0.08 * n, 0.51 - 0.38 * n)
    return (0.55 + 0.45 * n, 0.25 + 0.65 * n, 0.90 - 0.80 * n)


def _cjk_font_path():
    """Return a font file path able to render CJK, or None if unavailable."""
    import glob
    import platform

    if platform.system() == "Windows":
        patterns = (
            r"C:\Windows\Fonts\msyh*.ttc",
            r"C:\Windows\Fonts\msyh*.ttf",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        )
    elif platform.system() == "Darwin":
        patterns = (
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        )
    else:
        patterns = (
            "/usr/share/fonts/opentype/noto/*.ttc",
            "/usr/share/fonts/truetype/noto/*.ttc",
            "/usr/share/fonts/noto-cjk/*.ttc",
            "/usr/share/fonts/truetype/wqy/*.ttf",
            "/usr/share/fonts/opentype/source-han-sans/*.otf",
        )
    for pattern in patterns:
        for fp in glob.glob(pattern):
            return fp
    return None


def _draw_importance_legend(img, w, h, theme="dark"):
    """Append a horizontal importance scale below a structure image.

    Gives the per-bond saturation gradient an explicit reference: left end is
    low importance, right end is high. Labels are bilingual (低/Low, 高/High)
    when a CJK font is available. Colours follow the rendering theme.
    """
    from PIL import Image, ImageDraw, ImageFont

    cfg = _THEMES[theme]
    bar_w, bar_h = 180, 12
    gap_top, gap_bottom = 16, 20
    canvas = Image.new("RGBA", (w, h + gap_top + bar_h + gap_bottom), cfg["legend_bg"])
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    bx = (w - bar_w) // 2
    by = h + gap_top
    for i in range(bar_w):
        norm = i / (bar_w - 1)
        rgb = _importance_color(norm, theme)
        draw.line(
            [(bx + i, by), (bx + i, by + bar_h)],
            fill=tuple(int(round(c * 255)) for c in rgb) + (255,),
        )
    draw.rectangle([bx - 1, by - 1, bx + bar_w, by + bar_h], outline=cfg["legend_frame"])

    font = None
    fp = _cjk_font_path()
    if fp:
        try:
            font = ImageFont.truetype(fp, 13)
        except Exception:
            font = None
    has_cjk = font is not None
    if font is None:
        font = ImageFont.load_default()
    low, high = ("低 / Low", "高 / High") if has_cjk else ("Low", "High")

    y = by + bar_h // 2 - 7
    draw.text(
        (bx - draw.textlength(low, font=font) - 12, y),
        low,
        fill=cfg["legend_fg"],
        font=font,
    )
    draw.text(
        (bx + bar_w + 12, y),
        high,
        fill=cfg["legend_fg"],
        font=font,
    )
    return canvas


def mol_to_dark_image_with_importance(mol, bond_weights, size=(500, 400), theme="dark"):
    """Render a 2D molecular structure with bonds highlighted by GNN importance.

    Important bonds are drawn in warmer/saturated colours; less important bonds
    appear dimmer. `theme` picks the palette (see _THEMES) - "paper" for the
    light React UI, "dark" for the Streamlit app.

    Args:
        mol: RDKit Mol object.
        bond_weights: dict mapping bond_idx -> importance (0~1).
                      Bonds not in the dict get a subtle default colour.
        size: (width, height) in pixels.
        theme: "dark" (default) or "paper".

    Returns:
        PIL Image (RGBA) with highlighted bonds.
    """
    from io import BytesIO
    from PIL import Image
    from rdkit.Chem.Draw import rdMolDraw2D

    cfg = _THEMES[theme]
    w, h = size

    if not bond_weights:
        # Fall back to standard rendering
        return mol_to_dark_image(mol, size, theme)

    # Per-bond highlight colours, from the same gradient the legend uses.
    max_w = max(bond_weights.values()) if bond_weights else 1.0
    highlight_colours = {}
    for bidx, wgt in bond_weights.items():
        norm = wgt / max_w if max_w > 0 else 0.0
        highlight_colours[bidx] = _importance_color(norm, theme)

    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = cfg["bond_line_width"]
    opts.multipleBondOffset = 0.18
    opts.padding = 0.08
    opts.legendFontSize = 22
    opts.updateAtomPalette(cfg["atom_palette"])

    # Use RDKit's highlight bonds API (positional args for highlight_bonds)
    highlight_bond_list = list(highlight_colours.keys())
    drawer.DrawMolecule(mol, None, highlight_bond_list, None, highlight_colours)
    drawer.FinishDrawing()

    img = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")
    alpha = np.array(img, dtype=np.float32)[:, :, 3:4] / 255.0
    composed = _compose_on_theme(img, alpha, size, cfg)
    return _draw_importance_legend(composed, w, h, theme)
