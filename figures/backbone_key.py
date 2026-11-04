"""The one place where the colour and marker of each of the 12 backbones is **pinned to its name**.

Why this is a module
  Two figure scripts used to pick colours from a sequential colormap (GnBu/OrRd) with their own
  family_colors() and **zip them against a different sort order in each figure**. The two sorts
  differed (accuracy descending versus ΔΔG share ascending), so the same model appeared in two
  different colours in two figures. Colour has to follow **the object**, not the rank.

How the colours are chosen
  The families stay: the 8 structure-aware backbones are cool, the 4 structure-unaware ones warm.
  That way the "blue shades / red shades" in the captions, the x positions and the legend groups
  all say the same thing. What changed is that **no sequential ramp is used inside a family**.
  Eight points off a ramp leave neighbouring colours only 0.04 apart in lightness, which reads as
  one colour. Lightness is instead alternated light/dark in OKLCH coordinates so that
  **neighbouring slots are always far apart**.

Why markers are used as well
  Colour alone distinguishes at most about 8 categories, and there are 12 here. Both figures are
  scatter-like and the points can overlap, so it is not neighbouring pairs but **every pair** that
  must separate, and the cool region of sRGB has no room for that. Colour and shape are therefore
  used together. Shapes repeat **in pairs** (o o ^ ^ s s D D), and the two slots sharing a shape
  are, by the rule above, already the furthest apart in colour: a pair with similar colours always
  has different shapes.

Validation (light mode)
  Neighbouring pairs pass at CVD deltaE 13.0 (deutan) and normal-vision deltaE 15.1.
  Two warnings remain, accepted as limits of sRGB.
    - Chroma floor: ESM3 #008D8E (0.099) and ProteinMPNN #006985 (0.091). Among cool hues, teal
      tops out near 0.10 chroma at medium to dark lightness in sRGB and cannot go higher
      (confirmed by bisection). Lightness and shape carry the identity as well, so this stands.
    - Light colours fall below 3:1 contrast with the background. The points are small and have a
      white outline, so this is unavoidable. Both figures also export every value to _data.csv,
      so the numbers can be read as a table.
"""

# structure-aware 8 (cool). The slots follow the left-to-right order of the ablation figure
# (function-axis accuracy descending), which packs 12 points into one row and so has the tightest
# neighbour spacing.
AWARE = [
    ("AIDO_RAG16B", "#45C2AB", "o"),   # L .74  light teal-green
    ("VenusREM",    "#07599F", "o"),   # L .46  deep blue
    ("ProSST",      "#29A5D9", "^"),   # L .68  light sky
    ("S2F",         "#5E3CAD", "^"),   # L .46  deep indigo
    ("SaProt650M",  "#5992E7", "s"),   # L .66  mid blue
    ("ESM3",        "#008D8E", "s"),   # L .58  mid teal
    ("ESMIF1",      "#A791FA", "D"),   # L .72  light indigo
    ("ProteinMPNN", "#006985", "D"),   # L .48  deep steel blue
]
# structure-unaware 4 (warm)
UNAWARE = [
    ("VespaG",          "#ED9C55", "o"),   # L .76  light orange
    ("TranceptionLret", "#C2460D", "o"),   # L .56  strong orange
    ("ESM1v",           "#E8777D", "^"),   # L .70  salmon
    ("ESM2",            "#8C2C33", "^"),   # L .44  deep red
]

CMAP = {b: c for b, c, _ in AWARE + UNAWARE}
MARKER = {b: m for b, _, m in AWARE + UNAWARE}
STRUCT = {b for b, _, _ in AWARE}
