"""Weight grids. Every weight in the paper is fitted on the "uni3" grid.

uni3 puts all 7 slots on the same grid, −3.0 to 3.0 in steps of 0.1 (61 points). Using one list
for every slot lets the Methods state the grid in a single sentence, and lets every slot go
negative.

⚠ The older `wide` grid has a lower bound of 0 on the GEMME and ΔΔG slots, so those slots cannot
go negative. A value such as a_ddG = 0.0 fitted under `wide` is therefore produced by the grid
boundary, and any mechanism argued from an individual slot flips under uni3. The performance
conclusions (the ordering of the arms) are the same under both grids. `wide` is kept only as a
control.
"""
from __future__ import annotations

from contextlib import contextmanager

from . import common as C

NARROW = [list(g) for g in C.GRIDS]
WIDE_G = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0]
WIDE_D = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.2, 2.8, 3.6]
WIDE_R = [-0.6, -0.2, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4]
WIDE_S = [-0.6, -0.2, 0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.9, 1.2]
WIDE = [WIDE_G, WIDE_G, WIDE_D, WIDE_D, WIDE_R, WIDE_S, WIDE_S]


def _uniform(lo: float, hi: float, step: float = 0.1) -> list[float]:
    """The same evenly spaced grid on every slot."""
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 10) for i in range(n)]


# The confirmed protocol. The upper bound of 3.0 clips a_GEMME for weak backbones (measured at
# 4.0), but no slot sits on the boundary on ProteinGym, and widening to uni4 (±4.0) changes the
# mean by +0.0001, so 3.0 stands.
UNI3 = [_uniform(-3.0, 3.0)] * 7
UNI4 = [_uniform(-4.0, 4.0)] * 7
GRIDSETS = {"narrow": NARROW, "wide": WIDE, "uni3": UNI3, "uni4": UNI4}


@contextmanager
def use_grid(name: str):
    """Temporarily replace common.GRIDS. Always restored on exit."""
    if name not in GRIDSETS:
        raise ValueError(f"grid name must be one of {sorted(GRIDSETS)}")
    old = [list(g) for g in C.GRIDS]
    C.GRIDS[:] = [list(g) for g in GRIDSETS[name]]
    try:
        yield
    finally:
        C.GRIDS[:] = old
