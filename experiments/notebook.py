"""Draw single panels inline (e.g. in a notebook) from saved results.

    from experiments.notebook import list_panels, show
    list_panels("quick")            # panel names that have saved results
    show("fig08b", "quick")         # draws that panel from results/fig08_quick.npz

Results come from ``python -m experiments.reproduce`` (a whole run or a --merge of
shards); ``show`` only reads and draws, it never computes.
"""
from __future__ import annotations

from types import ModuleType

import matplotlib.pyplot as plt

from .common import RESULTS_DIR, Panel, draw_panel, load_config, load_results
from .reproduce import FIGURES, module


def _panels(profile: str) -> dict[str, tuple[ModuleType, Panel]]:
    out: dict[str, tuple[ModuleType, Panel]] = {}
    for name in FIGURES:
        mod = module(name)
        for panel, fn in mod.panels(load_config(mod.NAME, profile, mod.Config)).items():
            out[panel] = (mod, fn)
    return out


def list_panels(profile: str) -> list[str]:
    """Panels whose figure has saved results for ``profile``."""
    return [p for p, (mod, _) in _panels(profile).items()
            if (RESULTS_DIR / f"{mod.NAME}_{profile}.npz").exists()]


def show(panel: str, profile: str) -> plt.Figure:
    """Draw ``panel`` from the saved results of its figure and return the figure."""
    table = _panels(profile)
    if panel not in table:
        raise KeyError(f"unknown panel {panel!r}; known: {sorted(table)}")
    mod, fn = table[panel]
    res, _ = load_results(mod.NAME, profile)
    fig, _ = draw_panel(fn, res)
    return fig
