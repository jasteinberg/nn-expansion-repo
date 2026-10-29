"""Fig. 5: capacity alpha0_c vs inverse label-noise variance (mean field, sigma_in = 1).

alpha0_c(beta, sigma_out) = beta alpha_c(sigma_out), with alpha_c the kappa -> 0
root of Eqs. (C4)-(C5). Gardner limit: alpha_c -> 2 as sigma_out -> inf.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes

from nn_expansion.sweep import Series, SeriesResult
from nn_expansion.theory import capacity_alpha0

from .common import Panel, label_axes, linspace, overlay_published

NAME = "fig05"


@dataclass(frozen=True)
class Config:
    inv_sigma2: list[float]      # [start, stop, num] grid of sigma_out^-2
    betas: list[float]


def plan(cfg: Config) -> list[Series]:
    return []                                   # theory only


def compute(cfg: Config, sims: dict[str, SeriesResult]) -> dict[str, np.ndarray]:
    x = linspace(cfg.inv_sigma2)
    ac = np.array([[capacity_alpha0(b, 1.0 / np.sqrt(v)) for v in x] for b in cfg.betas])
    return {"inv_sigma2": x, "betas": np.array(cfg.betas), "alpha0_c": ac}


def _fig5(ax: Axes, res: dict[str, np.ndarray]) -> None:
    label_axes(ax, r"$\sigma_{out}^{-2}$", r"$\alpha_{0,c}$", "Fig. 5  capacity (grey: published)")
    for i, b in enumerate(res["betas"]):
        overlay_published(ax, "fig05", rf"\beta={b:g}", legend="published" if i == 0 else None)
        ax.plot(res["inv_sigma2"], res["alpha0_c"][i], label=rf"$\beta={b:g}$")


def panels(cfg: Config) -> dict[str, Panel]:
    return {"fig05": _fig5}
