"""Fig. 6: replica Eg vs alpha0 (a, b) and the best Eg vs beta (c, d), sigma_in = 1.

(a, b) Pruned Eg from the max-margin saddle (Eqs. B13-B14, B18, 23) for several
beta, the beta -> inf envelope (Eq. E8) and the noise floor (Eq. 4).
(c, d) "Eg minimised over alpha0" vs beta. The exact infimum is the capacity
limit kappa -> 0 (``eg_at_capacity``). The published curves took the minimum
over a finite alpha0 grid, which lies slightly above the infimum at small beta;
both are computed (the grids are config entries inferred from the published
points and checked in the tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from matplotlib.axes import Axes

from nn_expansion.sweep import Series, SeriesResult
from nn_expansion.theory import eg_at_capacity, eg_curve_sigma_in_1, large_beta_eg

from .common import Panel, label_axes, linspace, oracle_line, overlay_published

NAME = "fig06"


@dataclass(frozen=True)
class Config:
    sigma_out: list[float]          # panels (a, c) and (b, d)
    alpha0_grid: list[list[float]]  # per sigma_out, [start, stop, num] for (a, b)
    betas: list[float]              # finite betas of (a, b)
    beta_grid: list[float]          # [start, stop, num] for (c, d)
    min_alpha0_grid: list[list[float]]  # per sigma_out, grid minimised over in (c, d)


def plan(cfg: Config) -> list[Series]:
    return []                                   # theory only


def compute(cfg: Config, sims: dict[str, SeriesResult]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"sigma_out": np.array(cfg.sigma_out),
                                  "betas": np.array(cfg.betas)}
    bgrid = linspace(cfg.beta_grid)
    out["beta_grid"] = bgrid
    for k, so in enumerate(cfg.sigma_out):
        a0 = linspace(cfg.alpha0_grid[k])
        out[f"alpha0_{k}"] = a0
        out[f"eg_{k}"] = np.array([eg_curve_sigma_in_1(a0, b, so, pruned=True) for b in cfg.betas])
        out[f"eg_inf_{k}"] = np.where(a0 > 0, large_beta_eg(np.maximum(a0, 1e-300), so), np.nan)
        out[f"eg_cap_{k}"] = np.array([eg_at_capacity(b, so) for b in bgrid])
        g = linspace(cfg.min_alpha0_grid[k])
        out[f"eg_gridmin_{k}"] = np.array(
            [np.nanmin(eg_curve_sigma_in_1(g, b, so, pruned=True)) for b in bgrid])
    return out


def _eg_alpha0(ax: Axes, res: dict[str, np.ndarray], cfg: Config, k: int) -> None:
    so, panel = cfg.sigma_out[k], f"fig06{'ab'[k]}"
    label_axes(ax, r"$\alpha_0$", r"$E_g$", rf"Fig. 6{'ab'[k]}  $\sigma_{{out}}={so:g}$ (grey: published)")
    a0 = res[f"alpha0_{k}"]
    for i, b in enumerate(cfg.betas):
        overlay_published(ax, panel, rf"\beta={b:g}", legend="published" if i == 0 else None)
        ax.plot(a0, res[f"eg_{k}"][i], label=rf"$\beta={b:g}$")
    overlay_published(ax, panel, r"\beta=\infty")
    ax.plot(a0, res[f"eg_inf_{k}"], "r", label=r"$\beta=\infty$")
    oracle_line(ax, so)


def _eg_beta(ax: Axes, res: dict[str, np.ndarray], cfg: Config, k: int) -> None:
    so, panel = cfg.sigma_out[k], f"fig06{'cd'[k]}"
    label_axes(ax, r"$\beta$", r"$\min_{\alpha_0} E_g$", rf"Fig. 6{'cd'[k]}  $\sigma_{{out}}={so:g}$ (grey: published)")
    overlay_published(ax, panel, "", 0, legend="published")
    ax.plot(res["beta_grid"], res[f"eg_gridmin_{k}"], "o", ms=3, label="min over published grid")
    ax.plot(res["beta_grid"], res[f"eg_cap_{k}"], "r", label=r"exact: $\alpha_0\to\alpha_{0,c}$")
    oracle_line(ax, so)


def panels(cfg: Config) -> dict[str, Panel]:
    out: dict[str, Panel] = {}
    for k in range(len(cfg.sigma_out)):
        out[f"fig06{'ab'[k]}"] = partial(_eg_alpha0, cfg=cfg, k=k)
        out[f"fig06{'cd'[k]}"] = partial(_eg_beta, cfg=cfg, k=k)
    return out
