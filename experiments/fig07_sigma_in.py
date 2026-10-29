"""Fig. 7: the variance sigma_in^2 of the stochastic units.

(a) Pruned Eg vs alpha0 for several sigma_in: max-margin simulations of the
stochastically expanded perceptron (Eq. 10) against the general-sigma_in replica
theory (Eqs. B5-B11).
(b) Replica Eg vs sigma_in at fixed (alpha0, beta, sigma_out), and the optimal
sigma_in (App. F).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from matplotlib.axes import Axes

from nn_expansion.simulate import MaxMargin, Stochastic
from nn_expansion.sweep import Series, SeriesResult
from nn_expansion.theory import (capacity_alpha0, eg_from_overlap, optimal_sigma_in,
                            saddle_alpha0_sweep, saddle_sigma_in_sweep)

from .common import (Panel, alpha0_grid, errorbar, label_axes, linspace, oracle_line, overlay_published,
                     series_arrays)

NAME = "fig07"


@dataclass(frozen=True)
class Config:
    # (a)
    N0: int
    beta: float
    sigma_out: float
    sigma_in: list[float]
    alpha0_grid: list[float]
    trials: int
    seed: int
    rule: str                     # trial averaging, see nn_expansion.sweep.SweepResult.mean
    n_sem: float                  # error bars in standard errors (published: 2)
    # (b)
    b_alpha0: float
    b_beta: float
    b_sigma_out: float
    b_sigma_in_grid: list[float]


def plan(cfg: Config) -> list[Series]:
    a0 = linspace(cfg.alpha0_grid)
    return [Series(f"sim{k}", alpha0_grid(a0, N0=cfg.N0, teacher_a=1.0, sigma_out=cfg.sigma_out,
                                          expansion=Stochastic(cfg.beta, si), learner=MaxMargin(),
                                          student="linear", measure_unpruned=False, n_test=0),
                   cfg.trials, cfg.seed, k)
            for k, si in enumerate(cfg.sigma_in)]


def compute(cfg: Config, sims: dict[str, SeriesResult]) -> dict[str, np.ndarray]:
    a0 = linspace(cfg.alpha0_grid)
    out: dict[str, np.ndarray] = {"alpha0": a0, "sigma_in": np.array(cfg.sigma_in)}
    a0_thy = a0[a0 < capacity_alpha0(cfg.beta, cfg.sigma_out)]
    for k, si in enumerate(cfg.sigma_in):
        out.update(series_arrays(f"sim{k}", sims[f"sim{k}"].sweep, ("eg_pruned",), cfg.rule))
        sps = saddle_alpha0_sweep(a0_thy, cfg.beta, cfg.sigma_out, si)
        thy = np.full(a0.shape, np.nan)
        thy[: a0_thy.size] = [eg_from_overlap(p.R_pruned, cfg.sigma_out) for p in sps]
        out[f"thy{k}"] = thy
    sig = linspace(cfg.b_sigma_in_grid)
    sps = saddle_sigma_in_sweep(cfg.b_alpha0, cfg.b_beta, cfg.b_sigma_out, sig)
    out["b_sigma_in"] = sig
    out["b_eg"] = np.array([eg_from_overlap(p.R_pruned, cfg.b_sigma_out) for p in sps])
    s_opt, e_opt = optimal_sigma_in(cfg.b_alpha0, cfg.b_beta, cfg.b_sigma_out,
                                    (float(sig[0]), float(sig[-1])))
    out["b_opt"] = np.array([s_opt, e_opt])
    return out


def _fig7a(ax: Axes, res: dict[str, np.ndarray], cfg: Config) -> None:
    label_axes(ax, r"$\alpha_0$", r"$E_g$ (pruned)",
               rf"Fig. 7a  $\beta={cfg.beta:g}$, $\sigma_{{out}}={cfg.sigma_out:g}$, "
               rf"$N_0={cfg.N0}$, {cfg.trials} trials")
    for k, si in enumerate(cfg.sigma_in):
        overlay_published(ax, "fig07a", rf"\sigma_{{in}} = {si:g}", legend="published" if k == 0 else None)
        overlay_published(ax, "fig07a", "theory", k)
        line = ax.plot(res["alpha0"], res[f"thy{k}"], lw=1)[0]
        errorbar(ax, res["alpha0"], res, f"sim{k}", "eg_pruned", cfg.n_sem, fmt=".",
                 color=line.get_color(), label=rf"$\sigma_{{in}}={si:g}$ sim / theory")
    oracle_line(ax, cfg.sigma_out)


def _fig7b(ax: Axes, res: dict[str, np.ndarray], cfg: Config) -> None:
    label_axes(ax, r"$\sigma_{in}$", r"$E_g$ (pruned)",
               rf"Fig. 7b  $\alpha_0={cfg.b_alpha0:g}$, $\beta={cfg.b_beta:g}$, "
               rf"$\sigma_{{out}}={cfg.b_sigma_out:g}$ (theory)")
    overlay_published(ax, "fig07b", "", 0, legend="published")
    ax.plot(res["b_sigma_in"], res["b_eg"], label="replica")
    s_opt, e_opt = res["b_opt"]
    ax.plot([s_opt], [e_opt], "bx", ms=8, label=rf"optimum $\sigma_{{in}}={s_opt:.3f}$")


def panels(cfg: Config) -> dict[str, Panel]:
    return {"fig07a": partial(_fig7a, cfg=cfg), "fig07b": partial(_fig7b, cfg=cfg)}
