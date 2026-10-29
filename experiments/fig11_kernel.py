"""Fig. 11: quadratic-kernel student with stochastically expanded input.

Teacher: noisy quadratic form, h0 = (1/N0) x^T W0 x + sigma_out eps (Eq. 31 with
a = 0). Student: max margin on the features vec(z z^T) of z = [x, x_tilde]
(kernel K = (z.z')^2), x_tilde ~ N(0, 1) per pattern.
Pruning sets x_tilde = 0 at test time; keeping the units feeds fresh x_tilde.
(a) beta = 1 vs pruned beta > 1; (b) pruned vs unpruned at beta > 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from matplotlib.axes import Axes

from nn_expansion.simulate import Expansion, MaxMargin, NoExpansion, Stochastic, TrialSpec
from nn_expansion.sweep import Series, SeriesResult

from .common import (Panel, alpha0_grid, errorbar, label_axes, linspace, oracle_line, overlay_published,
                     series_arrays)

NAME = "fig11"


@dataclass(frozen=True)
class Config:
    N0: int
    beta: float
    sigma_out: float
    alpha0_grid: list[float]
    trials: int
    seed: int
    n_test: int
    rule: str                     # trial averaging, see nn_expansion.sweep.SweepResult.mean
    n_sem: float                  # error bars in standard errors (published: 2)


def _specs(cfg: Config, a0: np.ndarray, expansion: Expansion, unpruned: bool) -> tuple[TrialSpec, ...]:
    return alpha0_grid(a0, N0=cfg.N0, teacher_a=0.0, sigma_out=cfg.sigma_out, expansion=expansion,
                       learner=MaxMargin(), student="quadratic_kernel", measure_unpruned=unpruned,
                       n_test=cfg.n_test)


def plan(cfg: Config) -> list[Series]:
    a0 = linspace(cfg.alpha0_grid)
    return [Series("beta1", _specs(cfg, a0, NoExpansion(), False), cfg.trials, cfg.seed, 0),
            Series("expanded", _specs(cfg, a0, Stochastic(cfg.beta, 1.0), True), cfg.trials, cfg.seed, 1)]


def compute(cfg: Config, sims: dict[str, SeriesResult]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"alpha0": linspace(cfg.alpha0_grid)}
    out.update(series_arrays("beta1", sims["beta1"].sweep, ("eg_pruned",), cfg.rule))
    out.update(series_arrays("expanded", sims["expanded"].sweep, ("eg_pruned", "eg_unpruned"),
                             cfg.rule))
    return out


def _sub(cfg: Config) -> str:
    return rf"$\sigma_{{out}}={cfg.sigma_out:g}$, $N_0={cfg.N0}$, {cfg.trials} trials"


def _fig11a(ax: Axes, res: dict[str, np.ndarray], cfg: Config) -> None:
    label_axes(ax, r"$\alpha_0$", r"$E_g$", "Fig. 11a  kernel student, " + _sub(cfg))
    overlay_published(ax, "fig11a", r"\beta =1", legend="published")
    overlay_published(ax, "fig11a", rf"\beta ={cfg.beta:g}")
    errorbar(ax, res["alpha0"], res, "beta1", "eg_pruned", cfg.n_sem, label=r"$\beta=1$")
    errorbar(ax, res["alpha0"], res, "expanded", "eg_pruned", cfg.n_sem, label=rf"$\beta={cfg.beta:g}$ pruned")
    oracle_line(ax, cfg.sigma_out)


def _fig11b(ax: Axes, res: dict[str, np.ndarray], cfg: Config) -> None:
    label_axes(ax, r"$\alpha_0$", r"$E_g$", f"Fig. 11b  kernel student, β={cfg.beta:g}, " + _sub(cfg))
    overlay_published(ax, "fig11b", "pruned", legend="published")
    overlay_published(ax, "fig11b", "unpruned")
    errorbar(ax, res["alpha0"], res, "expanded", "eg_pruned", cfg.n_sem, label="pruned")
    errorbar(ax, res["alpha0"], res, "expanded", "eg_unpruned", cfg.n_sem, label="unpruned")
    oracle_line(ax, cfg.sigma_out)


def panels(cfg: Config) -> dict[str, Panel]:
    return {"fig11a": partial(_fig11a, cfg=cfg), "fig11b": partial(_fig11b, cfg=cfg)}
