"""Figs. 2-4: deterministic sparse/dense expansion, pruned vs unpruned (simulation).

A two-layer student (Eqs. 5-6, fixed random J, sparsity f, activity scale A)
learns the noisy linear teacher by max margin. Generalization error is measured
after pruning the hidden units (exact, Eq. 23) and with them kept (Monte Carlo
through the same fixed J).

Fig. 2: pruned Eg for beta = 1 and for sparse / dense expansion, at two betas.
Fig. 3: sparse expansion, pruned vs unpruned.  Fig. 4: dense, pruned vs unpruned.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from matplotlib.axes import Axes

from nn_expansion.simulate import Deterministic, Expansion, MaxMargin, NoExpansion, TrialSpec
from nn_expansion.sweep import Series, SeriesResult

from .common import (Panel, alpha0_grid, errorbar, label_axes, linspace, oracle_line, overlay_published,
                     series_arrays)

NAME = "fig02_04"
PANEL = "ab"            # panel letter per entry of Config.betas


@dataclass(frozen=True)
class Config:
    N0: int
    betas: list[float]
    f_sparse: float
    f_dense: float
    A: float
    sigma_out: float
    alpha0_grid: list[float]
    trials: int
    seed: int
    n_test: int
    rule: str                     # trial averaging, see nn_expansion.sweep.SweepResult.mean
    n_sem: float                  # error bars in standard errors (published: 1)


def _specs(cfg: Config, a0: np.ndarray, expansion: Expansion) -> tuple[TrialSpec, ...]:
    return alpha0_grid(a0, N0=cfg.N0, teacher_a=1.0, sigma_out=cfg.sigma_out, expansion=expansion,
                       learner=MaxMargin(), student="linear",
                       measure_unpruned=not isinstance(expansion, NoExpansion), n_test=cfg.n_test)


def plan(cfg: Config) -> list[Series]:
    a0 = linspace(cfg.alpha0_grid)
    out = [Series("beta1", _specs(cfg, a0, NoExpansion()), cfg.trials, cfg.seed, 0)]
    sid = 1
    for i, b in enumerate(cfg.betas):
        for kind, f in (("sparse", cfg.f_sparse), ("dense", cfg.f_dense)):
            out.append(Series(f"{kind}{i}", _specs(cfg, a0, Deterministic(b, f, cfg.A)),
                              cfg.trials, cfg.seed, sid))
            sid += 1
    return out


def compute(cfg: Config, sims: dict[str, SeriesResult]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"alpha0": linspace(cfg.alpha0_grid)}
    out.update(series_arrays("beta1", sims["beta1"].sweep, ("eg_pruned",), cfg.rule))
    for i in range(len(cfg.betas)):
        for kind in ("sparse", "dense"):
            out.update(series_arrays(f"{kind}{i}", sims[f"{kind}{i}"].sweep,
                                     ("eg_pruned", "eg_unpruned"), cfg.rule))
    return out


def _sub(cfg: Config) -> str:
    return rf"$\sigma_{{out}}={cfg.sigma_out:g}$, $A={cfg.A:g}$, $N_0={cfg.N0}$, {cfg.trials} trials"


def _fig2(ax: Axes, res: dict[str, np.ndarray], cfg: Config, i: int) -> None:
    p, b, a0 = PANEL[i], cfg.betas[i], res["alpha0"]
    label_axes(ax, r"$\alpha_0$", r"$E_g$ (pruned)", f"Fig. 2{p}  β={b:g}, " + _sub(cfg))
    for lab in (r"\beta =1", "f =0.02", "f =0.5"):
        overlay_published(ax, f"fig02{p}", lab, legend="published" if lab.startswith(r"\beta") else None)
    errorbar(ax, a0, res, "beta1", "eg_pruned", cfg.n_sem, label=r"$\beta=1$")
    errorbar(ax, a0, res, f"sparse{i}", "eg_pruned", cfg.n_sem, label=f"f={cfg.f_sparse:g}")
    errorbar(ax, a0, res, f"dense{i}", "eg_pruned", cfg.n_sem, label=f"f={cfg.f_dense:g}")
    oracle_line(ax, cfg.sigma_out)


def _pruning(ax: Axes, res: dict[str, np.ndarray], cfg: Config, i: int, fig_no: int) -> None:
    kind, f = ("sparse", cfg.f_sparse) if fig_no == 3 else ("dense", cfg.f_dense)
    p, a0 = PANEL[i], res["alpha0"]
    label_axes(ax, r"$\alpha_0$", r"$E_g$", f"Fig. {fig_no}{p}  f={f:g}, β={cfg.betas[i]:g}, " + _sub(cfg))
    overlay_published(ax, f"fig0{fig_no}{p}", "pruned", legend="published")
    overlay_published(ax, f"fig0{fig_no}{p}", "unpruned")
    errorbar(ax, a0, res, f"{kind}{i}", "eg_pruned", cfg.n_sem, label="pruned")
    errorbar(ax, a0, res, f"{kind}{i}", "eg_unpruned", cfg.n_sem, label="unpruned")
    oracle_line(ax, cfg.sigma_out)


def panels(cfg: Config) -> dict[str, Panel]:
    out: dict[str, Panel] = {}
    for i in range(len(cfg.betas)):
        out[f"fig02{PANEL[i]}"] = partial(_fig2, cfg=cfg, i=i)
        for fig_no in (3, 4):
            out[f"fig0{fig_no}{PANEL[i]}"] = partial(_pruning, cfg=cfg, i=i, fig_no=fig_no)
    return out
