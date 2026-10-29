"""Figs. 9-10: learning the quadratic teacher of Eq. (31) (sigma_out = 0, linear fraction a).

A linear student cannot represent the quadratic part, which then acts as label
noise of std (1-a)/a (Eq. 32).
Fig. 9a: stochastic expansion (sigma_in = 1) trained on the quadratic teacher vs
  the replica curve of the noisy linear teacher with sigma_out = (1-a)/a.
Fig. 9b: deterministic sparse expansion trained on the quadratic teacher vs the
  same network trained on the equivalent noisy linear teacher (both simulated).
Fig. 10: deterministic expansion at large beta, sparse (a) and dense (b), pruned
  vs unpruned, on the quadratic teacher.
Errors against the quadratic teacher are Monte Carlo (n_test fresh inputs).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from matplotlib.axes import Axes

from nn_expansion.simulate import (Deterministic, Expansion, MaxMargin, NoExpansion, Stochastic,
                              TrialSpec)
from nn_expansion.sweep import Series, SeriesResult
from nn_expansion.teacher import quad_to_noise_sigma
from nn_expansion.theory import eg_curve_sigma_in_1

from .common import (Panel, alpha0_grid, errorbar, label_axes, linspace, oracle_line, overlay_published,
                     series_arrays)

NAME = "fig09_10"


@dataclass(frozen=True)
class Config:
    N0: int
    a: float
    f_sparse: float
    f_dense: float
    A: float
    betas_9: list[float]           # betas of Fig. 9 (unexpanded and expanded)
    beta_10: float
    alpha0_grid_9a: list[float]
    alpha0_grid_9b_10: list[float]
    trials_9a: int
    trials_9b_10: int
    seed: int
    n_test: int
    rule: str                     # trial averaging, see nn_expansion.sweep.SweepResult.mean
    n_sem: float                  # error bars in standard errors (published: 2)


def _specs(cfg: Config, a0: np.ndarray, teacher_a: float, sigma_out: float, expansion: Expansion,
           unpruned: bool) -> tuple[TrialSpec, ...]:
    return alpha0_grid(a0, N0=cfg.N0, teacher_a=teacher_a, sigma_out=sigma_out, expansion=expansion,
                       learner=MaxMargin(), student="linear", measure_unpruned=unpruned,
                       n_test=cfg.n_test)


def plan(cfg: Config) -> list[Series]:
    s_eff = quad_to_noise_sigma(cfg.a)
    a9a, a9b = linspace(cfg.alpha0_grid_9a), linspace(cfg.alpha0_grid_9b_10)
    out: list[Series] = []
    for i, b in enumerate(cfg.betas_9):
        exp_s = NoExpansion() if b == 1.0 else Stochastic(b, 1.0)
        out.append(Series(f"9a_quad{i}", _specs(cfg, a9a, cfg.a, 0.0, exp_s, False),
                          cfg.trials_9a, cfg.seed, len(out)))
        exp_d = NoExpansion() if b == 1.0 else Deterministic(b, cfg.f_sparse, cfg.A)
        for tag, ta, so in (("quad", cfg.a, 0.0), ("noisy", 1.0, s_eff)):
            out.append(Series(f"9b_{tag}{i}", _specs(cfg, a9b, ta, so, exp_d, False),
                              cfg.trials_9b_10, cfg.seed, len(out)))
    for kind, f in (("sparse", cfg.f_sparse), ("dense", cfg.f_dense)):
        out.append(Series(f"10_{kind}", _specs(cfg, a9b, cfg.a, 0.0, Deterministic(cfg.beta_10, f, cfg.A), True),
                          cfg.trials_9b_10, cfg.seed, len(out)))
    return out


def compute(cfg: Config, sims: dict[str, SeriesResult]) -> dict[str, np.ndarray]:
    s_eff = quad_to_noise_sigma(cfg.a)
    a9a, a9b = linspace(cfg.alpha0_grid_9a), linspace(cfg.alpha0_grid_9b_10)
    out: dict[str, np.ndarray] = {"alpha0_9a": a9a, "alpha0_9b_10": a9b}
    for i, b in enumerate(cfg.betas_9):
        out.update(series_arrays(f"9a_quad{i}", sims[f"9a_quad{i}"].sweep, ("eg_pruned",), cfg.rule))
        out[f"9a_thy{i}"] = eg_curve_sigma_in_1(a9a, b, s_eff, pruned=True)
        for tag in ("quad", "noisy"):
            out.update(series_arrays(f"9b_{tag}{i}", sims[f"9b_{tag}{i}"].sweep, ("eg_pruned",),
                                     cfg.rule))
    for kind in ("sparse", "dense"):
        out.update(series_arrays(f"10_{kind}", sims[f"10_{kind}"].sweep, ("eg_pruned", "eg_unpruned"),
                                 cfg.rule))
    return out


def _sub(cfg: Config) -> str:
    return rf"$a={cfg.a:g}$, $N_0={cfg.N0}$"


def _fig9a(ax: Axes, res: dict[str, np.ndarray], cfg: Config) -> None:
    s_eff = quad_to_noise_sigma(cfg.a)
    label_axes(ax, r"$\alpha_0$", r"$E_g$ (pruned)", f"Fig. 9a  stochastic, {_sub(cfg)}, {cfg.trials_9a} trials")
    for i, b in enumerate(cfg.betas_9):
        overlay_published(ax, "fig09a", "quadratic", i, legend="published" if i == 0 else None)
        overlay_published(ax, "fig09a", "noisy", i)
        line = ax.plot(res["alpha0_9a"], res[f"9a_thy{i}"], lw=1,
                       label=rf"noisy theory $\sigma_{{out}}={s_eff:g}$, β={b:g}")[0]
        errorbar(ax, res["alpha0_9a"], res, f"9a_quad{i}", "eg_pruned", cfg.n_sem, fmt=".",
                 color=line.get_color(), label=f"quadratic sim, β={b:g}")
    oracle_line(ax, s_eff)


def _fig9b(ax: Axes, res: dict[str, np.ndarray], cfg: Config) -> None:
    label_axes(ax, r"$\alpha_0$", r"$E_g$ (pruned)",
               f"Fig. 9b  det. f={cfg.f_sparse:g}, A={cfg.A:g}, {_sub(cfg)}, {cfg.trials_9b_10} trials")
    for i, b in enumerate(cfg.betas_9):
        overlay_published(ax, "fig09b", "quadratic", i, legend="published" if i == 0 else None)
        overlay_published(ax, "fig09b", "noisy", i)
        color = ax.plot([], [])[0].get_color()
        errorbar(ax, res["alpha0_9b_10"], res, f"9b_quad{i}", "eg_pruned", cfg.n_sem, fmt=".",
                 color=color, label=f"quadratic, β={b:g}")
        errorbar(ax, res["alpha0_9b_10"], res, f"9b_noisy{i}", "eg_pruned", cfg.n_sem, fmt="--",
                 color=color, label=f"noisy, β={b:g}")
    oracle_line(ax, quad_to_noise_sigma(cfg.a))


def _fig10(ax: Axes, res: dict[str, np.ndarray], cfg: Config, kind: str) -> None:
    p, f = ("a", cfg.f_sparse) if kind == "sparse" else ("b", cfg.f_dense)
    label_axes(ax, r"$\alpha_0$", r"$E_g$",
               f"Fig. 10{p}  f={f:g}, β={cfg.beta_10:g}, {_sub(cfg)}, {cfg.trials_9b_10} trials")
    overlay_published(ax, f"fig10{p}", "pruned", legend="published")
    overlay_published(ax, f"fig10{p}", "unpruned")
    errorbar(ax, res["alpha0_9b_10"], res, f"10_{kind}", "eg_pruned", cfg.n_sem, label="pruned")
    errorbar(ax, res["alpha0_9b_10"], res, f"10_{kind}", "eg_unpruned", cfg.n_sem, label="unpruned")
    oracle_line(ax, quad_to_noise_sigma(cfg.a))


def panels(cfg: Config) -> dict[str, Panel]:
    return {"fig09a": partial(_fig9a, cfg=cfg), "fig09b": partial(_fig9b, cfg=cfg),
            "fig10a": partial(_fig10, cfg=cfg, kind="sparse"), "fig10b": partial(_fig10, cfg=cfg, kind="dense")}
