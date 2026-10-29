"""Shared plumbing for the figure scripts: configs, provenance, reference overlays.

Configs live in configs/<figure>.toml: [paper] holds the parameters of the
published figure, and [quick] (a small run that shows the effect) lists only the
keys it changes. [paper] must set every key of the figure's Config dataclass and
no other key is accepted, so a typo cannot silently fall back to a default.
"""
from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
import time
import tomllib
from pathlib import Path
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from nn_expansion.simulate import TrialSpec
from nn_expansion.teacher import min_generalization_error

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "configs"
REFERENCE_DIR = REPO / "reference" / "published"
RESULTS_DIR = REPO / "results"

C = TypeVar("C")


# --------------------------------------------------------------- configs ----

def load_config(figure: str, profile: str, cls: type[C]) -> C:
    """Profile ``profile`` of configs/<figure>.toml as dataclass ``cls`` (strict).

    [paper] must set every field of ``cls`` and nothing else. Any other profile
    is a table of overrides of [paper]: it may only name fields of [paper], and
    each override must differ from the paper value, so the table is exactly the
    list of differences.
    """
    path = CONFIG_DIR / f"{figure}.toml"
    with path.open("rb") as fh:
        tables = tomllib.load(fh)
    if "paper" not in tables or profile not in tables:
        raise KeyError(f"{path}: need [paper] and [{profile}] tables (has {sorted(tables)})")
    paper = tables["paper"]
    names = {f.name for f in dataclasses.fields(cls)}          # type: ignore[arg-type]
    missing, unknown = names - paper.keys(), paper.keys() - names
    if missing or unknown:
        raise KeyError(f"{path} [paper]: missing {sorted(missing)}, unknown {sorted(unknown)}")
    raw = dict(paper)
    if profile != "paper":
        overrides = tables[profile]
        unknown = overrides.keys() - names
        same = sorted(k for k in overrides.keys() & names if overrides[k] == paper[k])
        if unknown or same:
            raise KeyError(f"{path} [{profile}]: unknown {sorted(unknown)}, equal to [paper] {same}")
        raw.update(overrides)
    return cls(**raw)


def linspace(spec: list[float]) -> np.ndarray:
    """[start, stop, num] from a config -> np.linspace(start, stop, int(num))."""
    if len(spec) != 3:
        raise ValueError(f"grid spec must be [start, stop, num], got {spec}")
    return np.linspace(spec[0], spec[1], int(spec[2]))


# ------------------------------------------------------------ provenance ----

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def provenance(figure: str, profile: str, cfg: Any, runtime_s: float) -> dict[str, Any]:
    return {
        "figure": figure,
        "profile": profile,
        "config": dataclasses.asdict(cfg),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_s": round(runtime_s, 1),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def save_results(figure: str, profile: str, arrays: dict[str, np.ndarray],
                 meta: dict[str, Any]) -> Path:
    """results/<figure>_<profile>.npz (+ .json with config and git state)."""
    bad = [k for k, v in arrays.items() if np.asarray(v).dtype.kind == "f" and np.any(np.isinf(v))]
    if bad:
        raise ValueError(f"refusing to save infinite values in {bad}")
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"{figure}_{profile}"
    np.savez(stem.with_suffix(".npz"), **arrays)
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return stem.with_suffix(".npz")


def load_results(figure: str, profile: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stem = RESULTS_DIR / f"{figure}_{profile}"
    with np.load(stem.with_suffix(".npz")) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(stem.with_suffix(".json").read_text())
    if meta["figure"] != figure or meta["profile"] != profile:
        raise ValueError(f"{stem}.json describes {meta['figure']}/{meta['profile']}")
    return arrays, meta


# ------------------------------------------------------------- reference ----

def reference(panel: str) -> dict[str, Any]:
    """Published data of one panel (reference/published/<panel>.json)."""
    return json.loads((REFERENCE_DIR / f"{panel}.json").read_text())


def reference_series(panel: str, label: str, index: int = 0) -> dict[str, np.ndarray]:
    """The ``index``-th published series called ``label`` as float arrays."""
    hits = [s for s in reference(panel)["series"] if s["label"] == label]
    if len(hits) <= index:
        raise KeyError(f"{panel}: no series {label!r} #{index} "
                       f"(have {[s['label'] for s in reference(panel)['series']]})")
    s = hits[index]
    out = {k: np.array(s[k], float) for k in ("x", "y", "err_lo", "err_hi") if k in s}
    return out


# ---------------------------------------------------------------- plots ----

PUBLISHED_STYLE: dict[str, Any] = dict(color="0.6", lw=3.0, alpha=0.45, zorder=0)


Panel = Callable[[Axes, dict[str, np.ndarray]], None]     # draws one panel from saved results


def label_axes(ax: Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)


def overlay_published(ax: Axes, panel: str, label: str, index: int = 0,
                      legend: str | None = None) -> None:
    """Draw a published series as a thick grey band under our curves."""
    s = reference_series(panel, label, index)
    ax.plot(s["x"], s["y"], **PUBLISHED_STYLE, label=legend)


def draw_panel(panel: Panel, res: dict[str, np.ndarray]) -> tuple[plt.Figure, Axes]:
    fig, ax = plt.subplots(figsize=(4.6, 3.5), constrained_layout=True)
    panel(ax, res)
    ax.legend(fontsize=7, frameon=False)
    return fig, ax


def render(panels: dict[str, Panel], res: dict[str, np.ndarray], profile: str) -> list[Path]:
    """Save every panel as results/<panel>_<profile>.png."""
    RESULTS_DIR.mkdir(exist_ok=True)
    paths = []
    for name, panel in panels.items():
        fig, _ = draw_panel(panel, res)
        path = figure_path(name, profile)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def figure_path(panel: str, profile: str) -> Path:
    return RESULTS_DIR / f"{panel}_{profile}.png"


# ------------------------------------------------------------ sim series ----

def series_arrays(prefix: str, sweep: Any, fields: tuple[str, ...], rule: str) -> dict[str, np.ndarray]:
    """Mean, SEM (over separable trials) and separable counts of a SweepResult under ``prefix``.

    Points blanked by ``rule`` are NaN in both the mean and its SEM.
    """
    out = {f"{prefix}/n_separable": sweep.n_separable,
           f"{prefix}/trials": np.array(sweep.separable.shape[1])}
    for f in fields:
        m = sweep.mean(f, rule)
        out[f"{prefix}/{f}"] = m
        out[f"{prefix}/{f}_sem"] = np.where(np.isnan(m), np.nan, sweep.sem(f))
    return out


def errorbar(ax: Axes, x: np.ndarray, res: dict[str, np.ndarray], prefix: str, field: str,
             n_sem: float, **kw: Any) -> None:
    """Trial mean with error bars of ``n_sem`` standard errors (1 or 2 in the published figures)."""
    y = res[f"{prefix}/{field}"]
    if y.shape != np.shape(x):
        raise ValueError(f"{prefix}/{field}: {y.shape} values for {np.shape(x)} x positions")
    ax.errorbar(x, y, yerr=n_sem * res[f"{prefix}/{field}_sem"],
                fmt=kw.pop("fmt", ".-"), ms=4, capsize=2, **kw)


def oracle_line(ax: Axes, sigma_out: float) -> None:
    """Noise floor E_min of Eq. (4), the best any student can do."""
    ax.axhline(float(min_generalization_error(sigma_out)), color="k", ls="--", lw=1, label="oracle")


def alpha0_grid(alpha0s: np.ndarray, **fields: Any) -> tuple[TrialSpec, ...]:
    """One TrialSpec per alpha0, all other fields shared."""
    return tuple(TrialSpec(alpha0=float(a), **fields) for a in alpha0s)
