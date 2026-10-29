"""Planned sweeps of trials: run them (whole or in shards) and aggregate them.

A figure declares its simulations as a list of ``Series`` (a grid of TrialSpecs,
a trial count, a seed and a stream id). Every trial is a job (series key, point,
trial), and its random stream is derived from (seed, stream, point, trial)
through ``numpy.random.SeedSequence``. Results therefore do not depend on the
execution order, the number of worker processes, or how the jobs are split
into shards: a sharded run merged afterwards is identical to a single run.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from .backend import Device
from .simulate import TrialResult, TrialSpec, run_trial

SCALAR_FIELDS = ("R_pruned", "eg_pruned", "eg_unpruned", "kappa")
RULES = ("all_separable", "majority_separable", "separable_only")

Job = tuple[str, int, int]          # (series key, point index, trial index)


@dataclass(frozen=True)
class Series:
    key: str                         # unique within a figure
    specs: tuple[TrialSpec, ...]     # one per point of the grid
    trials: int
    seed: int
    stream: int                      # enters the seed derivation

    def __post_init__(self) -> None:
        if self.trials < 1:
            raise ValueError(f"series {self.key}: trials must be >= 1")


@dataclass(frozen=True)
class SweepResult:
    """Per-trial outcomes on a (points, trials) grid for one series."""

    separable: np.ndarray            # (n_points, trials) bool
    values: dict[str, np.ndarray]    # field -> (n_points, trials); NaN if not measured

    @property
    def n_separable(self) -> np.ndarray:
        return self.separable.sum(axis=1)

    def mean(self, name: str, rule: str) -> np.ndarray:
        """Trial mean of ``name`` per point over separable trials.

        rule = "all_separable": NaN unless every trial was separable (plain mean:
        one failed trial blanks the point; published Fig. 7a).
        rule = "majority_separable": NaN if more than half the trials failed, else
        the mean over separable trials (published Figs. 2-4, 8-11). Near capacity this conditions on
        separability, so the mean is over a selected subset of training sets.
        rule = "separable_only": NaN only if no trial was separable.
        """
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}")
        n = self.n_separable
        trials = self.separable.shape[1]
        v = np.where(self.separable, self.values[name], 0.0)
        if np.any(np.isnan(v)):
            raise ValueError(f"{name} was not measured on some separable trial")
        m = np.where(n > 0, v.sum(axis=1) / np.maximum(n, 1), np.nan)
        if rule == "all_separable":
            m = np.where(n == trials, m, np.nan)
        elif rule == "majority_separable":
            m = np.where(2 * n >= trials, m, np.nan)
        if m.shape != (self.separable.shape[0],):
            raise AssertionError(f"mean has shape {m.shape}, expected ({self.separable.shape[0]},)")
        return m

    def sem(self, name: str) -> np.ndarray:
        """Standard error of the mean over separable trials; NaN for n < 2."""
        n = self.n_separable
        v = np.where(self.separable, self.values[name], np.nan)
        out = np.full(n.shape, np.nan)
        for i in np.flatnonzero(n >= 2):
            row = v[i][self.separable[i]]
            out[i] = row.std(ddof=1) / np.sqrt(row.size)
        return out


@dataclass(frozen=True)
class SeriesResult:
    sweep: SweepResult
    raw: list[list[TrialResult]]     # [point][trial], e.g. for training traces


# ------------------------------------------------------------------ plans ----

def all_jobs(plan: list[Series]) -> list[Job]:
    keys = [s.key for s in plan]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate series keys in plan: {keys}")
    return [(s.key, i, t) for s in plan for i in range(len(s.specs)) for t in range(s.trials)]


def shard_jobs(plan: list[Series], shard: int, n_shards: int) -> list[Job]:
    """Jobs of shard ``shard`` of ``n_shards`` (round robin over the plan's job list)."""
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard must lie in 0..{n_shards - 1}, got {shard}")
    return all_jobs(plan)[shard::n_shards]


def fingerprint(plan: list[Series]) -> str:
    """Hash of everything that determines the trial results (not the device)."""
    h = hashlib.sha256()
    for s in plan:
        h.update(repr((s.key, s.specs, s.trials, s.seed, s.stream)).encode())
    return h.hexdigest()[:16]


def _one(spec: TrialSpec, seed_key: tuple[int, int, int, int], device: Device) -> TrialResult:
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_key)))
    return run_trial(spec, rng, device)


def run_jobs(plan: list[Series], jobs: Iterable[Job], n_jobs: int,
             device: Device) -> dict[Job, TrialResult]:
    """Run the given jobs of ``plan`` in parallel."""
    by_key = {s.key: s for s in plan}
    jobs = list(jobs)
    outs = Parallel(n_jobs=n_jobs)(
        delayed(_one)(by_key[k].specs[i], (by_key[k].seed, by_key[k].stream, i, t), device)
        for k, i, t in jobs)
    return dict(zip(jobs, outs))


def assemble(plan: list[Series], results: Mapping[Job, TrialResult]) -> dict[str, SeriesResult]:
    """Group per-job results by series; refuses missing or unexpected jobs."""
    expected = set(all_jobs(plan))
    missing, extra = expected - results.keys(), results.keys() - expected
    if missing or extra:
        raise ValueError(f"{len(missing)} jobs missing, {len(extra)} unexpected "
                         f"(e.g. {sorted(missing)[:3]} / {sorted(extra)[:3]})")
    out: dict[str, SeriesResult] = {}
    for s in plan:
        raw = [[results[(s.key, i, t)] for t in range(s.trials)] for i in range(len(s.specs))]
        sep = np.array([[r.separable for r in row] for row in raw], bool)
        vals = {f: np.array([[getattr(r, f) for r in row] for row in raw], float)
                for f in SCALAR_FIELDS}
        out[s.key] = SeriesResult(SweepResult(separable=sep, values=vals), raw)
    return out


def disposition(sims: Mapping[str, SeriesResult]) -> list[str]:
    """Per series: points, trials, and how many points each averaging rule blanks."""
    lines = []
    for key, r in sims.items():
        n, trials = r.sweep.n_separable, r.sweep.separable.shape[1]
        lines.append(f"{key}: {n.size} points x {trials} trials, {int(n.sum())}/{n.size * trials} "
                     f"separable; blanked: all_separable {int(np.sum(n < trials))}, "
                     f"majority_separable {int(np.sum(2 * n < trials))}, separable_only {int(np.sum(n == 0))}")
    return lines


def run_series(specs: list[TrialSpec], trials: int, seed: int, series_id: int, n_jobs: int,
               device: Device = Device.cpu()) -> tuple[SweepResult, list[list[TrialResult]]]:
    """Run one series in full (convenience for notebooks and tests)."""
    plan = [Series(key="s", specs=tuple(specs), trials=trials, seed=seed, stream=series_id)]
    res = assemble(plan, run_jobs(plan, all_jobs(plan), n_jobs, device))["s"]
    return res.sweep, res.raw
