"""Compute and plot the figures of the paper, optionally in shards or on a GPU.

    python -m experiments.reproduce fig07 --profile quick --jobs 8
    python -m experiments.reproduce all --profile quick --jobs 8
    python -m experiments.reproduce fig07 --profile quick --plot-only

Sharding (e.g. one SLURM array task per shard; any machine, any order):

    python -m experiments.reproduce fig02_04 --profile paper --jobs 8 --shard 3/20
    python -m experiments.reproduce fig02_04 --profile paper --jobs 8 --merge 20

Trial jobs are dealt round robin to shards, each shard writes its per-trial
results to results/shards/, and --merge checks that every job of the current
plan is present exactly once before computing the figure. Seeds depend only on
(seed, stream, point, trial), so the merged result equals an unsharded run.

--device cpu (default, NumPy) | torch-cpu | cuda | cuda:k runs the learning
rules there (nn_expansion.backend). With a single GPU use --jobs 1 or a few workers.

Results: results/<figure>_<profile>.npz/.json (config, git state, device) and
panels results/<panel>_<profile>.png over the published data in grey.
"""
from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib

from nn_expansion.backend import Device
from nn_expansion.sweep import all_jobs, assemble, disposition, fingerprint, run_jobs, shard_jobs

from .common import RESULTS_DIR, load_config, load_results, provenance, render, save_results

FIGURES = {
    "fig02_04": "fig02_04_deterministic",
    "fig05": "fig05_capacity",
    "fig06": "fig06_theory",
    "fig07": "fig07_sigma_in",
    "fig08": "fig08_stochastic_vs_deterministic",
    "fig09_10": "fig09_10_quadratic_teacher",
    "fig11": "fig11_kernel",
    "fig12_14": "fig12_14_logistic",
    "slack": "slack_correspondence",
}


def module(name: str) -> ModuleType:
    return importlib.import_module(f"experiments.{FIGURES[name]}")


def shard_dir(mod: ModuleType, profile: str, fp: str) -> Path:
    return RESULTS_DIR / "shards" / f"{mod.NAME}_{profile}" / fp


def parse_shard(text: str) -> tuple[int, int]:
    k, n = (int(v) for v in text.split("/"))
    if not 0 <= k < n:
        raise argparse.ArgumentTypeError(f"--shard k/n needs 0 <= k < n, got {text}")
    return k, n


def run(name: str, profile: str, jobs: int, device: Device, plot_only: bool,
        shard: tuple[int, int] | None, merge: int | None) -> None:
    mod = module(name)
    cfg = load_config(mod.NAME, profile, mod.Config)
    if plot_only:
        res, _ = load_results(mod.NAME, profile)
        _plot(mod, res, cfg, profile)
        return
    if not hasattr(mod, "plan"):
        if shard or merge:
            raise SystemExit(f"{name} is a check without a trial plan; run it without --shard/--merge")
        t0 = time.time()
        res = mod.run_check(cfg, jobs)
        _save(mod, profile, cfg, res, t0, Device.cpu(), "whole")
        _plot(mod, res, cfg, profile)
        return

    plan = mod.plan(cfg)
    fp = fingerprint(plan)
    t0 = time.time()
    results: dict[Any, Any]
    if shard is not None:
        k, n = shard
        jobs_k = shard_jobs(plan, k, n)
        results = run_jobs(plan, jobs_k, jobs, device)
        out = shard_dir(mod, profile, fp) / f"shard_{k}of{n}.joblib"
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"fingerprint": fp, "shard": k, "n_shards": n, "device": device.name,
                     "results": results}, out)
        print(f"{name}: shard {k}/{n}: {len(jobs_k)} of {len(all_jobs(plan))} trials "
              f"in {time.time() - t0:.0f}s -> {out.relative_to(RESULTS_DIR.parent)}")
        return
    if merge is not None:
        results = {}
        for k in range(merge):
            path = shard_dir(mod, profile, fp) / f"shard_{k}of{merge}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"missing shard {k}/{merge} for plan {fp}: {path}")
            part = joblib.load(path)
            if part["fingerprint"] != fp or part["n_shards"] != merge or part["shard"] != k:
                raise ValueError(f"{path} does not belong to plan {fp}, shard {k}/{merge}")
            overlap = results.keys() & part["results"].keys()
            if overlap:
                raise ValueError(f"shard {k} repeats {len(overlap)} jobs")
            results.update(part["results"])
        how = f"merged {merge} shards"
    else:
        results = run_jobs(plan, all_jobs(plan), jobs, device)
        how = "whole"
    sims = assemble(plan, results)
    for line in disposition(sims):
        print(f"  {line}")
    res = mod.compute(cfg, sims)
    _save(mod, profile, cfg, res, t0, device, how, fp)
    _plot(mod, res, cfg, profile)


def _save(mod: ModuleType, profile: str, cfg: Any, res: dict, t0: float, device: Device,
          how: str, fp: str | None = None) -> None:
    meta = provenance(mod.NAME, profile, cfg, time.time() - t0)
    meta.update({"device": device.name, "execution": how, "plan_fingerprint": fp})
    path = save_results(mod.NAME, profile, res, meta)
    print(f"{mod.NAME}: {how} in {time.time() - t0:.0f}s -> {path.name}")


def _plot(mod: ModuleType, res: dict, cfg: Any, profile: str) -> None:
    for p in render(mod.panels(cfg), res, profile):
        print(f"  {p.relative_to(p.parent.parent)}")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")                     # files only; notebooks keep their own backend
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=[*FIGURES, "all"])
    ap.add_argument("--profile", required=True, choices=["paper", "quick"])
    ap.add_argument("--jobs", type=int, required=True, help="worker processes (does not change results)")
    ap.add_argument("--device", default="cpu", help="cpu (default) | torch-cpu | cuda | cuda:k")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plot-only", action="store_true", help="re-plot saved results")
    mode.add_argument("--shard", type=parse_shard, metavar="K/N", help="run shard K of N and store it")
    mode.add_argument("--merge", type=int, metavar="N", help="merge N stored shards and finish")
    args = ap.parse_args()
    device = Device.parse(args.device)
    names = list(FIGURES) if args.figure == "all" else [args.figure]
    for n in names:
        if (args.shard or args.merge) and not hasattr(module(n), "plan"):
            print(f"{n}: skipped (no trial plan; run it unsharded)")
            continue
        run(n, args.profile, args.jobs, device, args.plot_only, args.shard, args.merge)


if __name__ == "__main__":
    main()
