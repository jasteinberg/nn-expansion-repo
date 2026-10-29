"""Sharded execution equals a single run; the torch backend agrees with NumPy."""
from __future__ import annotations

import numpy as np
import pytest

from nn_expansion.backend import Device
from nn_expansion.logistic import logistic_gd
from nn_expansion.simulate import Deterministic, Logistic, MaxMargin, NoExpansion, TrialSpec
from nn_expansion.svm import NotSeparable, max_margin
from nn_expansion.sweep import Series, all_jobs, assemble, fingerprint, run_jobs, shard_jobs


def _plan() -> list[Series]:
    mm = [TrialSpec(N0=15, alpha0=a, teacher_a=1.0, sigma_out=0.5, expansion=Deterministic(3.0, 0.1, 1.0),
                    learner=MaxMargin(), student="linear", measure_unpruned=True, n_test=300)
          for a in (1.0, 2.0, 6.0)]
    lg = [TrialSpec(N0=15, alpha0=4.0, teacher_a=1.0, sigma_out=0.5, expansion=NoExpansion(),
                    learner=Logistic(0.05, "published", 30, 0.1, 1e-6, 10), student="linear",
                    measure_unpruned=False, n_test=0)]
    return [Series("mm", tuple(mm), 3, 4, 0), Series("lg", tuple(lg), 2, 4, 1)]


def _same(a, b) -> None:
    for key in a:
        np.testing.assert_array_equal(a[key].sweep.separable, b[key].sweep.separable)
        for f, v in a[key].sweep.values.items():
            np.testing.assert_array_equal(v, b[key].sweep.values[f])
        for ra, rb in zip(sum(a[key].raw, []), sum(b[key].raw, [])):
            np.testing.assert_array_equal(ra.trace_R, rb.trace_R)


def test_shards_partition_the_jobs() -> None:
    plan = _plan()
    parts = [shard_jobs(plan, k, 4) for k in range(4)]
    flat = [j for p in parts for j in p]
    assert sorted(flat) == sorted(all_jobs(plan)) and len(flat) == len(set(flat))


def test_merged_shards_equal_single_run() -> None:
    plan = _plan()
    whole = assemble(plan, run_jobs(plan, all_jobs(plan), 1, Device.cpu()))
    merged: dict = {}
    for k in range(3):
        merged.update(run_jobs(plan, shard_jobs(plan, k, 3), 2, Device.cpu()))
    _same(whole, assemble(plan, merged))


def test_assemble_refuses_incomplete_results() -> None:
    plan = _plan()
    partial = run_jobs(plan, shard_jobs(plan, 0, 2), 1, Device.cpu())
    with pytest.raises(ValueError, match="missing"):
        assemble(plan, partial)


def test_fingerprint_tracks_the_plan() -> None:
    plan = _plan()
    changed = [Series("mm", plan[0].specs, 4, 4, 0), plan[1]]          # one more trial
    assert fingerprint(plan) == fingerprint(_plan())
    assert fingerprint(plan) != fingerprint(changed)


def test_device_names() -> None:
    assert Device.parse("cpu") == Device.cpu()
    with pytest.raises(ValueError):
        Device.parse("mps")
    with pytest.raises(ValueError):
        Device.parse("gpu")


def test_cuda_refused_when_unavailable() -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA present: nothing to refuse")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        Device.parse("cuda")


def _gpu_or_torch_cpu() -> Device:
    torch = pytest.importorskip("torch")
    return Device.parse("cuda" if torch.cuda.is_available() else "torch-cpu")


def test_torch_max_margin_matches_numpy() -> None:
    dev = _gpu_or_torch_cpu()
    rng = np.random.default_rng(8)
    Phi = rng.standard_normal((120, 40))
    y = np.where(Phi @ rng.standard_normal(40) + 0.3 * rng.standard_normal(120) >= 0, 1.0, -1.0)
    a, b = max_margin(Phi, y), max_margin(Phi, y, device=dev)
    # both stop at relative duality gap 1e-8 -> relative weight error <= ~1.4e-4 each
    np.testing.assert_allclose(b.W, a.W, atol=3e-4 * np.linalg.norm(a.W))
    with pytest.raises(NotSeparable):
        max_margin(rng.standard_normal((150, 50)), np.where(rng.random(150) < 0.5, -1.0, 1.0), device=dev)


def test_torch_logistic_matches_numpy() -> None:
    dev = _gpu_or_torch_cpu()
    rng = np.random.default_rng(9)
    Phi = rng.standard_normal((60, 20)) / np.sqrt(20)
    y = np.where(rng.random(60) < 0.5, -1.0, 1.0)
    W0 = 0.1 * rng.standard_normal(20)
    a, ta, va = logistic_gd(Phi, y, 0.3, 200, W0, 0.0, 50, lambda W: (float(W[0]),))
    b, tb, vb = logistic_gd(Phi, y, 0.3, 200, W0, 0.0, 50, lambda W: (float(W[0]),), dev)
    np.testing.assert_allclose(b, a, rtol=1e-10)
    np.testing.assert_array_equal(ta, tb)
    np.testing.assert_allclose(vb, va, rtol=1e-10)


def test_cli_shard_merge_equals_whole(tmp_path, monkeypatch) -> None:
    from experiments import common, reproduce
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "fig11.toml").write_text(
        "[paper]\nN0 = 6\nbeta = 2.0\nsigma_out = 1.0\nalpha0_grid = [0.5, 4.0, 3]\n"
        "trials = 2\nseed = 1\nn_test = 200\nrule = \"majority_separable\"\nn_sem = 2.0\n")
    monkeypatch.setattr(common, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(reproduce, "RESULTS_DIR", tmp_path / "results")
    cpu = Device.cpu()
    reproduce.run("fig11", "paper", 1, cpu, False, None, None)
    whole, _ = common.load_results("fig11", "paper")
    for k in range(3):
        reproduce.run("fig11", "paper", 1, cpu, False, (k, 3), None)
    reproduce.run("fig11", "paper", 1, cpu, False, None, 3)
    merged, meta = common.load_results("fig11", "paper")
    assert meta["execution"] == "merged 3 shards"
    assert whole.keys() == merged.keys()
    for key in whole:
        np.testing.assert_array_equal(whole[key], merged[key])
