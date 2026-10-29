"""Teacher, expansions, measurements, logistic GD, sweeps and configs."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from experiments import reproduce
from experiments.common import load_config
from nn_expansion.expansion import (design_matrix, fit_deterministic_expansion, quadratic_features,
                               stochastic_units)
from nn_expansion.logistic import effective_eta, logistic_gd
from nn_expansion.simulate import (Deterministic, Logistic, MaxMargin, NoExpansion, Stochastic,
                              TrialSpec, _eg, run_trial)
from nn_expansion.sweep import run_series
from nn_expansion.teacher import draw_inputs, draw_teacher, min_generalization_error
from nn_expansion.theory import saddle_sigma_in_1


def _mc_tol(p: float, n: int) -> float:
    return 4.0 * np.sqrt(p * (1 - p) / n)        # 4 binomial standard errors


def test_teacher_as_student_hits_noise_floor() -> None:
    # Eq. (4): the teacher's own weights err at E_min against its noisy labels
    rng = np.random.default_rng(0)
    so, n = 0.5, 200_000
    t = draw_teacher(50, 1.0, so, rng)
    X = draw_inputs(n, 50, rng)
    err = np.mean(np.where(X @ t.w0 >= 0, 1.0, -1.0) != t.labels(X, rng))
    emin = float(min_generalization_error(so))
    assert abs(err - emin) < _mc_tol(emin, n)


def test_quadratic_teacher_error_of_w0() -> None:
    # Against the quadratic teacher (sigma_out = 0) the linear rule sgn(w0.x) errs
    # when the quadratic part q flips the sign of the linear part s. For this
    # teacher, s ~ N(0,1) and q = x^T W0 x / N0 ~ N(b, v) with b = tr(W0)/N0 and
    # v = 2 tr(W_sym^2)/N0^2, independent of s up to O(1/N0). Eq. (32) is the b = 0,
    # v = 1 case: E = E_min((1-a)/a). Check MC against the 1-D quadrature for b, v.
    from scipy.integrate import quad
    from scipy.special import ndtr
    rng = np.random.default_rng(1)
    a, N0, n = 0.5, 200, 200_000
    t = draw_teacher(N0, a, 0.0, rng)
    Ws = 0.5 * (t.W0 + t.W0.T)
    b, v = np.trace(t.W0) / N0, 2 * np.trace(Ws @ Ws) / N0 ** 2
    c = (1 - a) / a
    phi = lambda s: np.exp(-s * s / 2) / np.sqrt(2 * np.pi)            # noqa: E731
    # P(s + c q < 0 | s > 0) and P(s + c q > 0 | s < 0)
    expected = quad(lambda s: phi(s) * ndtr((-s / c - b) / np.sqrt(v)), 0, np.inf)[0] \
        + quad(lambda s: phi(s) * ndtr((s / c + b) / np.sqrt(v)), -np.inf, 0)[0]
    X = draw_inputs(n, N0, rng)
    err = np.mean(np.where(X @ t.w0 >= 0, 1.0, -1.0) != t.labels(X, rng))
    assert abs(err - expected) < _mc_tol(expected, n) + 2.0 / N0      # + O(1/N0) s-q correlation
    assert expected == pytest.approx(float(min_generalization_error(c)), abs=0.03)


def test_deterministic_expansion_statistics() -> None:
    # threshold sets the sparsity; units have mean ~0 and variance ~A^2 (Eq. 6)
    rng = np.random.default_rng(2)
    X = draw_inputs(400, 100, rng)
    for f, A in ((0.02, 0.2), (0.5, 1.0)):
        det = fit_deterministic_expansion(X, 1000, f, A, rng)
        Z = det.transform(draw_inputs(400, 100, rng))
        assert np.mean(det.preactivation(X) > det.T) == pytest.approx(f, abs=0.005)
        assert np.mean(Z) == pytest.approx(0.0, abs=0.02 * A)
        assert np.var(Z) == pytest.approx(A ** 2, rel=0.1)


def test_quadratic_features_give_squared_kernel() -> None:
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((5, 7))
    F = quadratic_features(Z)
    np.testing.assert_allclose(F @ F.T, (Z @ Z.T) ** 2 / 49, rtol=1e-12)


@pytest.mark.parametrize("expansion", [Stochastic(3.0, 0.7), NoExpansion()])
def test_exact_eg_matches_monte_carlo(expansion) -> None:
    # pruned and stochastic-unpruned errors are computed exactly; check against MC
    rng = np.random.default_rng(4)
    spec = TrialSpec(N0=30, alpha0=2.0, teacher_a=1.0, sigma_out=0.5, expansion=expansion,
                     learner=MaxMargin(), student="linear", measure_unpruned=True, n_test=200_000)
    teacher = draw_teacher(30, 1.0, 0.5, rng)
    W = rng.standard_normal(spec.N) + 0.5 * np.concatenate([teacher.w0, np.zeros(spec.N - 30)])
    for pruned in (True, False):
        exact = _eg(spec, teacher, W, rng, pruned=pruned, det=None)
        Xt = draw_inputs(spec.n_test, 30, rng)
        yt = teacher.labels(Xt, rng)
        U = None
        if isinstance(expansion, Stochastic):
            U = (np.zeros((spec.n_test, spec.N - 30)) if pruned
                 else stochastic_units(spec.n_test, spec.N - 30, expansion.sigma_in, rng))
        mc = np.mean(np.where(design_matrix(Xt, U) @ W >= 0, 1.0, -1.0) != yt)
        assert abs(exact - mc) < _mc_tol(exact, spec.n_test)


def test_logistic_first_step_is_hebbian() -> None:
    # from W = 0, one step of size eta is (eta/2) sum_mu y_mu Phi_mu
    rng = np.random.default_rng(5)
    Phi = rng.standard_normal((20, 8))
    y = np.where(rng.random(20) < 0.5, -1.0, 1.0)
    W, _, _ = logistic_gd(Phi, y, eta=0.3, epochs=1, W_init=np.zeros(8), stop_tol=0.0)
    np.testing.assert_allclose(W, 0.15 * (y @ Phi), rtol=1e-12)


def test_logistic_step_is_gradient_descent() -> None:
    # finite-difference gradient of L = sum log(1 + exp(-y W.Phi))
    rng = np.random.default_rng(6)
    Phi = rng.standard_normal((15, 4))
    y = np.where(rng.random(15) < 0.5, -1.0, 1.0)
    W1, _, _ = logistic_gd(Phi, y, eta=0.1, epochs=1, W_init=np.zeros(4), stop_tol=0.0)
    W2, _, _ = logistic_gd(Phi, y, eta=0.1, epochs=2, W_init=np.zeros(4), stop_tol=0.0)
    L = lambda W: np.sum(np.log1p(np.exp(-y * (Phi @ W))))          # noqa: E731
    h = 1e-6
    grad = np.array([(L(W1 + h * e) - L(W1 - h * e)) / (2 * h) for e in np.eye(4)])
    np.testing.assert_allclose(W2, W1 - 0.1 * grad, atol=1e-8)


def test_published_eta_unit_reproduces_published_update() -> None:
    # update of the published runs, written out: h = w X / sqrt(N),
    # dw = (s - 1/(1+exp(-h))) X^T, w = w + lr dw, with s = (y + 1)/2
    rng = np.random.default_rng(7)
    P, N, lr = 30, 9, 0.05
    X = rng.standard_normal((P, N))
    y = np.where(rng.random(P) < 0.5, -1.0, 1.0)
    w = 0.1 * rng.standard_normal(N) / np.sqrt(N)
    W_init = w.copy()
    s = (y > 0).astype(float)
    for _ in range(5):
        h = X @ w / np.sqrt(N)
        w = w + lr * ((s - 1.0 / (1.0 + np.exp(-h))) @ X)
    W, _, _ = logistic_gd(X / np.sqrt(N), y, effective_eta(lr, "published", N), 5, W_init, 0.0)
    np.testing.assert_allclose(W, w, rtol=1e-12)


def test_logistic_early_stop_holds_trace() -> None:
    # separable data with a huge stop tolerance: stops after epoch 1, trace repeats
    Phi = np.array([[1.0, 0.0], [0.0, 1.0]])
    W, ts, vals = logistic_gd(Phi, np.array([1.0, 1.0]), 0.1, 50, np.zeros(2), stop_tol=1e3,
                              record_every=10, record=lambda W: (float(W[0]),))
    np.testing.assert_array_equal(ts, [1, 11, 21, 31, 41])
    assert len(vals) == 5 and all(v == vals[0] for v in vals)
    np.testing.assert_allclose(W, [0.05, 0.05])


def test_max_margin_simulation_matches_replica() -> None:
    # beta = 1, sigma_out = 0.25, alpha0 = 4: finite-size N0 = 100 against the
    # replica overlap. 0.03 covers the O(1/sqrt(N)) trial spread over 6 trials.
    specs = [TrialSpec(N0=100, alpha0=4.0, teacher_a=1.0, sigma_out=0.25, expansion=NoExpansion(),
                       learner=MaxMargin(), student="linear", measure_unpruned=False, n_test=0)]
    sw, _ = run_series(specs, 6, 0, 0, 1)
    assert sw.separable.all()
    m = saddle_sigma_in_1(4.0, 0.25).m
    assert sw.mean("R_pruned", "all_separable")[0] == pytest.approx(m, abs=0.03)
    assert sw.mean("kappa", "all_separable")[0] == pytest.approx(
        saddle_sigma_in_1(4.0, 0.25).kappa, abs=0.05)


def test_sweep_is_seed_deterministic_and_job_count_invariant() -> None:
    specs = [TrialSpec(N0=20, alpha0=a, teacher_a=1.0, sigma_out=0.5, expansion=Deterministic(3.0, 0.1, 1.0),
                       learner=MaxMargin(), student="linear", measure_unpruned=True, n_test=500)
             for a in (1.0, 2.0)]
    a, _ = run_series(specs, 3, 11, 0, 1)
    b, _ = run_series(specs, 3, 11, 0, 2)
    c, _ = run_series(specs, 3, 12, 0, 1)
    for k in ("eg_pruned", "eg_unpruned"):
        np.testing.assert_array_equal(a.values[k], b.values[k])
    assert not np.array_equal(a.values["eg_pruned"], c.values["eg_pruned"])


def test_mean_rules() -> None:
    from nn_expansion.sweep import SweepResult
    sep = np.array([[True, True, True], [True, True, False], [True, False, False], [False, False, False]])
    v = np.array([[0.1, 0.3, 0.2], [0.2, 0.4, np.nan], [0.5, np.nan, np.nan], [np.nan] * 3])
    sw = SweepResult(separable=sep, values={"eg_pruned": v})
    np.testing.assert_allclose(sw.mean("eg_pruned", "all_separable"), [0.2, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(sw.mean("eg_pruned", "majority_separable"), [0.2, 0.3, np.nan, np.nan])
    np.testing.assert_allclose(sw.mean("eg_pruned", "separable_only"), [0.2, 0.3, 0.5, np.nan])


def test_dead_hidden_layer_raises() -> None:
    with pytest.raises(ValueError, match="dead"):
        fit_deterministic_expansion(np.zeros((10, 5)), 20, 0.1, 1.0, np.random.default_rng(0))


def test_kkt_check_rejects_a_wrong_solution() -> None:
    from nn_expansion.svm import MaxMarginResult, SolverFailure, _check_kkt, max_margin
    rng = np.random.default_rng(10)
    Phi = rng.standard_normal((30, 12))
    y = np.where(Phi @ rng.standard_normal(12) >= 0, 1.0, -1.0)
    good = max_margin(Phi, y)
    with pytest.raises(SolverFailure):
        _check_kkt(y[:, None] * Phi, MaxMarginResult(good.W * 0.9, good.lam, 1, 0.0), 1e-8)


def test_trial_above_capacity_is_reported() -> None:
    spec = TrialSpec(N0=30, alpha0=12.0, teacher_a=1.0, sigma_out=1.0, expansion=NoExpansion(),
                     learner=MaxMargin(), student="linear", measure_unpruned=False, n_test=0)
    assert not run_trial(spec, np.random.default_rng(0)).separable


def test_logistic_trial_runs_above_capacity() -> None:
    spec = TrialSpec(N0=30, alpha0=12.0, teacher_a=1.0, sigma_out=1.0, expansion=NoExpansion(),
                     learner=Logistic(0.05, "paper", 200, 0.0, 0.0, None), student="linear", measure_unpruned=False, n_test=0)
    r = run_trial(spec, np.random.default_rng(0))
    assert r.separable and 0.5 < r.R_pruned < 1.0


@pytest.mark.parametrize("name", list(reproduce.FIGURES))
def test_configs_load_strictly(name: str) -> None:
    mod = reproduce.module(name)
    for profile in ("paper", "quick"):
        cfg = load_config(mod.NAME, profile, mod.Config)
        assert dataclasses.is_dataclass(cfg)


def test_config_profiles_are_strict(tmp_path, monkeypatch) -> None:
    from experiments import common
    cls = reproduce.module("fig05").Config
    paper = "[paper]\ninv_sigma2 = [0.01, 8.0, 30]\nbetas = [1.0, 3.0]\n"
    monkeypatch.setattr(common, "CONFIG_DIR", tmp_path)
    cases = {
        "typo in override": paper + "[quick]\nbetass = [2.0]\n",
        "override equal to paper": paper + "[quick]\nbetas = [1.0, 3.0]\n",
        "unknown key in paper": paper.replace("[paper]\n", "[paper]\nextra = 1\n") + "[quick]\n",
        "missing key in paper": "[paper]\nbetas = [1.0]\n[quick]\n",
    }
    for what, text in cases.items():
        (tmp_path / "fig05.toml").write_text(text)
        with pytest.raises(KeyError):
            load_config("fig05", "quick", cls)
            pytest.fail(what)
    (tmp_path / "fig05.toml").write_text(paper + "[quick]\nbetas = [5.0]\n")
    cfg = load_config("fig05", "quick", cls)
    assert cfg.betas == [5.0] and cfg.inv_sigma2 == [0.01, 8.0, 30]


def test_show_draws_one_panel(tmp_path, monkeypatch) -> None:
    import shutil

    from experiments import common, notebook
    from experiments.common import CONFIG_DIR
    shutil.copytree(CONFIG_DIR, tmp_path / "configs")
    (tmp_path / "configs" / "fig11.toml").write_text(
        "[paper]\nN0 = 6\nbeta = 2.0\nsigma_out = 1.0\nalpha0_grid = [0.5, 4.0, 3]\n"
        "trials = 2\nseed = 1\nn_test = 200\nrule = \"majority_separable\"\nn_sem = 2.0\n[quick]\n")
    for mod in (common, reproduce, notebook):
        monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path / "results", raising=False)
    monkeypatch.setattr(common, "CONFIG_DIR", tmp_path / "configs")
    from nn_expansion.backend import Device
    reproduce.run("fig11", "paper", 1, Device.cpu(), False, None, None)
    assert "fig11b" in notebook.list_panels("paper")
    fig = notebook.show("fig11b", "paper")
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert {"pruned", "unpruned", "oracle"} <= set(labels)
