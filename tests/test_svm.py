"""Max-margin IPM and L2-slack SVM against closed forms and the KKT certificate."""
from __future__ import annotations

import numpy as np
import pytest

from nn_expansion.svm import NotSeparable, l2_slack_svm, max_margin


def test_single_pattern_closed_form() -> None:
    # min |w|^2/2 s.t. g.w >= 1  ->  w = g/|g|^2
    g = np.array([[3.0, 4.0]])
    np.testing.assert_allclose(max_margin(g, np.array([1.0])).W, [0.12, 0.16], atol=1e-9)


def test_orthogonal_patterns_closed_form() -> None:
    # orthogonal constraints decouple: w = sum_mu y_mu g_mu/|g_mu|^2
    Phi = np.array([[2.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    np.testing.assert_allclose(max_margin(Phi, np.array([1.0, -1.0])).W, [0.5, -0.2, 0.0], atol=1e-9)


def test_inseparable_pair_raises() -> None:
    with pytest.raises(NotSeparable):
        max_margin(np.array([[1.0, 2.0], [-1.0, -2.0]]), np.array([1.0, 1.0]))


def test_random_labels_above_gardner_capacity_raise() -> None:
    # P = 3N random labels: far above alpha_c = 2
    rng = np.random.default_rng(0)
    with pytest.raises(NotSeparable):
        max_margin(rng.standard_normal((150, 50)), np.where(rng.random(150) < 0.5, -1.0, 1.0))


@pytest.mark.parametrize("P,N", [(30, 10), (40, 25), (25, 60), (300, 100)])
def test_kkt_certificate(P: int, N: int) -> None:
    # For this convex QP the KKT conditions are sufficient for optimality, so they
    # certify the solution independently of how it was found.
    rng = np.random.default_rng(P * N)
    Phi = rng.standard_normal((P, N))
    y = np.where(Phi @ rng.standard_normal(N) >= 0, 1.0, -1.0)      # separable by construction
    res = max_margin(Phi, y)
    G = y[:, None] * Phi
    assert np.all(res.lam >= 0)
    np.testing.assert_allclose(G.T @ res.lam, res.W, atol=1e-6 * np.linalg.norm(res.W))
    assert np.min(G @ res.W) == pytest.approx(1.0, abs=1e-7)
    assert np.max(res.lam * (G @ res.W - 1.0)) < 1e-6
    assert res.lam.sum() == pytest.approx(res.W @ res.W, rel=1e-6)   # zero duality gap


def test_scale_and_permutation_invariance() -> None:
    rng = np.random.default_rng(3)
    Phi = rng.standard_normal((40, 20))
    y = np.where(Phi @ rng.standard_normal(20) >= 0, 1.0, -1.0)
    W = max_margin(Phi, y).W
    # The objective is 1-strongly convex, so |W - W*|^2 / 2 <= duality gap; with the
    # solver's relative gap 1e-8 the relative error is <= sqrt(2e-8) ~ 1.4e-4.
    tol = 2e-4 * np.linalg.norm(W)
    np.testing.assert_allclose(max_margin(7.0 * Phi, y).W, W / 7.0, atol=tol / 7.0)
    perm = rng.permutation(40)
    np.testing.assert_allclose(max_margin(Phi[perm], y[perm]).W, W, atol=tol)


def test_slack_svm_single_pattern_closed_form() -> None:
    # min w^2 + C xi^2 s.t. w.x >= 1 - xi, |x|^2 = a: w = C x/(1 + C a), xi = 1/(1 + C a)
    x = np.array([[1.0, 2.0, 2.0]])          # a = 9
    for C in (0.1, 1.0, 10.0):
        r = l2_slack_svm(x, np.array([1.0]), C)
        np.testing.assert_allclose(r.w, C * x[0] / (1 + 9 * C), rtol=1e-7)
        assert r.xi[0] == pytest.approx(1.0 / (1 + 9 * C), rel=1e-7)


def test_slack_svm_large_C_is_hard_margin() -> None:
    rng = np.random.default_rng(5)
    Phi = rng.standard_normal((30, 15))
    y = np.where(Phi @ rng.standard_normal(15) >= 0, 1.0, -1.0)
    W = max_margin(Phi, y).W
    w = l2_slack_svm(Phi, y, 1e8).w
    assert w @ W / (np.linalg.norm(w) * np.linalg.norm(W)) > 1 - 1e-6
