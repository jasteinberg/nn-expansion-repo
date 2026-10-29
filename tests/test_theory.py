"""Replica theory against limits, internal reductions, and the published curves."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from nn_expansion import theory as T
from nn_expansion.teacher import min_generalization_error

REF = Path(__file__).resolve().parent.parent / "reference" / "published"


def _series(panel: str, label: str, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    s = [s for s in json.loads((REF / f"{panel}.json").read_text())["series"]
         if s["label"] == label][index]
    return np.array(s["x"], float), np.array(s["y"], float)


def test_eg_at_R1_is_noise_floor() -> None:
    # Eq. (23) at R = 1 equals Eq. (4) (different closed forms)
    for so in (0.1, 0.25, 0.5, 1.0, 2.0):
        assert T.eg_from_overlap(1.0, so) == pytest.approx(min_generalization_error(so), abs=1e-14)


def test_eg_at_R0_is_chance() -> None:
    assert T.eg_from_overlap(0.0, 0.7) == pytest.approx(0.5, abs=1e-15)


def test_J_closed_form() -> None:
    for k in (0.0, 0.3, 1.7):
        val, _ = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi) * (k + x) ** 2 * x, -k, np.inf)
        assert T._J(k) == pytest.approx(val, rel=1e-10)


def test_capacity_gardner_limit() -> None:
    # random labels (sigma_out -> inf): alpha_c -> 2 (Gardner)
    assert T.capacity_alpha(1e3) == pytest.approx(2.0, abs=1e-4)


def test_capacity_scaling_with_beta() -> None:
    assert T.capacity_alpha0(7.0, 0.25) == pytest.approx(7.0 * T.capacity_alpha0(1.0, 0.25), rel=1e-14)


def test_saddle_tends_to_capacity() -> None:
    so = 0.5
    ac = T.capacity_alpha(so)
    sp = T.saddle_sigma_in_1(0.99999 * ac, so)
    assert sp.kappa < 1e-3
    assert sp.m == pytest.approx(T.capacity_m_bar(so) * np.sqrt(1 + so ** 2), abs=1e-3)
    with pytest.raises(T.AboveCapacity):
        T.saddle_sigma_in_1(1.0001 * ac, so)


def test_large_beta_limit() -> None:
    # exact saddle at large beta approaches Eq. (E8) with 1/(1 + sigma_out^2)
    for so, a0 in ((1.0, 8.0), (0.5, 4.0)):
        beta = 2000.0
        R = T.overlap_pruned_sigma_in_1(T.saddle_sigma_in_1(a0 / beta, so).m, beta)
        assert R ** 2 == pytest.approx(T.large_beta_R2(a0, so), abs=2e-3)


def test_general_saddle_reduces_to_sigma_in_1() -> None:
    # at sigma_in = 1 the sigma_in = 1 solution with r_tilde from (B15) solves all three equations
    for a0, beta, so in ((5.0, 5.0, 0.25), (3.0, 3.0, 1.0)):
        sp = T.saddle_sigma_in_1(a0 / beta, so)
        res = T._general_residuals(np.array([sp.kappa, sp.m, T.rtilde_sigma_in_1(sp.m, beta)]),
                                   a0, beta, so, 1.0)
        assert np.max(np.abs(res)) < 1e-9


def test_fig05_capacity_matches_published() -> None:
    for b in (1, 3, 5):
        x, y = _series("fig05", rf"\beta={b}")
        th = np.array([T.capacity_alpha0(b, 1 / np.sqrt(v)) for v in x])
        np.testing.assert_allclose(th, y, rtol=5e-5)


@pytest.mark.parametrize("panel,so", [("fig06a", 0.25), ("fig06b", 1.0)])
def test_fig06ab_matches_published(panel: str, so: float) -> None:
    for b in (1, 5, 10):
        x, y = _series(panel, rf"\beta={b}")
        th = T.eg_curve_sigma_in_1(x, b, so, pruned=True)
        np.testing.assert_array_equal(np.isnan(th), np.isnan(y))      # same capacity cut
        ok = ~np.isnan(y)
        # 2e-4: the published curves were solved to saddle residuals of up to 5e-4
        np.testing.assert_allclose(th[ok], y[ok], atol=2e-4)
    x, y = _series(panel, r"\beta=\infty")
    ok = x > 0
    np.testing.assert_allclose(T.large_beta_eg(x[ok], so), y[ok], atol=1e-12)


def test_fig07b_matches_published() -> None:
    x, y = _series("fig07b", "", 0)
    sw = T.saddle_sigma_in_sweep(3.0, 5.0, 1.0, x)
    np.testing.assert_allclose([T.eg_from_overlap(p.R_pruned, 1.0) for p in sw], y, atol=1e-4)
    s_opt, e_opt = T.optimal_sigma_in(3.0, 5.0, 1.0, (0.2, 1.0))
    xo, yo = _series("fig07b", "", 1)
    assert s_opt == pytest.approx(xo[0], abs=1e-3)
    assert e_opt == pytest.approx(yo[0], abs=1e-5)


def test_fig07a_theory_matches_published() -> None:
    # sigma_in = 0.4 curve; the published sigma_in = 0.6 curve has one point
    # (alpha0 = 5.34) off its own smooth curve, so 0.4 is used here.
    x, y = _series("fig07a", "theory", 2)
    ok = ~np.isnan(y)
    sps = T.saddle_alpha0_sweep(x[ok][::4], 5.0, 0.1, 0.4)
    th = [T.eg_from_overlap(p.R_pruned, 0.1) for p in sps]
    np.testing.assert_allclose(th, y[ok][::4], atol=1e-4)
