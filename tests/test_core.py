import numpy as np

from sqigpi import (
    authority_matrix,
    eigendirectional_authority,
    horizontal_lift,
    quotient_metric,
)


def random_spd(rng, n, shift=0.5):
    A = rng.normal(size=(n, n))
    return A.T @ A + shift * np.eye(n)


def test_horizontal_lift_energy_equals_schur_complement():
    rng = np.random.default_rng(20260812)
    p, q = 4, 2
    Guu = random_spd(rng, p, 2.0)
    Gee = random_spd(rng, q, 1.0)
    Gue = rng.normal(size=(p, q)) * 0.15
    # Make the full block SPD by keeping the cross block modest.
    Gq = quotient_metric(Guu, Gue, Gee)
    du = rng.normal(size=p)
    de = horizontal_lift(du, Gue.T, Gee)
    full = np.block([[Guu, Gue], [Gue.T, Gee]])
    d = np.r_[du, de]
    assert np.allclose(d @ full @ d, du @ Gq @ du, rtol=1e-11, atol=1e-11)


def test_nuisance_reparameterization_invariance():
    rng = np.random.default_rng(7)
    Guu = random_spd(rng, 3, 2.0)
    Gee = random_spd(rng, 2, 1.0)
    Gue = rng.normal(size=(3, 2)) * 0.1
    T = np.array([[1.4, 0.3], [-0.2, 0.8]])
    Gq = quotient_metric(Guu, Gue, Gee)
    Gue2 = Gue @ T
    Gee2 = T.T @ Gee @ T
    Gq2 = quotient_metric(Guu, Gue2, Gee2)
    assert np.allclose(Gq, Gq2, rtol=1e-11, atol=1e-11)


def test_information_can_only_decrease_after_profiling():
    rng = np.random.default_rng(18)
    Guu = random_spd(rng, 3, 3.0)
    Gee = random_spd(rng, 2, 1.0)
    Gue = rng.normal(size=(3, 2)) * 0.12
    Gq = quotient_metric(Guu, Gue, Gee)
    assert np.min(np.linalg.eigvalsh(Gq)) > -1e-10
    assert np.min(np.linalg.eigvalsh(Guu - Gq)) > -1e-10


def test_authority_matches_eigendirectional_formula():
    rng = np.random.default_rng(2)
    Gq = random_spd(rng, 3, 0.2)
    A = authority_matrix(Gq)
    lam, V, ai = eigendirectional_authority(Gq)
    assert np.allclose(V.T @ A @ V, np.diag(ai), rtol=1e-11, atol=1e-11)
