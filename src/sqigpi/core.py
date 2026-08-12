"""Linear-algebra core of scale-quotient petrophysical information geometry.

The functions in this module are intentionally small and dependency-light. They
implement the exact equations used in the manuscript and are suitable for unit
tests, pedagogical examples, and integration into larger inversion codes.
"""
from __future__ import annotations

import numpy as np


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError("expected a 2-D array")
    return a


def fisher_blocks(
    J_u: np.ndarray,
    J_eta: np.ndarray,
    C_d_inv: np.ndarray,
    C_eta_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the petrophysical, cross, and nuisance Fisher/GN blocks.

    Parameters
    ----------
    J_u
        Data Jacobian with respect to prior-whitened petrophysical coordinates.
    J_eta
        Data Jacobian with respect to nuisance coordinates.
    C_d_inv
        Inverse data/model-discrepancy covariance.
    C_eta_inv
        Optional inverse nuisance-prior covariance. If omitted, a zero matrix is
        used.
    """
    Ju = _as_2d(J_u)
    Je = _as_2d(J_eta)
    Cd = _as_2d(C_d_inv)
    if Ju.shape[0] != Je.shape[0] or Cd.shape != (Ju.shape[0], Ju.shape[0]):
        raise ValueError("incompatible Jacobian/covariance dimensions")
    if C_eta_inv is None:
        Ce = np.zeros((Je.shape[1], Je.shape[1]), dtype=float)
    else:
        Ce = _as_2d(C_eta_inv)
        if Ce.shape != (Je.shape[1], Je.shape[1]):
            raise ValueError("C_eta_inv has incompatible dimensions")

    G_uu = Ju.T @ Cd @ Ju
    G_ueta = Ju.T @ Cd @ Je
    G_etaeta = Je.T @ Cd @ Je + Ce
    return G_uu, G_ueta, G_etaeta


def horizontal_lift(
    du: np.ndarray,
    G_etau: np.ndarray,
    G_etaeta: np.ndarray,
) -> np.ndarray:
    """Return the nuisance adjustment that minimizes local information energy.

    Implements d eta* = -G_etaeta^{-1} G_etau du.
    """
    du = np.asarray(du, dtype=float).reshape(-1)
    Geu = _as_2d(G_etau)
    Gee = _as_2d(G_etaeta)
    if Geu.shape[1] != du.size or Gee.shape != (Geu.shape[0], Geu.shape[0]):
        raise ValueError("incompatible dimensions")
    return -np.linalg.solve(Gee, Geu @ du)


def quotient_metric(
    G_uu: np.ndarray,
    G_ueta: np.ndarray,
    G_etaeta: np.ndarray,
    *,
    symmetrize: bool = True,
) -> np.ndarray:
    """Profile nuisance directions from local petrophysical information.

    Returns
    -------
    G_Q = G_uu - G_ueta G_etaeta^{-1} G_etau
    """
    Guu = _as_2d(G_uu)
    Gue = _as_2d(G_ueta)
    Gee = _as_2d(G_etaeta)
    if Guu.shape[0] != Guu.shape[1]:
        raise ValueError("G_uu must be square")
    if Gue.shape[0] != Guu.shape[0] or Gee.shape != (Gue.shape[1], Gue.shape[1]):
        raise ValueError("incompatible block dimensions")
    # solve is numerically preferable to an explicit inverse
    Gq = Guu - Gue @ np.linalg.solve(Gee, Gue.T)
    if symmetrize:
        Gq = 0.5 * (Gq + Gq.T)
    return Gq


def authority_matrix(G_q: np.ndarray) -> np.ndarray:
    """Return local prior-relative authority A_Q = G_Q (I + G_Q)^{-1}."""
    Gq = _as_2d(G_q)
    if Gq.shape[0] != Gq.shape[1]:
        raise ValueError("G_q must be square")
    I = np.eye(Gq.shape[0])
    A = np.linalg.solve((I + Gq).T, Gq.T).T
    return 0.5 * (A + A.T)


def eigendirectional_authority(G_q: np.ndarray):
    """Return descending eigenvalues/eigenvectors and A_i=lambda_i/(1+lambda_i)."""
    Gq = _as_2d(G_q)
    lam, V = np.linalg.eigh(0.5 * (Gq + Gq.T))
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    V = V[:, order]
    A = lam / (1.0 + lam)
    return lam, V, A
