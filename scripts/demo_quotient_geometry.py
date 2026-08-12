#!/usr/bin/env python3
"""Small no-data demonstration of the scale-quotient metric."""
from __future__ import annotations

import numpy as np
from sqigpi import authority_matrix, eigendirectional_authority, horizontal_lift, quotient_metric

rng = np.random.default_rng(20260812)
A = rng.normal(size=(3, 3))
Guu = A.T @ A + 3.0 * np.eye(3)
B = rng.normal(size=(2, 2))
Gee = B.T @ B + np.eye(2)
Gue = 0.12 * rng.normal(size=(3, 2))
Gq = quotient_metric(Guu, Gue, Gee)
lam, V, ai = eigendirectional_authority(Gq)

# Horizontal-lift identity for one perturbation.
du = rng.normal(size=3)
de = horizontal_lift(du, Gue.T, Gee)
Gjoint = np.block([[Guu, Gue], [Gue.T, Gee]])
err = abs(np.r_[du, de] @ Gjoint @ np.r_[du, de] - du @ Gq @ du)

# Nuisance reparameterization.
T = np.array([[1.2, 0.25], [-0.1, 0.85]])
Gq2 = quotient_metric(Guu, Gue @ T, T.T @ Gee @ T)
inv_err = np.linalg.norm(Gq - Gq2) / np.linalg.norm(Gq)

print("SQ-IGPI core verification")
print("eigenvalues G_Q:", np.array2string(lam, precision=6))
print("directional authority:", np.array2string(ai, precision=6))
print("property-axis authority diag:", np.array2string(np.diag(authority_matrix(Gq)), precision=6))
print(f"horizontal-lift energy error: {err:.3e}")
print(f"nuisance-reparameterization relative error: {inv_err:.3e}")
print("min eig(G_Q):", f"{np.min(np.linalg.eigvalsh(Gq)):.6e}")
print("min eig(G_uu-G_Q):", f"{np.min(np.linalg.eigvalsh(Guu-Gq)):.6e}")
