"""Controlled Volve-derived forward model used for reviewer reproduction.

This module contains the compact nonlinear forward model used in the manuscript's
stress test. It intentionally depends only on the small ASCII tables distributed
with this repository. Raw Volve seismic files are not required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

from .core import quotient_metric, eigendirectional_authority


@dataclass(frozen=True)
class BenchmarkConfig:
    prior_scales: np.ndarray
    u_true: np.ndarray
    eta_prior_sd: float
    angles_deg: np.ndarray
    rho_vertical: float
    rho_angle: float
    sigma_noise: float
    tau: float = 2.85022522
    nominal_scale_m: float = 4.0
    wavelet_hz: float = 22.0


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(root: Path | None = None) -> BenchmarkConfig:
    root = repository_root() if root is None else Path(root)
    meta = json.loads((root / "data/compact/benchmark_metadata.json").read_text())
    angles = np.linspace(-25.0, 25.0, 50)[25:]
    return BenchmarkConfig(
        prior_scales=np.asarray(meta["prior_scales_phi_vsh_sw"], float),
        u_true=np.asarray(meta["controlled_truth_u"], float),
        eta_prior_sd=float(meta["eta_prior_sd"]),
        angles_deg=angles,
        rho_vertical=float(meta["vertical_noise_lag1"]),
        rho_angle=float(meta["angular_noise_lag1"]),
        sigma_noise=float(meta["noise_sigma"]),
    )


def load_training_data(root: Path | None = None):
    root = repository_root() if root is None else Path(root)
    frames = []
    for well in ("19A", "BT2"):
        frames.append(pd.read_csv(root / f"data/compact/{well}_training_window.csv"))
    d = pd.concat(frames, ignore_index=True)
    X = d[["phi", "vsh", "sw"]].to_numpy(float)
    Y = d[["vp_mps", "vs_mps", "rho_gcc"]].to_numpy(float)
    return X, Y


def fit_reference_surrogate(root: Path | None = None, degree: int = 3, alpha: float = 0.1):
    """Fit the paper's reference polynomial/Ridge petroelastic surrogate."""
    X, Y = load_training_data(root)
    models = []
    for j in range(3):
        model = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            StandardScaler(),
            Ridge(alpha=alpha),
        )
        model.fit(X, Y[:, j])
        models.append(model)
    return models


def ricker(f_hz: float, dt_s: float = 0.004, duration_s: float = 0.128) -> np.ndarray:
    t = np.arange(-duration_s / 2, duration_s / 2 + dt_s, dt_s)
    a = (np.pi * f_hz * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return w / np.max(np.abs(w))


def backus_vti(vp, vs, rho, scale_m: float, dz_m: float):
    """Gaussian-weighted Backus-style VTI homogenization used in the benchmark."""
    vp, vs, rho = [np.asarray(x, float) for x in (vp, vs, rho)]
    lam = rho * (vp * vp - 2.0 * vs * vs)
    mu = rho * vs * vs
    c = lam + 2.0 * mu
    f = lam
    sigma = max(scale_m / dz_m, 1e-3)

    def av(x):
        return gaussian_filter1d(np.asarray(x, float), sigma, mode="nearest", truncate=4.0)

    invc = av(1.0 / c)
    invmu = av(1.0 / mu)
    fc = av(f / c)
    c33 = 1.0 / invc
    c44 = 1.0 / invmu
    c13 = fc / invc
    c11 = av(c - f * f / c) + fc * fc / invc
    rr = av(rho)
    vp0 = np.sqrt(np.maximum(c33 / rr, 1e-12))
    vs0 = np.sqrt(np.maximum(c44 / rr, 1e-12))
    eps = (c11 - c33) / (2.0 * c33)
    den = 2.0 * c33 * (c33 - c44)
    delta = ((c13 + c44) ** 2 - (c33 - c44) ** 2) / np.where(np.abs(den) > 1e-20, den, np.nan)
    return vp0, vs0, rr, np.nan_to_num(eps), np.nan_to_num(delta)


def ruger_pp(vp, vs, rho, eps, delta, angles_deg):
    """Weak-anisotropy PP approximation used after VTI homogenization."""
    vpa = (vp[:-1] + vp[1:]) / 2.0
    vsa = (vs[:-1] + vs[1:]) / 2.0
    rhoa = (rho[:-1] + rho[1:]) / 2.0
    dvp, dvs, dr = np.diff(vp), np.diff(vs), np.diff(rho)
    de, dd = np.diff(eps), np.diff(delta)
    th = np.deg2rad(np.asarray(angles_deg))[None, :]
    s2, t2 = np.sin(th) ** 2, np.tan(th) ** 2
    dvpn = (dvp / vpa)[:, None]
    dvsn = (dvs / vsa)[:, None]
    drn = (dr / rhoa)[:, None]
    rat = (vsa / vpa)[:, None] ** 2
    r0 = 0.5 * (dvpn + drn)
    g = 0.5 * dvpn - 2.0 * rat * (drn + 2.0 * dvsn)
    f = 0.5 * dvpn
    riso = r0 + g * s2 + f * (t2 - s2)
    return riso + 0.5 * dd[:, None] * s2 + 0.5 * de[:, None] * s2 * t2


class ControlledBenchmark:
    """Portable nonlinear forward model for the compact 19A controlled window."""

    def __init__(self, root: Path | None = None, degree: int = 3, alpha: float = 0.1):
        self.root = repository_root() if root is None else Path(root)
        self.cfg = load_config(self.root)
        self.models = fit_reference_surrogate(self.root, degree=degree, alpha=alpha)
        d = pd.read_csv(self.root / "data/compact/19A_controlled_window.csv")
        self.depth = d["depth_m"].to_numpy(float)
        self.pet = d[["phi", "vsh", "sw"]].to_numpy(float)
        self.elas = d[["vp_mps", "vs_mps", "rho_gcc"]].to_numpy(float)
        self.inside = d["in_hugin"].to_numpy(int).astype(bool)
        self.dz = float(np.median(np.diff(self.depth)))
        self.base_rp = np.column_stack([m.predict(self.pet) for m in self.models])
        means = self.pet[self.inside].mean(axis=0)
        stds = self.pet[self.inside].std(axis=0)
        zscore = (self.pet - means) / np.maximum(stds, 1e-12)
        self.B = np.zeros_like(self.pet)
        for k in range(3):
            raw = (np.cos(self.cfg.tau) + np.sin(self.cfg.tau) * zscore[:, k]) * self.inside
            self.B[:, k] = raw / np.sqrt(np.mean(raw[self.inside] ** 2))

        wave = ricker(self.cfg.wavelet_hz)
        n = len(self.depth)
        eye = np.eye(n)
        self.W = np.column_stack([fftconvolve(eye[:, j], wave, mode="same") for j in range(n)])
        self.Cz = toeplitz(self.cfg.rho_vertical ** np.arange(n))
        self.Ca = toeplitz(self.cfg.rho_angle ** np.arange(len(self.cfg.angles_deg)))
        self.Qz = np.linalg.inv(self.Cz)
        self.Qa = np.linalg.inv(self.Ca)
        self.Lz = np.linalg.cholesky(self.Cz)
        self.La = np.linalg.cholesky(self.Ca)

    @property
    def shape(self):
        return len(self.depth), len(self.cfg.angles_deg)

    def forward(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, float)
        u, eta = theta[:3], float(theta[3])
        scale_m = self.cfg.nominal_scale_m * np.exp(eta)
        pp = self.pet + self.B * (self.cfg.prior_scales * u)[None, :]
        pp[:, 0] = np.clip(pp[:, 0], 0.005, 0.45)
        pp[:, 1:] = np.clip(pp[:, 1:], 0.0, 1.0)
        rp = np.column_stack([m.predict(pp) for m in self.models])
        el = self.elas + (rp - self.base_rp)
        vp, vs, rho, eps, delta = backus_vti(el[:, 0], el[:, 1], el[:, 2], scale_m, self.dz)
        refl = ruger_pp(vp, vs, rho, eps, delta, self.cfg.angles_deg)
        return self.W @ np.vstack([np.zeros((1, len(self.cfg.angles_deg))), refl])

    def quad(self, residual: np.ndarray) -> float:
        r = np.asarray(residual, float)
        return float(np.sum(r * (self.Qz @ r @ self.Qa)))

    def generate_noise(self, rng: np.random.Generator) -> np.ndarray:
        nz, na = self.shape
        z = rng.standard_normal((nz, na))
        return self.cfg.sigma_noise * (self.Lz @ z @ self.La.T)

    def jacobian(self, theta: np.ndarray, step: float = 1e-4) -> np.ndarray:
        theta = np.asarray(theta, float)
        ndata = np.prod(self.shape)
        J = np.zeros((ndata, 4), float)
        for k in range(4):
            tp, tm = theta.copy(), theta.copy()
            tp[k] += step
            tm[k] -= step
            J[:, k] = ((self.forward(tp) - self.forward(tm)) / (2.0 * step)).ravel()
        return J

    def precision_gram(self, J: np.ndarray) -> np.ndarray:
        cols = []
        nz, na = self.shape
        for k in range(J.shape[1]):
            r = J[:, k].reshape(nz, na)
            cols.append((self.Qz @ r @ self.Qa).ravel())
        cj = np.column_stack(cols)
        return (J.T @ cj) / (self.cfg.sigma_noise ** 2)

    def local_geometry(self, theta: np.ndarray):
        J = self.jacobian(theta)
        G = self.precision_gram(J)
        Guu = G[:3, :3]
        Gue = G[:3, 3:4]
        Gee = np.array([[G[3, 3] + 1.0 / self.cfg.eta_prior_sd ** 2]])
        Gq = quotient_metric(Guu, Gue, Gee)
        lam, V, authority = eigendirectional_authority(Gq)
        H = G + np.diag([1.0, 1.0, 1.0, 1.0 / self.cfg.eta_prior_sd ** 2])
        C = np.linalg.inv(H)
        return {"G": G, "Gq": Gq, "lambda": lam, "V": V, "authority": authority, "laplace_cov": C}
