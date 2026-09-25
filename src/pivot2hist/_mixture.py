"""Gaussian mixture models: EM with a BIC-chosen component count, no scipy/sklearn.

Two uses:

* **1-D bins** (:mod:`._binning`'s ``bins="mixture"``): fit modes to a numeric column and
  cut between them at the valley of the combined density — bins that respect the data's
  actual shape (a bimodal byte-size column gets one bin range per mode) rather than
  arbitrary equal widths or hard k-means partitions.
* **Soft clustering** (:func:`pivot2hist.View.cluster` with ``method="gmm"``): the hard
  assignment (highest-responsibility component) becomes the cluster label, same interface
  as the other cluster methods; :func:`pivot2hist.View.modes` reports the raw components
  (weight, mean, std) for a column, a compact "how many peaks, where" summary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


def _gaussian_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    cov = cov + np.eye(d) * 1e-6
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + np.eye(d) * 1e-3
        _, logdet = np.linalg.slogdet(cov)
    prec = np.linalg.inv(cov)
    diff = X - mean
    maha = np.einsum("ij,jk,ik->i", diff, prec, diff)
    return -0.5 * (d * math.log(2 * math.pi) + logdet + maha)


@dataclass(frozen=True)
class GMMFit:
    """A fitted mixture: ``k`` Gaussian components with weights, means and covariances."""

    weights: np.ndarray  # (k,)
    means: np.ndarray  # (k, d)
    covs: np.ndarray  # (k, d, d)
    k: int
    n: int
    d: int
    loglik: float
    bic: float

    def log_density(self, X: np.ndarray) -> np.ndarray:
        """Log of the mixture density at each row of ``X``."""
        X = np.atleast_2d(X)
        logpdfs = np.stack([_gaussian_logpdf(X, self.means[j], self.covs[j]) for j in range(self.k)], axis=1)
        logw = np.log(np.maximum(self.weights, 1e-300))
        combined = logpdfs + logw
        m = combined.max(axis=1, keepdims=True)
        return (m[:, 0] + np.log(np.exp(combined - m).sum(axis=1)))

    def responsibilities(self, X: np.ndarray) -> np.ndarray:
        """Posterior probability of each component for each row of ``X``: ``(n, k)``."""
        X = np.atleast_2d(X)
        logpdfs = np.stack([_gaussian_logpdf(X, self.means[j], self.covs[j]) for j in range(self.k)], axis=1)
        logw = np.log(np.maximum(self.weights, 1e-300))
        combined = logpdfs + logw
        m = combined.max(axis=1, keepdims=True)
        unnorm = np.exp(combined - m)
        return unnorm / unnorm.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard assignment: the most probable component for each row of ``X``."""
        return self.responsibilities(X).argmax(axis=1)

    def components(self) -> List[dict]:
        """Components as plain dicts, sorted by mean (1-D convenience: uses ``means[:, 0]``)."""
        order = np.argsort(self.means[:, 0])
        out = []
        for i in order:
            out.append({
                "weight": float(self.weights[i]),
                "mean": self.means[i].tolist() if self.d > 1 else float(self.means[i, 0]),
                "std": float(np.sqrt(max(self.covs[i, 0, 0], 0.0))) if self.d == 1 else None,
            })
        return out

    def __repr__(self) -> str:
        parts = ", ".join(f"{c['weight']:.0%}@{c['mean']:.3g}" for c in self.components()) if self.d == 1 else f"{self.k} components"
        return f"<GMMFit k={self.k} {parts} bic={self.bic:.1f}>"


def fit_gmm(X, k: int, *, seed: int = 0, iters: int = 100, tol: float = 1e-5, n_init: int = 3) -> GMMFit:
    """Fit a ``k``-component Gaussian mixture by EM (best of ``n_init`` random starts)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1 and X.shape[1] != 1:
        X = X.T
    n, d = X.shape
    k = max(1, min(int(k), n))
    from ._cluster import _kmeans_pp_init

    rng = np.random.default_rng(seed)
    best: Optional[GMMFit] = None
    for init in range(max(1, n_init)):
        means = _kmeans_pp_init(X, k, rng).astype(float).copy()  # spread-out seeds: EM rarely gets stuck merging distinct modes
        base_cov = np.cov(X.T).reshape(d, d) if n > d else np.eye(d)
        covs = np.array([base_cov + np.eye(d) * (1e-3 * (np.trace(base_cov) / d + 1e-6)) for _ in range(k)])
        weights = np.full(k, 1.0 / k)
        prev_ll = -np.inf
        for _ in range(iters):
            logpdfs = np.stack([_gaussian_logpdf(X, means[j], covs[j]) for j in range(k)], axis=1)
            logw = np.log(np.maximum(weights, 1e-300))
            combined = logpdfs + logw
            m = combined.max(axis=1, keepdims=True)
            log_norm = m[:, 0] + np.log(np.exp(combined - m).sum(axis=1))
            resp = np.exp(combined - log_norm[:, None])
            ll = float(log_norm.sum())
            if not np.isfinite(ll):
                break
            if abs(ll - prev_ll) < tol * (abs(prev_ll) + 1):
                prev_ll = ll
                break
            prev_ll = ll
            Nk = resp.sum(axis=0) + 1e-12
            weights = Nk / n
            means = (resp.T @ X) / Nk[:, None]
            new_covs = np.empty((k, d, d))
            for j in range(k):
                diff = X - means[j]
                cj = (resp[:, j : j + 1] * diff).T @ diff / Nk[j]
                new_covs[j] = cj + np.eye(d) * 1e-6
            covs = new_covs
        if not np.isfinite(prev_ll):
            continue
        k_params = k * d + k * d * (d + 1) // 2 + (k - 1)
        bic = k_params * math.log(max(n, 2)) - 2 * prev_ll
        fit = GMMFit(weights, means, covs, k, n, d, prev_ll, bic)
        if best is None or fit.bic < best.bic:
            best = fit
    if best is None:  # degenerate data (e.g. all-identical points): fall back to a single wide component
        mean = X.mean(axis=0, keepdims=True)
        cov = (np.eye(d) * max(float(np.var(X)), 1e-6))[None, :, :]
        ll = float(_gaussian_logpdf(X, mean[0], cov[0]).sum())
        best = GMMFit(np.array([1.0]), mean, cov, 1, n, d, ll, d * math.log(max(n, 2)) - 2 * ll)
    return best


def choose_gmm_k(X, k_max: int = 8, *, seed: int = 0, n_init: int = 4) -> int:
    """Best component count in ``1..k_max`` by BIC."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1 and X.shape[1] != 1:
        X = X.T
    n = X.shape[0]
    if n < 4:
        return 1
    k_max = max(1, min(k_max, n // 2))
    best_k, best_bic = 1, math.inf
    for k in range(1, k_max + 1):
        fit = fit_gmm(X, k, seed=seed, n_init=n_init)
        if fit.bic < best_bic - 1e-6:
            best_k, best_bic = k, fit.bic
        elif k > best_k + 1:  # BIC stopped improving for two consecutive k: stop early
            break
    return best_k


def mixture_cutpoints(fit: GMMFit, *, grid_points: int = 400) -> np.ndarray:
    """Cut point between each pair of adjacent (by mean) 1-D components: the local minimum
    of the combined density in between — where the modes actually separate, not merely
    their midpoint."""
    if fit.d != 1:
        raise ValueError("mixture_cutpoints needs a 1-D mixture")
    order = np.argsort(fit.means[:, 0])
    means = fit.means[order, 0]
    cuts = []
    for i in range(len(means) - 1):
        lo, hi = means[i], means[i + 1]
        if hi <= lo:
            cuts.append((lo + hi) / 2)
            continue
        pad = 0.15 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, grid_points).reshape(-1, 1)
        grid = grid[(grid[:, 0] >= lo) & (grid[:, 0] <= hi)]
        if grid.size == 0:
            cuts.append((lo + hi) / 2)
            continue
        dens = fit.log_density(grid)
        cuts.append(float(grid[np.argmin(dens), 0]))
    return np.array(cuts, dtype=float)


__all__ = ["GMMFit", "fit_gmm", "choose_gmm_k", "mixture_cutpoints"]
