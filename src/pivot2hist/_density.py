"""Which distribution does this column look like? A small, dependency-free answer.

:func:`fit_distribution` fits a handful of closed-form families to a numeric sample and
keeps the one with the lowest BIC (Bayesian Information Criterion: log-likelihood
penalised by parameter count, so a family only wins by explaining the data better, not
just by having more knobs). This is the "guess the density" half of type guessing,
inspired by the automatic-model-selection idea in probabilistic-modeling libraries —
written from scratch, with a handful of named families rather than a general engine.

Deliberately not wired into :func:`pivot2hist.profile` or the auto-fit search: fitting
five-plus distributions is too slow to run on every candidate column during a layout
search, so this is opt-in (:func:`pivot2hist.distribution`, ``View.distribution()``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

_LOG_2PI = math.log(2 * math.pi)


def _lgamma(x: np.ndarray) -> np.ndarray:
    """Vectorised log-gamma via the stdlib (no scipy dependency)."""
    return np.array([math.lgamma(float(v)) for v in np.atleast_1d(x)]).reshape(np.shape(x))


@dataclass(frozen=True)
class DistFit:
    """One fitted family: its parameters, log-likelihood and BIC (lower is better)."""

    name: str
    params: Dict[str, float]
    loglik: float
    bic: float
    n: int
    k: int  # parameter count

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Density (or probability mass, for the discrete families) at ``x``."""
        x = np.asarray(x, dtype=float)
        p = self.params
        if self.name == "normal":
            z = (x - p["mu"]) / p["sigma"]
            return np.exp(-0.5 * z * z) / (p["sigma"] * math.sqrt(2 * math.pi))
        if self.name == "lognormal":
            out = np.zeros_like(x)
            pos = x > 0
            z = (np.log(x[pos]) - p["mu"]) / p["sigma"]
            out[pos] = np.exp(-0.5 * z * z) / (x[pos] * p["sigma"] * math.sqrt(2 * math.pi))
            return out
        if self.name == "exponential":
            return np.where(x >= 0, p["rate"] * np.exp(-p["rate"] * x), 0.0)
        if self.name == "gamma":
            k, theta = p["shape"], p["scale"]
            out = np.zeros_like(x)
            pos = x > 0
            xv = x[pos]
            out[pos] = np.exp((k - 1) * np.log(xv) - xv / theta - k * math.log(theta) - math.lgamma(k))
            return out
        if self.name == "uniform":
            lo, hi = p["lo"], p["hi"]
            return np.where((x >= lo) & (x <= hi), 1.0 / max(hi - lo, 1e-12), 0.0)
        if self.name == "poisson":
            lam = p["lambda"]
            return np.exp(x * math.log(lam) - lam - _lgamma(x + 1))
        if self.name == "geometric":
            pp = p["p"]
            return pp * (1 - pp) ** x
        if self.name == "bernoulli":
            pp = p["p"]
            return np.where(x >= 0.5, pp, 1 - pp)
        if self.name == "duniform":
            lo, hi = p["lo"], p["hi"]
            span = hi - lo + 1
            return np.where((x >= lo) & (x <= hi), 1.0 / span, 0.0)
        raise ValueError(self.name)  # pragma: no cover

    def describe(self) -> str:
        parts = ", ".join(f"{k}={v:.3g}" for k, v in self.params.items())
        return f"{self.name.capitalize()}({parts})"

    def __repr__(self) -> str:
        return f"<DistFit {self.describe()} bic={self.bic:.1f}>"


def _bic(loglik: float, k: int, n: int) -> float:
    return k * math.log(max(n, 2)) - 2.0 * loglik


def _fit_normal(x: np.ndarray) -> Optional[DistFit]:
    n = x.size
    mu, sigma = float(x.mean()), float(x.std(ddof=0))
    if sigma <= 0:
        return None
    ll = float(-0.5 * n * (_LOG_2PI + 2 * math.log(sigma) + 1))
    return DistFit("normal", {"mu": mu, "sigma": sigma}, ll, _bic(ll, 2, n), n, 2)


def _fit_lognormal(x: np.ndarray) -> Optional[DistFit]:
    if not np.all(x > 0):
        return None
    n = x.size
    lx = np.log(x)
    mu, sigma = float(lx.mean()), float(lx.std(ddof=0))
    if sigma <= 0:
        return None
    ll = float(-0.5 * n * (_LOG_2PI + 2 * math.log(sigma) + 1) - lx.sum())
    return DistFit("lognormal", {"mu": mu, "sigma": sigma}, ll, _bic(ll, 2, n), n, 2)


def _fit_exponential(x: np.ndarray) -> Optional[DistFit]:
    if not np.all(x >= 0) or x.mean() <= 0:
        return None
    n = x.size
    rate = 1.0 / float(x.mean())
    ll = float(n * math.log(rate) - rate * x.sum())
    return DistFit("exponential", {"rate": rate}, ll, _bic(ll, 1, n), n, 1)


def _fit_gamma(x: np.ndarray, iters: int = 50) -> Optional[DistFit]:
    """MLE via Newton's method on the shape parameter (the standard small iteration;
    ``digamma``/``trigamma`` approximated with well-known asymptotic series, good to
    a few parts in 1e6 for the shapes this ever sees)."""
    if not np.all(x > 0):
        return None
    n = x.size
    xbar = float(x.mean())
    s = math.log(xbar) - float(np.log(x).mean())
    if s <= 1e-10:
        return None
    k = (3 - s + math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(iters):
        dg, tg = _digamma(k), _trigamma(k)
        step = (math.log(k) - dg - s) / (1.0 / k - tg)
        k_new = k - step
        if not (k_new > 0):
            break
        if abs(k_new - k) < 1e-9:
            k = k_new
            break
        k = k_new
    if not (k > 0):
        return None
    theta = xbar / k
    ll = float((k - 1) * np.log(x).sum() - x.sum() / theta - n * k * math.log(theta) - n * math.lgamma(k))
    return DistFit("gamma", {"shape": k, "scale": theta}, ll, _bic(ll, 2, n), n, 2)


def _digamma(x: float) -> float:
    r = 0.0
    while x < 6:
        r -= 1.0 / x
        x += 1
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f / 240)))


def _trigamma(x: float) -> float:
    r = 0.0
    while x < 6:
        r += 1.0 / (x * x)
        x += 1
    f = 1.0 / (x * x)
    return r + 1 / x + f / 2 + f / x * (1 / 6 - f * (1 / 30 - f * (1 / 42 - f / 30)))


def _fit_uniform(x: np.ndarray) -> Optional[DistFit]:
    n = x.size
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return None
    ll = float(-n * math.log(hi - lo))
    return DistFit("uniform", {"lo": lo, "hi": hi}, ll, _bic(ll, 2, n), n, 2)


def _is_count_data(x: np.ndarray) -> bool:
    return bool(np.all(x >= 0) and np.all(np.mod(x, 1) == 0))


def _fit_poisson(x: np.ndarray) -> Optional[DistFit]:
    if not _is_count_data(x):
        return None
    n = x.size
    lam = float(x.mean())
    if lam <= 0:
        return None
    ll = float((x * math.log(lam) - lam).sum() - _lgamma(x + 1).sum())
    return DistFit("poisson", {"lambda": lam}, ll, _bic(ll, 1, n), n, 1)


def _fit_geometric(x: np.ndarray) -> Optional[DistFit]:
    if not _is_count_data(x):
        return None
    n = x.size
    mean = float(x.mean())
    p = 1.0 / (mean + 1.0)
    if not (0 < p < 1):
        return None
    ll = float(x.sum() * math.log(1 - p) + n * math.log(p))
    return DistFit("geometric", {"p": p}, ll, _bic(ll, 1, n), n, 1)


def _fit_discrete_uniform(x: np.ndarray) -> Optional[DistFit]:
    """Discrete uniform on the integers ``[lo, hi]`` — a properly normalised competitor
    for ``poisson``/``geometric``/``bernoulli`` (unlike the continuous ``uniform``, whose
    density can exceed 1 and so would win unfairly on any narrow discrete range)."""
    if not _is_count_data(x):
        return None
    n = x.size
    lo, hi = float(x.min()), float(x.max())
    span = int(round(hi - lo)) + 1
    if span < 2:
        return None
    ll = float(-n * math.log(span))
    return DistFit("duniform", {"lo": lo, "hi": hi}, ll, _bic(ll, 2, n), n, 2)


def _fit_bernoulli(x: np.ndarray) -> Optional[DistFit]:
    uniq = set(np.unique(x).tolist())
    if not uniq <= {0.0, 1.0}:
        return None
    n = x.size
    p = float(x.mean())
    if not (0 < p < 1):
        return None
    ones = float(x.sum())
    ll = float(ones * math.log(p) + (n - ones) * math.log(1 - p))
    return DistFit("bernoulli", {"p": p}, ll, _bic(ll, 1, n), n, 1)


_FITTERS = {
    "normal": _fit_normal,
    "lognormal": _fit_lognormal,
    "exponential": _fit_exponential,
    "gamma": _fit_gamma,
    "uniform": _fit_uniform,
    "poisson": _fit_poisson,
    "geometric": _fit_geometric,
    "bernoulli": _fit_bernoulli,
    "duniform": _fit_discrete_uniform,
}
FAMILIES = tuple(_FITTERS)
#: A probability *density* (normal, lognormal, exponential, gamma, uniform) and a
#: probability *mass* (poisson, geometric, bernoulli, duniform) aren't comparable by raw
#: log-likelihood: a density can exceed 1 (and always does somewhere, or it wouldn't
#: integrate to 1 over a support narrower than that) and so can "win" BIC on discrete
#: data by exploiting repeated values, not by fitting better. So the default search picks
#: one pool by whether the data is integer-valued; ``duniform`` (discrete uniform on
#: ``[lo, hi]``) is the properly normalised analogue of ``uniform`` for that pool.
_DISCRETE = ("poisson", "geometric", "bernoulli", "duniform")
_CONTINUOUS = ("normal", "lognormal", "exponential", "gamma", "uniform")


def _default_families(x: np.ndarray, discrete_max: int) -> Sequence[str]:
    """Discrete pool for genuinely low-cardinality integer data (labels, small counts);
    continuous pool otherwise — including for wide-range integer data (bytes, latencies
    in whole ms ...), where nothing repeats often enough for the density-vs-mass unfairness
    in the module docstring to bite, and a heavy-tailed continuous family (lognormal,
    gamma) usually describes it far better than forcing it through a Poisson/Geometric."""
    return _DISCRETE if _is_count_data(x) and np.unique(x).size <= discrete_max else _CONTINUOUS


def fit_distribution(
    values, *, families: Optional[Sequence[str]] = None, min_n: int = 8, discrete_max: int = 20
) -> Optional[DistFit]:
    """Best-fitting family for ``values`` by BIC, or ``None`` if too little data (or
    nothing fits — an all-equal column, for instance).

    With no ``families``, low-cardinality integer data (at most ``discrete_max`` distinct
    values — small counts, a handful of codes) is compared only against the discrete
    families (``poisson``, ``geometric``, ``bernoulli``, ``duniform``); everything else,
    including wide-range integer data like byte counts, only against the continuous ones
    (``normal``, ``lognormal``, ``exponential``, ``gamma``, ``uniform``) — mixing density
    and mass families is a category error (see the module docstring). Pass ``families``
    explicitly to override this.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < min_n:
        return None
    names = families or _default_families(x, discrete_max)
    best: Optional[DistFit] = None
    for name in names:
        fitter = _FITTERS.get(name)
        if fitter is None:
            raise ValueError(f"unknown family {name!r}; use one of {FAMILIES}")
        try:
            fit = fitter(x)
        except (ValueError, ArithmeticError, OverflowError):
            fit = None
        if fit is not None and (best is None or fit.bic < best.bic):
            best = fit
    return best


def rank_distributions(
    values, *, families: Optional[Sequence[str]] = None, min_n: int = 8, discrete_max: int = 20
) -> List[DistFit]:
    """Every family that fits ``values``, best (lowest BIC) first. See :func:`fit_distribution`
    for how the default family pool is chosen."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < min_n:
        return []
    names = families or _default_families(x, discrete_max)
    fits: List[DistFit] = []
    for name in names:
        try:
            fit = _FITTERS[name](x)
        except (ValueError, ArithmeticError, OverflowError):
            fit = None
        if fit is not None:
            fits.append(fit)
    return sorted(fits, key=lambda f: f.bic)


__all__ = ["DistFit", "fit_distribution", "rank_distributions", "FAMILIES"]
