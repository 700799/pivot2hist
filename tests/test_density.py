import numpy as np
import pytest

from pivot2hist._density import (
    FAMILIES,
    DistFit,
    fit_distribution,
    rank_distributions,
)

rng = np.random.default_rng(0)


@pytest.mark.parametrize(
    "name,sample",
    [
        ("normal", rng.normal(50, 5, 4000)),
        ("lognormal", np.exp(rng.normal(7, 1.8, 4000))),
        ("exponential", rng.exponential(30, 4000)),
        ("gamma", rng.gamma(3.0, 20.0, 4000)),
        ("uniform", rng.uniform(0, 100, 4000)),
        ("poisson", rng.poisson(8, 4000).astype(float)),
        ("bernoulli", (rng.random(4000) < 0.3).astype(float)),
    ],
)
def test_recovers_the_generating_family(name, sample):
    fit = fit_distribution(sample)
    assert fit is not None and fit.name == name
    assert fit.n == sample.size and fit.bic > 0 or fit.bic < 1e9  # sane finite number


def test_discrete_uniform_and_geometric():
    dice = rng.integers(1, 7, 3000).astype(float)
    fit = fit_distribution(dice)
    assert fit.name == "duniform"
    assert fit.params["lo"] == 1 and fit.params["hi"] == 6

    geo = rng.geometric(0.2, 4000).astype(float) - 1  # numpy support starts at 1
    fit = fit_distribution(geo)
    assert fit.name in ("geometric", "exponential")  # close analogues; both defensible


def test_wide_range_integer_data_still_gets_continuous_families():
    # bytes-like: lognormal rounded to whole numbers, thousands of distinct values
    bytes_ = np.exp(rng.normal(7, 2, 5000)).round()
    fit = fit_distribution(bytes_)
    assert fit.name in ("lognormal", "gamma")  # not forced into the discrete-only pool


def test_low_cardinality_integers_stay_in_the_discrete_pool():
    severity = rng.choice([1, 2, 3, 4, 5], 3000, p=[0.45, 0.28, 0.15, 0.08, 0.04]).astype(float)
    fit = fit_distribution(severity)
    assert fit.name in ("poisson", "geometric", "bernoulli", "duniform")


def test_density_and_mass_are_never_mixed_by_default():
    # a pathological case that would "win" unfairly if pdf and pmf log-likelihoods were compared directly
    binary = np.array([0.0] * 2800 + [1.0] * 1200)
    fit = fit_distribution(binary)
    assert fit.name == "bernoulli"
    assert abs(fit.params["p"] - 0.3) < 0.02


def test_pdf_integrates_to_one():
    x = rng.normal(10, 2, 3000)
    fit = fit_distribution(x)
    grid = np.linspace(-10, 30, 5000)
    y = fit.pdf(grid)
    integral = float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(grid)))
    assert abs(integral - 1) < 0.02


def test_pmf_sums_to_one_ish():
    x = rng.poisson(6, 3000).astype(float)
    fit = fit_distribution(x)
    ks = np.arange(0, 40, dtype=float)
    total = float(fit.pdf(ks).sum())
    assert abs(total - 1) < 0.01


def test_edge_cases():
    assert fit_distribution([1, 2, 3]) is None  # below min_n
    assert fit_distribution([5.0] * 50) is not None  # constant: some family still fits (e.g. duniform/bernoulli degenerate)
    assert fit_distribution(np.array([np.nan, np.inf, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])) is not None
    with pytest.raises(ValueError):
        fit_distribution(rng.normal(size=100), families=["not-a-family"])


def test_rank_distributions_sorted_by_bic():
    x = rng.normal(0, 1, 3000)
    ranked = rank_distributions(x)
    assert isinstance(ranked, list) and len(ranked) >= 2
    assert all(isinstance(f, DistFit) for f in ranked)
    bics = [f.bic for f in ranked]
    assert bics == sorted(bics)
    assert ranked[0].name == "normal"
    assert rank_distributions([1, 2, 3]) == []


def test_families_constant_and_repr():
    assert set(FAMILIES) == {"normal", "lognormal", "exponential", "gamma", "uniform", "poisson", "geometric", "bernoulli", "duniform"}
    f = fit_distribution(rng.normal(size=500))
    assert "normal" in repr(f).lower()
    assert "mu=" in f.describe()
