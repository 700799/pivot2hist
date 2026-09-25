import math

import numpy as np
import pandas as pd
import pytest

import pivot2hist as p2h
from pivot2hist._fit import Dim, FitOptions, OBJECTIVES, Shape, _bic_association, score_layout


def test_default_objective_is_heuristic_and_unchanged():
    assert FitOptions().objective == "heuristic"
    auth = p2h.sample.auth_logs(1500, seed=1)
    v_default = p2h.fit(auth)
    v_explicit = p2h.fit(auth, objective="heuristic")
    assert v_default.layout == v_explicit.layout


def test_unknown_objective_raises():
    auth = p2h.sample.auth_logs(200, seed=1)
    with pytest.raises(ValueError):
        p2h.fit(auth, objective="bogus").pivot()


def test_bic_association_rewards_signal_over_noise():
    # Same n; a table entirely explained by independence (mutual_info=0) should never
    # be worth its own complexity once it has any real degrees of freedom.
    no_signal = Shape(n_rows=5, n_cols=5, cells=25, entropy=2.0, mutual_info=0.0, other_frac=0.0, n=1000)
    assert _bic_association(no_signal) < 0


def test_bic_association_penalizes_a_bigger_table_more_for_the_same_mutual_info():
    small = Shape(n_rows=3, n_cols=3, cells=9, entropy=1.5, mutual_info=0.3, other_frac=0.0, n=500)
    big = Shape(n_rows=30, n_cols=10, cells=250, entropy=4.5, mutual_info=0.3, other_frac=0.0, n=500)
    assert _bic_association(small) > _bic_association(big)


def test_bic_association_grows_with_more_evidence():
    # the same shape and mutual_info, but more sample rows: less skepticism is warranted,
    # so the BIC score for a fixed association should not get worse as n grows.
    shape_small_n = Shape(n_rows=10, n_cols=5, cells=50, entropy=2.0, mutual_info=0.05, other_frac=0.0, n=200)
    shape_big_n = Shape(n_rows=10, n_cols=5, cells=50, entropy=2.0, mutual_info=0.05, other_frac=0.0, n=200_000)
    assert _bic_association(shape_big_n) > _bic_association(shape_small_n)


def test_bic_mode_penalizes_bigger_table_relative_to_heuristic_mode():
    # isolate just the association-term swap: same shape fed to both objectives, so any
    # difference in the (bic - heuristic) delta is caused only by that term.
    small = Shape(n_rows=3, n_cols=3, cells=9, entropy=1.5, mutual_info=0.3, other_frac=0.0, n=500)
    big = Shape(n_rows=30, n_cols=10, cells=250, entropy=4.5, mutual_info=0.3, other_frac=0.0, n=500)
    r_small, c_small = [Dim("r", "categorical", 3)], [Dim("c", "categorical", 3)]
    r_big, c_big = [Dim("r", "categorical", 30)], [Dim("c", "categorical", 10)]
    opts_bic, opts_heur = FitOptions(objective="bic"), FitOptions(objective="heuristic")

    delta_small = score_layout(small, r_small, c_small, opts_bic) - score_layout(small, r_small, c_small, opts_heur)
    delta_big = score_layout(big, r_big, c_big, opts_bic) - score_layout(big, r_big, c_big, opts_heur)
    assert delta_big < delta_small


def test_bic_mode_produces_a_valid_layout_end_to_end():
    auth = p2h.sample.auth_logs(3000, seed=1)
    v = p2h.fit(auth, objective="bic")
    assert v.layout.rows and v.pivot().shape[0] >= 1


def test_row_only_scoring_is_well_defined_for_bic():
    # during the row-selection phase (no columns yet), mutual_info is 0 by convention;
    # bic mode must not blow up or divide by zero there.
    shape = Shape(n_rows=5, n_cols=1, cells=5, entropy=1.6, mutual_info=0.0, other_frac=0.0, n=500)
    sc = score_layout(shape, [Dim("r", "categorical", 5)], [], FitOptions(objective="bic"))
    assert math.isfinite(sc)


def test_objectives_constant_exported():
    assert OBJECTIVES == ("heuristic", "bic")
    assert p2h.OBJECTIVES == OBJECTIVES
