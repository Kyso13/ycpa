"""
Basic test suite for ycpa.
Golden tests: verify the C++ and pure-Python paths against fixed reference
values. If C++ is absent the tests still pass (fallback).
"""
import math
import numpy as np
import pandas as pd
import pytest

import ycpa


# Real Kepler planetary data (NASA NSSDC)
KEP_A = np.array([0.387, 0.723, 1.0, 1.524, 5.203, 9.537, 19.19, 30.07])
KEP_T = np.array([0.2408, 0.6152, 1.0, 1.8809, 11.8618, 29.4567, 84.0107, 164.786])


def test_import_and_version():
    assert hasattr(ycpa, "fit")
    assert isinstance(ycpa.__version__, str)


def test_cpp_status_runs():
    st = ycpa.cpp_status()
    assert "cpp_active" in st


def test_kepler_recovers_three_halves():
    """Most important test: must recover the Kepler exponent as 3/2."""
    df = pd.DataFrame({"a": KEP_A, "T": KEP_T})
    res = ycpa.fit(df, target="T", features=["a"], use_mcmc=False)
    assert res.exponents, "no model found"
    feat, label, value = res.exponents[0]
    assert feat == "a"
    assert abs(value - 1.5) < 1e-3
    assert label == "3/2"


def test_fit_result_formula():
    df = pd.DataFrame({"a": KEP_A, "T": KEP_T})
    res = ycpa.fit(df, target="T", features=["a"], use_mcmc=False)
    assert "T =" in res.formula
    assert "a^" in res.formula


def test_features_auto_inferred():
    """if features omitted, columns other than target must be used."""
    df = pd.DataFrame({"a": KEP_A, "T": KEP_T})
    res = ycpa.fit(df, target="T", use_mcmc=False)
    assert res.exponents


# --- Golden tests: scoring constants (C++ path tested if active) ---

def test_golden_mdl_score():
    from ycpa._engine import mdl_score
    yt = np.array([1., 2., 3., 4., 5., 6.5, 8.])
    yp = np.array([1.1, 1.9, 3.2, 3.8, 5.1, 6.4, 8.2])
    val = mdl_score(yp, yt, 3, 20)
    assert abs(val - 4.742845012432204) < 1e-9


def test_mcmc_returns_expected_keys():
    from ycpa._engine import tournament_mcmc
    X = np.column_stack([np.ones_like(np.log(KEP_A)), np.log(KEP_A)])
    y = np.log(KEP_T)
    mc = tournament_mcmc(X, y)
    for key in ("beta_mcmc", "energy", "gibbs", "n_chains", "n_elite"):
        assert key in mc


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# --- discover() general-SR + confidence-score tests ---

def test_discover_finds_sin():
    import ycpa
    x = np.linspace(0.1, 6, 80)
    df = pd.DataFrame({"x": x, "y": 3 * np.sin(x) + 2})
    r = ycpa.discover(df, "y")
    atoms = sorted(a for _, a in r.terms)
    assert "sin" in atoms
    assert r.confidence >= 0.8  # clean data → high confidence


def test_discover_low_confidence_out_of_pool():
    import ycpa
    x = np.linspace(0.1, 5, 80)
    df = pd.DataFrame({"x": x, "y": np.tanh(x)})  # tanh not in pool
    r = ycpa.discover(df, "y")
    # tanh out-of-pool → confidence must be low (despite high R²)
    assert r.confidence < 0.6


def test_discover_formula_format():
    import ycpa
    x = np.linspace(0.1, 5, 80)
    df = pd.DataFrame({"x": x, "y": x**2 + np.sin(x)})
    r = ycpa.discover(df, "y")
    assert "y =" in r.formula
    assert r.confidence is not None


# --- Usability: clear errors + helpers ---

def test_list_atoms():
    import ycpa
    pool = ycpa.list_atoms(pool_only=True)
    assert "sin" in pool
    allatoms = ycpa.list_atoms()
    assert len(allatoms) >= len(pool)


def test_clear_error_bad_target():
    import ycpa
    df = pd.DataFrame({"a": [1., 2, 3, 4], "T": [1., 4, 9, 16]})
    with pytest.raises(ValueError, match="target column"):
        ycpa.fit(df, target="WRONG", features=["a"])


def test_clear_error_bad_feature():
    import ycpa
    df = pd.DataFrame({"a": [1., 2, 3, 4], "T": [1., 4, 9, 16]})
    with pytest.raises(ValueError, match="feature column"):
        ycpa.fit(df, target="T", features=["nope"])


def test_clear_error_bad_atom():
    import ycpa
    df = pd.DataFrame({"a": [1., 2, 3, 4], "T": [1., 4, 9, 16]})
    with pytest.raises(ValueError, match="unknown atom"):
        ycpa.discover(df, target="T", atoms=["badatom"])


def test_clear_error_not_dataframe():
    import ycpa
    with pytest.raises(TypeError, match="DataFrame"):
        ycpa.fit([1, 2, 3], target="T", features=["a"])


def test_odd_length_data():
    """Odd row counts must not crash (regression for symmetry check)."""
    import ycpa
    df = pd.DataFrame({"a": [1., 2, 3, 4, 5], "T": [1., 4, 9, 16, 25]})
    r = ycpa.fit(df, target="T", features=["a"])
    assert r.exponents
