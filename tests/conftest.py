import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_lc(tmp_path):
    """Path to a plain 3 column light curve CSV (time, flux, fluxerr)."""
    rng = np.random.default_rng(42)
    n = 120
    t = np.linspace(2460000.0, 2460000.1, n)
    flux = 1.0 + rng.normal(0, 1e-3, n)
    fluxerr = np.full(n, 1e-3)
    fp = tmp_path / 'plain.csv'
    pd.DataFrame({'time': t, 'flux': flux, 'fluxerr': fluxerr}).to_csv(fp, index=False)
    return str(fp)


@pytest.fixture
def synthetic_lc_aux(tmp_path):
    """Path to a light curve CSV with two auxiliary covariate columns."""
    rng = np.random.default_rng(7)
    n = 120
    t = np.linspace(2460000.0, 2460000.1, n)
    flux = 1.0 + rng.normal(0, 1e-3, n)
    fluxerr = np.full(n, 1e-3)
    fp = tmp_path / 'aux.csv'
    pd.DataFrame({
        'time': t,
        'flux': flux,
        'fluxerr': fluxerr,
        # deliberately not linear or quadratic in time: a covariate that lies
        # in the span of the polynomial trend basis would make the design
        # matrix rank deficient and break the rank assertions in test_io.py
        'airmass': 1.3 - 0.3 * np.cos(np.linspace(0.0, 2.5, n)) + rng.normal(0, 1e-3, n),
        'dx': rng.normal(0, 1, n),
    }).to_csv(fp, index=False)
    return str(fp)


@pytest.fixture
def unit_spaced_lc(tmp_path):
    """A 10 point light curve one day apart, with two covariates.

    read_generic subtracts int(x.min()), so this reads back as x = [0..9]
    exactly, which makes trim boundaries assertable as literals rather than
    as floating point neighborhoods.
    """
    n = 10
    t = 2460000.0 + np.arange(n, dtype=float)
    fp = tmp_path / 'unit.csv'
    pd.DataFrame({
        'time': t,
        'flux': np.full(n, 1.0) + np.arange(n) * 1e-4,
        'fluxerr': np.full(n, 1e-3),
        'airmass': 1.2 + np.arange(n) * 0.01,
        'dx': np.arange(n) * 0.5,
    }).to_csv(fp, index=False)
    return str(fp)


@pytest.fixture
def gapped_lc(tmp_path):
    """Two blocks of 4 points one day apart, separated by a gap of exactly 2 days.

    Times read back as [0, 1, 2, 3, 5, 6, 7, 8], so the only diff above 1.0 is
    the 2.0 gap. That makes the chunk threshold comparison decidable by hand at
    the boundary: a threshold of exactly 2.0 must not split, a smaller one must.
    """
    t = 2460000.0 + np.array([0., 1., 2., 3., 5., 6., 7., 8.])
    fp = tmp_path / 'gapped.csv'
    pd.DataFrame({
        'time': t,
        'flux': np.full(len(t), 1.0),
        'fluxerr': np.full(len(t), 1e-3),
    }).to_csv(fp, index=False)
    return str(fp)


@pytest.fixture
def map_soln():
    """A minimal MAP solution dict: one dataset named 'g', 100 points, 1 planet.

    Shapes match what util.get_map_soln produces from a PyMC trace: scalar
    sites come back as 0-d arrays and vector sites keep their length.
    """
    n = 100
    return {
        't0': np.array(0.05),
        'period': np.array([3.0]),
        'ror': np.array([0.05]),
        'b': np.array([0.3]),
        'dur': np.array([0.1]),
        'u_star_g': np.array([0.4, 0.2]),
        'g_mean': np.array(0.1),
        'g_log_sigma_lc': np.array(-1.0),
        'g_lm': np.full(n, 0.2),
        'g_light_curves': np.full(n, -1.0),
        'g_light_curves_hr': np.full(500, -1.0),
    }


@pytest.fixture
def map_soln_multiplanet():
    """The same dataset fitted with two planets, so the light curves stay 2-D.

    This is the shape that reaches the `lcs.ndim > 1` branches in
    util.get_residuals, util.get_outlier_mask and model._add_gp_predictions.
    The two planets have different depths, so summing over the planet axis is
    distinguishable from picking either column.
    """
    n = 100
    return {
        't0': np.array([0.05, 0.06]),
        'period': np.array([3.0, 7.0]),
        'ror': np.array([0.05, 0.03]),
        'b': np.array([0.3, 0.1]),
        'dur': np.array([0.1, 0.15]),
        'u_star_g': np.array([0.4, 0.2]),
        'g_mean': np.array(0.1),
        'g_log_sigma_lc': np.array(-1.0),
        'g_lm': np.full(n, 0.2),
        'g_light_curves': np.column_stack([np.full(n, -1.0), np.full(n, -0.25)]),
        'g_light_curves_hr': np.column_stack([np.full(500, -1.0), np.full(500, -0.25)]),
    }
