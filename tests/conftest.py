import os

import numpy as np
import pandas as pd
import pytest
import yaml


# index of the outlier make_project injects when clip=True
OUTLIER_INDEX = 17


def _flare(t0, ampl=6.0):
    """A flare peaking just after mid-transit, in ppt."""
    return dict(tpeak=t0 + 0.035, tpeak_unc=0.01, tpeak_prior='uniform',
                fwhm=0.008, fwhm_unc=0.01, fwhm_prior='uniform',
                ampl=ampl, ampl_unc=ampl * 2, ampl_prior='uniform')


def _bump(t0, ampl=2.0):
    """A Gaussian bump inside the transit, as a spot crossing would be."""
    return dict(tcenter=t0 - 0.005, tcenter_unc=0.01, tcenter_prior='uniform',
                width=0.004, width_unc=0.008, width_prior='uniform',
                ampl=ampl, ampl_unc=ampl * 2, ampl_prior='uniform')


def make_project(root, n=60, use_gp=False, clip=False, uniform=None,
                 flare=False, bump=False, bands=('g',)):
    """Write a minimal but complete timer project into `root`.

    One dataset in a single band, a linear trend, a boxy transit and white
    noise. Small enough that a full build plus a short sample runs in seconds,
    and real enough that every stage of the pipeline actually executes.

    With use_gp the light curve also carries a smooth wobble several times the
    white noise, so there is something for the GP to absorb and the GP branch
    of every downstream consumer is genuinely exercised.

    The other switches turn on the configuration branches that no fixture used
    to reach:

    clip    inject one 12 sigma point at a known index and set clip: true, so
            clip_outliers actually removes something and the refit path and
            every masked consumer run.
    uniform pass a dict straight through to fit.yaml's `uniform` block.
    flare   include_flare, with a flare planted in the light curve.
    bump    include_bump, with a bump planted in the light curve.
    bands   one dataset per band. More than one with chromatic exercises the
            per-band ror sites.

    Returns (fit_params, sys_params). The injected outlier is always at index
    OUTLIER_INDEX so tests can assert which point was clipped.
    """
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(11)
    t0 = 2460423.06
    t = np.linspace(2460423.0, 2460423.12, n)
    depth = np.where(np.abs(t - t0) < 0.02, -0.004, 0.0)
    trend = 0.002 * (t - t.mean()) / 0.06

    data_cfg = {}
    for band in bands:
        flux = 1.0 + depth + trend + rng.normal(0, 3e-4, n)
        if use_gp:
            flux = flux + 0.003 * np.sin(2 * np.pi * (t - t[0]) / 0.05)
        if flare:
            spec = _flare(t0)
            dt = t - spec['tpeak']
            flux = flux + 1e-3 * spec['ampl'] * np.exp(
                -0.5 * (dt / (spec['fwhm'] / 2.355))**2) * (dt > -spec['fwhm'])
        if bump:
            spec = _bump(t0)
            flux = flux + 1e-3 * spec['ampl'] * np.exp(
                -0.5 * ((t - spec['tcenter']) / spec['width'])**2)
        if clip:
            # one unmistakable outlier, far outside anything the noise produces
            flux[OUTLIER_INDEX] += 12 * 3e-4
        fn = f'{band}.csv'
        pd.DataFrame({'time': t, 'flux': flux,
                      'fluxerr': np.full(n, 3e-4)}).to_csv(
            os.path.join(root, fn), index=False)
        data_cfg[band] = {'file': fn, 'band': band, 'trend': 1,
                          'binsize': None, 'format': 'generic'}
        if clip:
            data_cfg[band].update(clip=True, clip_nsig=5)

    fit_params = {
        'data': data_cfg,
        'planets': 'c',
        'tc_pred': t0,
        'tc_pred_unc': 0.02,
        'chromatic': len(bands) > 1,
        'fixed': ['period', 'u_star'],
        'tune': 5, 'draws': 5, 'chains': 1, 'cores': 1,
    }
    if uniform:
        fit_params['uniform'] = uniform
    if flare:
        fit_params['include_flare'] = True
        fit_params['flare'] = _flare(t0)
    if bump:
        fit_params['include_bump'] = True
        fit_params['bump'] = _bump(t0)
    sys_params = {
        'star': {'teff': [5675, 75], 'logg': [4.2, 0.2], 'feh': [0.0, 0.5]},
        'planets': {'c': {'b': [0.15, 0.15], 'dur': [0.04, 0.005],
                          'period': [14.334894, 3e-5], 'ror': [0.06, 0.01],
                          't0': [2458602.5025, 0.0022]}},
    }
    if use_gp:
        fit_params['use_gp'] = True
        fit_params['gp'] = {
            'log_amp': 0.0, 'log_amp_unc': 6.0, 'log_amp_prior': 'uniform',
            'log_scale': -2.0, 'log_scale_unc': 6.0, 'log_scale_prior': 'uniform',
        }
    with open(os.path.join(root, 'fit.yaml'), 'w') as f:
        yaml.safe_dump(fit_params, f)
    with open(os.path.join(root, 'sys.yaml'), 'w') as f:
        yaml.safe_dump(sys_params, f)
    return fit_params, sys_params


@pytest.fixture(scope='module')
def make_project_module():
    return make_project


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
