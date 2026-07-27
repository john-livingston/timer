import os

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.slow


def _run(wd, fit_params, sys_params):
    from timer import fit
    tf = fit.TransitFit(sys_params, fit_params, wd=str(wd))
    tf.build_model(verbose=False, plot=False)
    tf.clip_outliers()
    tf.sample(plot_fit=False, plot_systematics=False)
    tf.save_results()
    return tf


def _ic_rows(outdir):
    rows = {}
    for line in open(os.path.join(outdir, 'ic.txt')):
        key, value = line.split()
        rows[key] = float(value)
    return rows


@pytest.fixture(scope='module')
def plain_run(tmp_path_factory, make_project_module):
    wd = tmp_path_factory.mktemp('e2e') / 'plain'
    fit_params, sys_params = make_project_module(str(wd))
    return wd, _run(wd, fit_params, sys_params)


@pytest.fixture(scope='module')
def gp_run(tmp_path_factory, make_project_module):
    wd = tmp_path_factory.mktemp('e2e') / 'gp'
    fit_params, sys_params = make_project_module(str(wd), use_gp=True)
    return wd, _run(wd, fit_params, sys_params)


def test_the_pipeline_produces_every_documented_output(plain_run):
    wd, _ = plain_run
    out = os.path.join(wd, 'out')
    for fn in ('summary.csv', 'tc.txt', 'ic.txt', 'map.pkl', 'model.pkl',
               'mask.pkl', 'trace.pkl', 'cache.json',
               'posterior_samples.csv.gz', 'plain-g-cor.csv'):
        assert os.path.exists(os.path.join(out, fn)), fn


def test_every_resume_artifact_is_recorded_in_the_manifest(plain_run):
    """Otherwise the next run silently recomputes everything, which looks like
    the cache working rather than like the cache never being written."""
    from timer import cache

    wd, tf = plain_run
    manifest = cache.read_manifest(os.path.join(wd, 'out'))
    assert manifest['model.pkl'] == tf._cache_keys['model']
    assert manifest['map.pkl'] == tf._cache_keys['model']
    assert manifest['mask.pkl'] == tf._cache_keys['model']
    assert manifest['trace.pkl'] == tf._cache_keys['run']


def test_transit_time_is_reported_in_the_data_native_system(plain_run):
    """tc.txt must carry the absolute BJD, not the ref_time subtracted value.

    The transit is injected at 2460423.06, so anything near 0.06 means
    ref_time was dropped and anything far from either means the fit failed.
    """
    wd, _ = plain_run
    planet, tc, unc = open(os.path.join(wd, 'out', 'tc.txt')).read().split()
    assert planet == 'c'
    assert float(tc) == pytest.approx(2460423.06, abs=0.01)
    assert float(unc) >= 0.0


def test_get_ic_agrees_with_the_bic_written_to_ic_txt(plain_run):
    """get_ic and save_results counted parameters two different ways, and one
    of them used an attribute PyMC does not have."""
    wd, tf = plain_run
    assert tf.get_ic(ic='BIC') == pytest.approx(_ic_rows(os.path.join(wd, 'out'))['BIC'],
                                                abs=0.01)


def test_a_fit_without_a_gp_reports_no_edf_rows(plain_run):
    wd, _ = plain_run
    assert set(_ic_rows(os.path.join(wd, 'out'))) == {'BIC', 'AIC', 'AICc'}


def test_a_gp_fit_reports_far_more_degrees_of_freedom_than_hyperparameters(gp_run):
    """The reason the correction exists. Two hyperparameters buy a smoother
    with tens of degrees of freedom, and the uncorrected BIC prices it at two.
    """
    wd, tf = gp_run
    rows = _ic_rows(os.path.join(wd, 'out'))
    n_hyper = tf._count_gp_hyperparams()
    assert n_hyper == 2
    assert rows['edf'] > 5 * n_hyper
    assert rows['nparams_edf'] == pytest.approx(rows['nparams'] - n_hyper + rows['edf'],
                                                abs=0.01)
    assert rows['BIC_edf'] > rows['BIC']


def test_the_corrected_light_curve_has_the_gp_removed(gp_run):
    """Subtracting only the linear model leaves the GP trend in every
    *-cor.csv, which is the file downstream analysis reads.

    Discriminating form: if the GP is already gone, subtracting it again can
    only add scatter; if it is still there, subtracting it would remove it and
    the scatter would drop.
    """
    wd, tf = gp_run
    cor = pd.read_csv(os.path.join(wd, 'out', 'gp-g-cor.csv'))
    gp_pred = np.asarray(tf.map_soln['g_gp_pred']) * 1e-3   # ppt to relative flux

    assert len(cor) == len(gp_pred)
    assert np.std(cor.y.values - gp_pred) > 1.2 * np.std(cor.y.values)


def test_the_corrected_light_curve_keeps_the_transit(gp_run):
    """The control for the test above: detrending must not also remove the
    signal. The transit is injected 4 ppt deep between 2460423.04 and .08.
    """
    wd, _ = gp_run
    cor = pd.read_csv(os.path.join(wd, 'out', 'gp-g-cor.csv'))
    x = cor.x.values
    in_transit = np.abs(x - 2460423.06) < 0.015
    out_of_transit = np.abs(x - 2460423.06) > 0.025
    depth = cor.y.values[out_of_transit].mean() - cor.y.values[in_transit].mean()
    assert depth == pytest.approx(0.004, abs=0.002)


def test_residual_scatter_is_at_the_injected_noise_level(gp_run):
    """End to end sanity: the GP fit should leave white noise behind, and the
    data was generated with 300 ppm of it."""
    from timer import util

    wd, tf = gp_run
    resid = util.get_residuals('g', tf.data['g']['y'], tf.map_soln,
                               mask=tf.masks['g'])
    assert np.std(resid) * 1e-3 < 6e-4
