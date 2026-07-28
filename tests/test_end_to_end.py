import os
import sys
from unittest import mock

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from timer import plot


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


def test_summary_reports_every_sampled_parameter(plain_run):
    """summary.csv is the headline table, and get_var_names decides its rows.

    Asserting the file exists cannot see a dropped row: removing the jitter or
    the GP hyperparameters from get_var_names silently shrinks the table.
    """
    wd, _ = plain_run
    index = set(pd.read_csv(os.path.join(wd, 'out', 'summary.csv'),
                            index_col=0).index)
    # period and u_star are fixed in the fixture, so they must be absent
    assert {'t0[0]', 'ror[0]', 'b[0]', 'dur[0]', 'g_log_sigma_lc'} <= index
    assert not any(name.startswith('period') for name in index)


def test_summary_reports_the_gp_hyperparameters(gp_run):
    wd, _ = gp_run
    index = set(pd.read_csv(os.path.join(wd, 'out', 'summary.csv'),
                            index_col=0).index)
    assert {'gp_log_amp', 'gp_log_scale'} <= index


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

    The bound is wide because the fixture samples five draws, so the MAP GP
    absorbs a variable share of the transit. The claim being tested is that
    the transit is still there at roughly its injected depth: subtracting the
    light curve along with the systematics would leave nothing at all.
    """
    wd, tf = gp_run
    cor = pd.read_csv(os.path.join(wd, 'out', 'gp-g-cor.csv'))
    x = cor.x.values
    in_transit = np.abs(x - 2460423.06) < 0.015
    out_of_transit = np.abs(x - 2460423.06) > 0.025
    depth = cor.y.values[out_of_transit].mean() - cor.y.values[in_transit].mean()
    assert 0.002 < depth < 0.010
    # the file is in relative flux, not the ppt the model works in: the
    # baseline sits at 1 and the errors are the per point errors scaled to
    # match. A difference of means cannot see either, so assert them here.
    assert cor.y.values[out_of_transit].mean() == pytest.approx(1.0, abs=2e-3)
    # errors are the ones the likelihood used, photometric plus fitted jitter,
    # converted from ppt to relative flux
    jitter = np.exp(np.squeeze(tf.map_soln['g_log_sigma_lc']))
    expected = np.sqrt(tf.data['g']['yerr']**2 + jitter**2) * 1e-3
    assert cor.yerr.values == pytest.approx(expected, rel=1e-9)
    assert (cor.yerr.values > tf.data['g']['yerr'] * 1e-3).all()


def test_systematics_panels_slice_the_real_design_matrix(gp_run):
    """The layout comes all the way from io.read_generic through load_data.

    This fit has one trend column and no covariates, so the trend panel must
    show column 0 rather than whatever a config derived offset would land on.
    """
    _, tf = gp_run

    fig = plot.systematics(tf, 'g', style=2)

    ax = next(a for a in fig.axes if a.get_title() == 'trend')
    assert ax.lines[0].get_ydata() == pytest.approx(tf.data['g']['X'][:, 0])


@pytest.fixture(scope='module')
def cli_run(tmp_path_factory, make_project_module):
    """The command line entry point, start to finish, on a GP configuration."""
    from timer import fit
    wd = tmp_path_factory.mktemp('cli') / 'proj'
    make_project_module(str(wd), use_gp=True)
    with mock.patch.object(sys, 'argv', ['timer-fit', str(wd)]):
        code = fit.cli()
    return wd, code


def test_the_cli_reports_success(cli_run):
    _, code = cli_run
    assert code == 0


def test_the_cli_produces_every_plot(cli_run):
    """cli() catches plotting failures and still returns 0, so the exit code
    alone does not tell you the corner and trace plots were written."""
    wd, _ = cli_run
    for fn in ('data.png', 'fit.png', 'corner.png', 'trace.png', 'sys-g.png'):
        assert os.path.exists(os.path.join(wd, 'out', fn)), fn


def test_residual_scatter_is_at_the_injected_noise_level(gp_run):
    """End to end sanity: the GP fit should leave white noise behind, and the
    data was generated with 300 ppm of it."""
    from timer import util

    wd, tf = gp_run
    resid = util.get_residuals('g', tf.data['g']['y'], tf.map_soln,
                               mask=tf.masks['g'])
    assert np.std(resid) * 1e-3 < 6e-4
