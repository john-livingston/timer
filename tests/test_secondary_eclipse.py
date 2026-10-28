"""Physical occultation and configuration regressions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytest
from conftest import make_project
from timer import cache, model, util
from timer.fit import TransitFit

@pytest.mark.parametrize('enabled', [False, True])
def test_occultation_has_independent_depth_and_preserves_primary_epoch(enabled):
    # Uniform stellar disk gives an analytic primary depth ror**2, diluted
    # by the companion flux. A full occultation removes exactly eclipse_depth.
    x = np.array([0., .25, .5, .75])
    d = dict(x=x, y=np.array([-9.98, 0., -2., 0.]), yerr=np.ones(4)*.1,
             X=None, texp=.001, x_hr=x, band='g')
    priors = dict(t0=np.array([0.]), period=np.array([1.]), dur=np.array([.1]),
                  ror=np.array([.1]), b=np.array([0.]),
                  u_star={'g': np.array([0., 0.])}, eclipse_depth=np.array([.002]))
    fixed = ['t0', 'period', 'dur', 'ror', 'b', 'u_star', 'eclipse_depth']
    m, soln = model.build({'g': d}, priors, 1, fixed=fixed,
                          masks={'g': None}, secondary_eclipse=enabled,
                          include_mean=False, use_custom_optimizer=False)
    expected = [-9.98, 0., -2., 0.] if enabled else [-10., 0., 0., 0.]
    np.testing.assert_allclose(soln['g_light_curves'].ravel(), expected, atol=1e-6)
    np.testing.assert_allclose(soln['g_light_curves_hr'].ravel(), expected, atol=1e-6)
    assert all(np.isfinite(list(m.point_logps(m.initial_point()).values())))

@pytest.fixture
def make_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(util.ld, 'claret', lambda *a: np.array([.3, .01, .2, .01]))
    def factory(**options):
        params, system = make_project(tmp_path)
        params.update(options)
        return TransitFit(system, params, wd=str(tmp_path))
    yield factory
    plt.close('all')

def test_wrapper_forwards_eclipse_and_depth_prior(make_fit, monkeypatch):
    fit = make_fit(secondary_eclipse=True, uniform={'eclipse_depth': [0., .01]})
    def capture(data, priors, nplanets, **kwargs):
        assert kwargs['secondary_eclipse'] is True
        assert priors['eclipse_depth'] == pytest.approx(.005)
        assert priors['eclipse_depth_unc'] == pytest.approx(.01)
        raise RuntimeError('checked')
    monkeypatch.setattr(model, 'build', capture)
    with pytest.raises(RuntimeError, match='checked'):
        fit.build_model(plot=False)

def test_secondary_event_markers_are_at_phase_half(make_fit):
    fit = make_fit(secondary_eclipse=True, plot_midtransit=True, plot_ingress_egress=True)
    fit.priors.update(t0=np.array([-.44]), period=np.array([1.]), dur=np.array([.04]))
    plt.figure()
    fit.plot_data()
    markers = {l.get_label(): l.get_xdata()[0] for l in plt.gca().lines if not l.get_label().startswith('_')}
    assert markers == pytest.approx({'mid-eclipse': .06, 'ingress': .04, 'egress': .08})

@pytest.mark.parametrize('bounds', [[-.01, .01], [0., 1.1], [0., float('nan')]])
def test_invalid_depth_bounds_rejected(make_fit, bounds):
    with pytest.raises(ValueError, match='eclipse_depth'):
        make_fit(secondary_eclipse=True, uniform={'eclipse_depth': bounds})

def test_secondary_option_invalidates_model_cache(make_fit):
    fit = make_fit()
    before = cache.compute_keys(fit.fit_params, fit.sys_params, fit.wd)
    fit.fit_params['secondary_eclipse'] = True
    assert cache.compute_keys(fit.fit_params, fit.sys_params, fit.wd)['model'] != before['model']


def test_default_center_is_an_eclipse_guess(make_fit):
    fit = make_fit(secondary_eclipse=True)
    # Remove the explicit key to exercise the actual omitted-value contract.
    del fit.fit_params['tc_pred']
    fit.set_priors()
    center = fit.priors['t0'] + fit.priors['period'] / 2
    assert center == pytest.approx([np.mean(fit.data['g']['x'])])


def test_eclipse_rejects_multiple_planets_before_model_build():
    with pytest.raises(ValueError, match='one planet'):
        model.build({}, {}, 2, secondary_eclipse=True)


@pytest.mark.slow
def test_chromatic_eclipse_recovers_depth_and_exports(tmp_path, monkeypatch):
    import exoplanet as xo
    import pandas as pd
    monkeypatch.setattr(util.ld, 'claret', lambda *a: np.array([.3, .01, .2, .01]))
    params, system = make_project(tmp_path, n=40, bands=('g', 'r'))
    params.update(secondary_eclipse=True, chromatic=True,
                  fixed=['t0', 'period', 'dur', 'ror', 'b', 'u_star'],
                  uniform={'eclipse_depth': [0., .01]},
                  tune=30, draws=30, chains=1, cores=1)
    params['tc_pred'] = 2460423.06
    system['planets']['c'].update(period=[1., .001], ror=[.1, .01])
    times = np.linspace(.46, .66, 40)
    for band, depth in [('g', .002), ('r', .003)]:
        orbit = xo.orbits.KeplerianOrbit(period=1., t0=.06, duration=.04, ror=.1, b=.15)
        lc = xo.SecondaryEclipseLightCurve([.3, .2], [0., 0.], depth / (1-depth) / .1**2)
        values = lc.get_light_curve(orbit=orbit, r=.1, t=times, texp=np.median(np.diff(times)), use_in_transit=False).eval().ravel()
        pd.DataFrame(dict(time=2460423+times, flux=1+values, fluxerr=np.full(40, 2e-5))).to_csv(tmp_path / f'{band}.csv', index=False)
        params['data'][band]['trend'] = None
    fit = TransitFit(system, params, wd=str(tmp_path))
    fit.build_model(plot=False)
    for band, depth in [('g', .002), ('r', .003)]:
        assert fit.map_soln[f'eclipse_depth_{band}'] == pytest.approx(depth, abs=5e-5)
    fit.sample(plot_fit=False, plot_systematics=False)
    assert any(i.startswith('eclipse_depth_g') for i in fit.summary.index)
    assert any(i.startswith('eclipse_depth_r') for i in fit.summary.index)
    fit.plot_corner()
    fit.plot_trace()
    fit.save_results()
    for name in ['summary.csv', 'corner.png', 'trace.png']:
        assert (tmp_path / 'out' / name).is_file()
    plt.close('all')
