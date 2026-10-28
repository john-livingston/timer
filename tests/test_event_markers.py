"""Configuration-to-figure checks for predicted event markers."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import make_project
from timer import cache, util
from timer.fit import TransitFit

@pytest.fixture
def make_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(util.ld, 'claret', lambda *a: np.array([0.3, 0.01, 0.2, 0.01]))
    def factory(**options):
        params, system = make_project(tmp_path)
        params.update(options)
        return TransitFit(system, params, wd=str(tmp_path))
    yield factory
    plt.close('all')

@pytest.mark.parametrize('mid,edges,count', [(False, False, 0), (True, False, 1), (False, True, 2), (True, True, 3)])
def test_independent_marker_switches(make_fit, mid, edges, count):
    fit = make_fit(plot_midtransit=mid, plot_ingress_egress=edges)
    plt.figure()
    fit.plot_data()
    lines = [line for line in plt.gca().lines if line.get_label() in ('midtransit', 'ingress', 'egress')]
    assert len(lines) == count
    expected = {'midtransit': (0.06, '-'), 'ingress': (0.04, '--'), 'egress': (0.08, '--')}
    for line in lines:
        position, style = expected[line.get_label()]
        assert line.get_xdata() == pytest.approx([position, position], abs=1e-8)
        assert line.get_linestyle() == style
    assert (Path(fit.outdir) / 'data.png').is_file()
    assert 'g' in plt.gca().get_legend_handles_labels()[1]

def test_multi_planet_markers_use_each_ephemeris_and_duration(make_fit):
    fit = make_fit(plot_midtransit=True, plot_ingress_egress=True)
    fit.planets = 'bc'
    fit.nplanets = 2
    fit.priors.update(t0=np.array([0.02, 1.09]), period=np.array([2., 1.]), dur=np.array([0.02, 0.04]))
    plt.figure()
    fit.plot_data()
    markers = {line.get_label(): line.get_xdata()[0] for line in plt.gca().lines if not line.get_label().startswith('_')}
    assert markers == pytest.approx({'midtransit b': .02, 'ingress b': .01, 'egress b': .03,
                                     'midtransit c': .09, 'ingress c': .07, 'egress c': .11})

def test_plot_options_do_not_invalidate_fit_cache(make_fit):
    fit = make_fit()
    before = cache.compute_keys(fit.fit_params, fit.sys_params, fit.wd)
    fit.fit_params.update(plot_midtransit=True, plot_ingress_egress=True)
    assert cache.compute_keys(fit.fit_params, fit.sys_params, fit.wd) == before
