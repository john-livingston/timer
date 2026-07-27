import matplotlib
matplotlib.use('Agg')

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pytest

from timer import io, plot


def _fit_with_chunk_offsets():
    """A fit stub whose design matrix is one covariate, one trend column and
    two chunk offsets, in the order read_generic appends them.

    Deriving the block sizes from the config alone gives ncovariates = 4 - 1 =
    3, so the covariate panel would show the trend and the first chunk offset
    and the trend panel would show the second chunk offset.
    """
    n = 12
    x = np.linspace(0.0, 1.0, n)
    covariate = np.sin(3 * x)
    trend = x - x.mean()
    chunk0 = np.where(np.arange(n) < 6, 1.0, 0.0)
    chunk1 = 1.0 - chunk0
    X = np.column_stack([covariate, trend, chunk0, chunk1])
    layout = {'covariates': 1, 'trend': 1, 'spline': 0, 'bias': 0, 'chunk': 2}

    class _Fit:
        use_gp = False
        masks = {'g': None}
        map_soln = {'g_weights': np.array([2.0, 3.0, 4.0, 5.0])}
        data = {'g': dict(x=x, X=X, ncols=layout)}
        fit_params = {'data': {'g': dict(trend=1, spline=False, spline_knots=5,
                                         add_bias=False, chunk_offset=True,
                                         chunk_thresh=0.02)}}

    return _Fit(), X


def _panel(fig, title):
    return next(ax for ax in fig.axes if ax.get_title() == title)


def test_systematics_plots_the_trend_column_in_the_trend_panel():
    fit, X = _fit_with_chunk_offsets()

    fig = plot.systematics(fit, 'g', style=2)

    ax = _panel(fig, 'trend')
    assert len(ax.lines) == 1
    assert ax.lines[0].get_ydata() == pytest.approx(X[:, 1])


def test_systematics_plots_only_the_covariate_in_the_covariate_panel():
    fit, X = _fit_with_chunk_offsets()

    fig = plot.systematics(fit, 'g', style=2)

    ax = _panel(fig, 'covariates')
    # plot_basis draws one line per basis vector, then a black sum line
    basis_lines = [line for line in ax.lines if line.get_color() != 'k']
    assert len(basis_lines) == 1
    assert basis_lines[0].get_ydata() == pytest.approx(X[:, 0])


def test_light_curve_gp_overlay_uses_the_same_noise_diagonal_as_the_fit():
    """plot.light_curve rebuilds the GP from the posterior to draw the trend.

    lcjit is already exp(log_sigma_lc), so exponentiating it a second time
    gives exp(2*exp(log_sigma)) where the model used exp(2*log_sigma). The
    curve drawn on fit.png is then conditioned on a much larger noise
    diagonal, and so is smoother than the GP actually subtracted in
    *-cor.csv.

    The expectation is built from celerite2 directly with the model's own
    diagonal, not from plot.py.
    """
    from celerite2 import GaussianProcess, terms

    n, log_sigma = 12, np.log(0.5)
    x = np.linspace(0.0, 1.0, n)
    yerr = np.full(n, 0.3)
    y = np.sin(6 * x)
    data = dict(x=x, y=y, yerr=yerr, x_hr=np.linspace(0, 1, 500), ref_time=0.0)

    zeros = np.zeros(n)
    soln = {'g_mean': np.array(0.0), 'g_log_sigma_lc': np.array(log_sigma),
            'g_lm': zeros, 'g_light_curves': zeros,
            'g_light_curves_hr': np.zeros(500)}
    trace = az.from_dict(posterior={
        'g_mean': np.zeros((1, 3)),
        'g_log_sigma_lc': np.full((1, 3), log_sigma),
        'g_lm': np.zeros((1, 3, n)),
        'g_light_curves': np.zeros((1, 3, n, 1)),
        'g_light_curves_hr': np.zeros((1, 3, 500, 1)),
        'gp_log_amp': np.zeros((1, 3)),              # 10**0 = amplitude 1
        'gp_log_scale': np.full((1, 3), np.log10(0.05)),
    })

    plot.light_curve(data, 'g', soln, 1, trace=trace, use_gp=True)
    drawn = next(line for ax in plt.gcf().axes for line in ax.lines
                 if line.get_label() == 'systematics').get_ydata()

    gp = GaussianProcess(terms.Matern32Term(sigma=1.0, rho=0.05))
    gp.compute(x, diag=np.exp(2 * log_sigma) + yerr**2)   # model.py's diagonal
    assert drawn == pytest.approx(gp.predict(y), abs=1e-9)


def test_systematics_plots_the_gp_when_there_is_no_design_matrix():
    """A GP only fit has no design matrix, and therefore no weights site
    either, so reading X.shape or {name}_weights raises.

    sample() calls plot_systematics for every dataset, so this takes down the
    whole run after the sampling has already finished.
    """
    n = 12
    x = np.linspace(0.0, 1.0, n)
    gp_pred = np.sin(5 * x)

    class _Fit:
        use_gp = True
        masks = {'g': None}
        map_soln = {'g_gp_pred': gp_pred}
        data = {'g': dict(x=x, X=None, ncols=dict.fromkeys(io.COLUMN_BLOCKS, 0))}
        fit_params = {'data': {'g': dict(trend=None, spline=False, spline_knots=5,
                                         add_bias=False, chunk_offset=False,
                                         chunk_thresh=0.02)}}

    fig = plot.systematics(_Fit(), 'g', style=2)

    ax = _panel(fig, 'GP')
    assert ax.lines[0].get_ydata() == pytest.approx(gp_pred)


def test_systematics_sums_the_whole_design_matrix():
    """The sum panel is the full linear model, chunk offsets included, so it
    must not be restricted to the blocks that got their own panel."""
    fit, X = _fit_with_chunk_offsets()
    w = fit.map_soln['g_weights']

    fig = plot.systematics(fit, 'g', style=2)

    ax = _panel(fig, 'sum')
    assert ax.lines[0].get_ydata() == pytest.approx(X @ w)
