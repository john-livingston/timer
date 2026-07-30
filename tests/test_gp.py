import logging

import numpy as np
import pytest


def _dataset(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 0.2, n))
    return {
        'g': dict(x=x, y=np.zeros(n), yerr=np.full(n, 0.2),
                  X=None, texp=0.001, x_hr=x, band='g', ref_time=0.0)
    }


def _soln(log_amp, log_scale, log_sigma_lc=np.log(0.35)):
    return {
        'gp_log_amp': np.array(float(log_amp)),
        'gp_log_scale': np.array(float(log_scale)),
        'g_log_sigma_lc': np.array(float(log_sigma_lc)),
    }


def _dense_matern32(x, amp, rho):
    """The Matern 3/2 covariance, written out in numpy.

    Shares nothing with celerite2, so the references built from it are
    independent of the implementation under test.
    """
    r = np.sqrt(3) * np.abs(x[:, None] - x[None, :]) / rho
    return amp**2 * (1 + r) * np.exp(-r)


def _dense_joint_edf(x, yerr, X, amp, rho, jitter=0.35):
    """tr(P + K C^-1 (I - P)), the joint effective degrees of freedom of a
    parametric mean plus a GP, formed densely."""
    K = _dense_matern32(x, amp, rho)
    C = K + np.diag(jitter**2 + yerr**2)
    Ci = np.linalg.inv(C)
    P = X @ np.linalg.inv(X.T @ Ci @ X) @ X.T @ Ci
    return np.trace(P + K @ Ci @ (np.eye(len(x)) - P))


def test_edf_matches_the_dense_smoother_trace():
    """The identity edf = n - tr(S (K+S)^-1) must agree with the direct
    tr(K (K+S)^-1) built from the Matern 3/2 kernel by hand. This is the whole
    correctness claim of the feature."""
    from timer import model

    data = _dataset()
    x = data['g']['x']
    yerr = data['g']['yerr']
    amp, rho, jit = 1.3, 0.018, 0.35
    soln = _soln(np.log10(amp), np.log10(rho), np.log(jit))

    edf = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)['g']

    diag = jit**2 + yerr**2
    d = x[:, None] - x[None, :]
    r = np.sqrt(3) * np.abs(d) / rho
    K = amp**2 * (1 + r) * np.exp(-r)
    edf_dense = np.trace(K @ np.linalg.inv(K + np.diag(diag)))

    assert edf == pytest.approx(edf_dense, abs=1e-6)


def test_negligible_amplitude_gives_near_zero_edf():
    """A GP that explains nothing must be charged nothing."""
    from timer import model

    data = _dataset()
    soln = _soln(log_amp=-6.0, log_scale=-2.0)   # amplitude 1e-6 against noise ~0.4
    edf = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)['g']
    assert 0.0 <= edf < 0.1


def test_dominant_amplitude_approaches_the_number_of_points():
    """A GP that interpolates every point costs one degree of freedom each."""
    from timer import model

    data = _dataset()
    n = len(data['g']['x'])
    soln = _soln(log_amp=4.0, log_scale=-3.0)    # huge amplitude, short scale
    edf = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)['g']
    assert edf > 0.9 * n
    assert edf <= n + 1e-6


def test_the_mask_restricts_the_computation():
    """Clipped points are not in the likelihood, so they cannot cost degrees
    of freedom either."""
    from timer import model

    data = _dataset()
    n = len(data['g']['x'])
    mask = np.zeros(n, dtype=bool)
    mask[:20] = True
    soln = _soln(log_amp=0.0, log_scale=-2.0)
    edf = model.compute_gp_edf(soln, data, {'g': mask}, gp_config=None)['g']
    assert edf <= 20.0


def test_per_dataset_hyperparameters_are_looked_up_per_dataset():
    """With gp.per_dataset the amplitude site is named gp_log_amp_<name>, and
    reading the shared name instead would silently use the wrong kernel."""
    from timer import model

    data = _dataset()
    shared = _soln(log_amp=0.0, log_scale=-2.0)
    per_ds = {
        'gp_log_amp_g': np.array(0.0),
        'gp_log_scale': np.array(-2.0),
        'g_log_sigma_lc': np.array(np.log(0.35)),
    }
    expected = model.compute_gp_edf(shared, data, {'g': None}, gp_config=None)['g']
    got = model.compute_gp_edf(per_ds, data, {'g': None},
                               gp_config={'per_dataset': ['log_amp']})['g']
    assert got == pytest.approx(expected)


def test_returns_none_and_warns_above_max_points(caplog):
    """O(n^2) is fine at n~140 and prohibitive at survey scale, so it must skip
    rather than silently stall."""
    from timer import model

    data = _dataset()
    soln = _soln(log_amp=0.0, log_scale=-2.0)
    with caplog.at_level(logging.WARNING):
        result = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None,
                                      max_points=10)
    assert result is None
    assert 'max_points=10' in caplog.text
    assert 'dataset g' in caplog.text
    assert str(len(data['g']['x'])) in caplog.text


def test_edf_matches_the_joint_hat_matrix_trace():
    """The joint effective degrees of freedom of a parametric mean plus a GP is
    tr(P + K C^-1 (I - P)), not p + tr(K C^-1). The difference is the overlap
    between the GP and the design, which approaches p whenever the GP can
    reproduce the design columns, and on a polynomial trend that is most of the
    low frequency power.
    """
    from timer import model

    rng = np.random.default_rng(0)
    n, p = 60, 3
    x = np.sort(rng.uniform(0.0, 0.2, n))
    yerr = np.full(n, 0.35)
    X = np.column_stack([np.ones(n), x - x.mean(), rng.normal(size=n)])
    amp, rho = 1.3, 0.02

    data = {'g': dict(x=x, y=np.zeros(n), yerr=yerr, X=X,
                      texp=0.001, x_hr=x, band='g', ref_time=0.0)}
    # _soln defaults log_sigma_lc to log(0.35), the jitter the reference assumes
    soln = _soln(np.log10(amp), np.log10(rho))

    edf = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)['g']

    # compute_gp_edf returns the GP's share; the p design columns are already
    # counted in nparams, so p + edf is the joint figure.
    # abs=1e-6 because celerite2's eps approximation caps agreement near 7e-8.
    assert p + edf == pytest.approx(
        _dense_joint_edf(x, yerr, X, amp, rho), abs=1e-6)


def test_edf_is_strictly_below_the_gp_alone_trace_when_a_design_is_present():
    """The overlap is non-negative, so correcting for it can only reduce the
    count. Pins the sign of the subtraction, and that it cannot exceed p."""
    from timer import model

    rng = np.random.default_rng(1)
    n = 50
    x = np.sort(rng.uniform(0.0, 0.2, n))
    yerr = np.full(n, 0.3)
    X = np.column_stack([np.ones(n), x - x.mean()])
    soln = _soln(0.0, -1.7, np.log(0.3))
    base = dict(x=x, y=np.zeros(n), yerr=yerr, texp=0.001, x_hr=x,
                band='g', ref_time=0.0)

    with_design = model.compute_gp_edf(
        soln, {'g': dict(base, X=X)}, {'g': None}, gp_config=None)['g']
    without = model.compute_gp_edf(
        soln, {'g': dict(base, X=None)}, {'g': None}, gp_config=None)['g']

    assert with_design < without
    assert without - with_design <= X.shape[1] + 1e-9, 'overlap cannot exceed p'


def test_edf_is_unchanged_for_a_gp_only_fit():
    """X is None means p = 0, so there is nothing to overlap with and the value
    must stay the plain smoother trace tr(K C^-1)."""
    from timer import model

    rng = np.random.default_rng(2)
    n = 40
    x = np.sort(rng.uniform(0.0, 0.2, n))
    yerr = np.full(n, 0.3)
    amp, rho, jitter = 1.0, 10**-1.7, 0.3
    soln = _soln(np.log10(amp), np.log10(rho), np.log(jitter))
    data = {'g': dict(x=x, y=np.zeros(n), yerr=yerr, X=None, texp=0.001,
                      x_hr=x, band='g', ref_time=0.0)}

    edf = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)['g']

    K = _dense_matern32(x, amp, rho)
    Ci = np.linalg.inv(K + np.diag(jitter**2 + yerr**2))
    assert edf == pytest.approx(np.trace(K @ Ci), abs=1e-6)


def test_edf_returns_none_for_a_rank_deficient_design(caplog):
    """A singular X^T A leaves the overlap undefined, so no *_edf row may be
    written at all. Reached in practice by add_bias with chunk_offset, whose
    chunk indicators sum to the bias column, and by clipping emptying a chunk
    indicator."""
    from timer import model

    rng = np.random.default_rng(3)
    n = 40
    x = np.sort(rng.uniform(0.0, 0.2, n))
    yerr = np.full(n, 0.3)
    col = x - x.mean()
    X = np.column_stack([col, col])          # exactly collinear
    soln = _soln(0.0, -1.7, np.log(0.3))
    data = {'g': dict(x=x, y=np.zeros(n), yerr=yerr, X=X, texp=0.001,
                      x_hr=x, band='g', ref_time=0.0)}

    with caplog.at_level(logging.WARNING):
        result = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)

    assert result is None
    assert 'dataset g' in caplog.text and 'design' in caplog.text


def test_edf_returns_none_for_a_structurally_singular_design(caplog):
    """The realistic rank deficient case, which exact column duplication does
    not reach.

    `add_bias: true` with `chunk_offset: true` produces [ones, chunk0, chunk1]
    where the chunk indicators sum to the bias column: rank p-1, but no two
    columns are equal. np.linalg.solve only raises LinAlgError when LAPACK
    finds an exactly zero pivot, and rounding on a matrix like this leaves it
    nominally invertible, so relying on the exception returns a confident wrong
    number instead of skipping the rows.
    """
    from timer import model

    rng = np.random.default_rng(5)
    n = 40
    x = np.sort(rng.uniform(0.0, 0.2, n))
    yerr = np.full(n, 0.3)
    chunk0 = np.where(np.arange(n) < n // 2, 1.0, 0.0)
    X = np.column_stack([np.ones(n), chunk0, 1.0 - chunk0])
    assert np.linalg.matrix_rank(X) == 2, 'fixture must be rank deficient'
    soln = _soln(0.0, -1.7, np.log(0.3))
    data = {'g': dict(x=x, y=np.zeros(n), yerr=yerr, X=X, texp=0.001,
                      x_hr=x, band='g', ref_time=0.0)}

    with caplog.at_level(logging.WARNING):
        result = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)

    assert result is None
    assert 'dataset g' in caplog.text and 'design' in caplog.text


def test_edf_masks_the_design_matrix_the_same_as_x_and_yerr():
    """Every other test here pairs a design matrix with an all-True mask or a
    real mask with X=None, so none exercises the masking of X itself. This mask
    keeps every other point, so a design matrix not restricted the same way as
    x and yerr changes both the shapes and the number.
    """
    from timer import model

    rng = np.random.default_rng(4)
    n, p = 50, 2
    x = np.sort(rng.uniform(0.0, 0.2, n))
    yerr = np.full(n, 0.3)
    X = np.column_stack([np.ones(n), x - x.mean()])
    mask = np.zeros(n, dtype=bool)
    mask[::2] = True
    amp, rho = 1.1, 0.02

    data = {'g': dict(x=x, y=np.zeros(n), yerr=yerr, X=X,
                      texp=0.001, x_hr=x, band='g', ref_time=0.0)}
    soln = _soln(np.log10(amp), np.log10(rho))

    edf = model.compute_gp_edf(soln, data, {'g': mask}, gp_config=None)['g']

    assert p + edf == pytest.approx(
        _dense_joint_edf(x[mask], yerr[mask], X[mask], amp, rho), abs=1e-6)


def test_gp_predictions_without_a_mean_treat_it_as_zero():
    """{name}_mean is only a model site when include_mean=True, so reading it
    unguarded makes every include_mean=False GP fit crash after optimization."""
    from timer import model

    data = _dataset(n=20)
    n = len(data['g']['x'])
    data['g']['y'] = np.linspace(-1.0, 1.0, n)
    soln = _soln(log_amp=0.0, log_scale=-2.0)
    soln['g_light_curves'] = np.zeros(n)

    out = model._add_gp_predictions(dict(soln), data, {'g': None}, None)

    assert out['g_gp_pred'].shape == (n,)


def test_gp_predictions_use_the_mean_when_it_is_present():
    """The control: the mean shifts the residuals the GP conditions on, so a
    guard that dropped it would change the prediction."""
    from timer import model

    data = _dataset(n=20)
    n = len(data['g']['x'])
    data['g']['y'] = np.linspace(-1.0, 1.0, n)
    soln = _soln(log_amp=0.0, log_scale=-2.0)
    soln['g_light_curves'] = np.zeros(n)

    without = model._add_gp_predictions(dict(soln), data, {'g': None}, None)['g_gp_pred']
    with_mean = model._add_gp_predictions(
        dict(soln, g_mean=np.array(5.0)), data, {'g': None}, None)['g_gp_pred']

    assert not np.allclose(without, with_mean)


# ------------------------------------------------------ ic.txt integration

def _fit_for_save_results(tmp_path, use_gp, monkeypatch, nparams=13, ndata=50):
    """A TransitFit carrying only what save_results reads."""
    import arviz as az
    from timer import fit

    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.outdir = str(tmp_path)
    tf.planets = 'c'
    tf.nplanets = 1
    tf.ref_time = 2460000.0
    tf.clobber = False
    tf.use_gp = use_gp
    tf.gp_config = None
    tf.data = _dataset(n=20)
    tf.masks = {'g': None}
    tf.map_soln = _soln(log_amp=0.0, log_scale=-2.0)
    # the GP hyperparameters have to be in the posterior, since the edf is now
    # read at the likelihood maximizing draw rather than from map_soln. Every
    # draw carries the same values as map_soln so the expectations stay simple.
    tf.trace = az.from_dict(
        posterior={
            't0': np.full((1, 4, 1), 0.5),
            'gp_log_amp': np.zeros((1, 4)),
            'gp_log_scale': np.full((1, 4), -2.0),
            'g_log_sigma_lc': np.full((1, 4), np.log(0.35)),
        },
        sample_stats={'lp': np.array([[-101.0, -100.0, -102.0, -103.0]])})

    monkeypatch.setattr(fit.util, 'get_map_soln', lambda trace: ({}, -100.0))
    monkeypatch.setattr(fit.TransitFit, '_max_loglike', lambda self: (-100.0, (0, 0)))
    monkeypatch.setattr(fit.TransitFit, '_count_params', lambda self: nparams)
    monkeypatch.setattr(fit.TransitFit, '_count_data', lambda self: ndata)
    monkeypatch.setattr(fit.TransitFit, '_count_gp_hyperparams', lambda self: 2)
    monkeypatch.setattr(fit.TransitFit, 'save_posterior_samples', lambda self: None)
    monkeypatch.setattr(fit.TransitFit, 'save_corrected', lambda self: None)
    return tf


def _fit_with_two_draws(tmp_path, monkeypatch, ll_index):
    """A save_results harness whose two draws have different GP amplitudes.

    Draw 0 is the maximum posterior draw, which is what map_soln holds. Draw 1
    has a larger amplitude and therefore a much larger edf. ll_index says which
    draw the maximized likelihood came from, so pointing it at draw 1 must move
    the reported edf onto draw 1's hyperparameters.
    """
    import arviz as az
    from timer import fit

    n = 20
    data = _dataset(n=n)
    amps = [-1.0, 0.5]                      # log10 amplitude per draw
    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.outdir = str(tmp_path)
    tf.planets, tf.nplanets = 'c', 1
    tf.ref_time, tf.clobber = 2460000.0, False
    tf.use_gp, tf.gp_config = True, None
    tf.data, tf.masks = data, {'g': None}
    # map_soln is the maximum posterior draw, i.e. draw 0
    tf.map_soln = _soln(amps[0], -2.0)
    tf.trace = az.from_dict(
        posterior={
            't0': np.full((1, 2, 1), 0.5),
            'gp_log_amp': np.array([amps]),
            'gp_log_scale': np.full((1, 2), -2.0),
            'g_log_sigma_lc': np.full((1, 2), np.log(0.35)),
        },
        sample_stats={'lp': np.array([[-100.0, -101.0]])})

    monkeypatch.setattr(fit.util, 'get_map_soln', lambda trace: ({}, -100.0))
    monkeypatch.setattr(fit.TransitFit, '_max_loglike',
                        lambda self: (-100.0, ll_index))
    monkeypatch.setattr(fit.TransitFit, '_count_params', lambda self: 13)
    monkeypatch.setattr(fit.TransitFit, '_count_data', lambda self: 50)
    monkeypatch.setattr(fit.TransitFit, '_count_gp_hyperparams', lambda self: 2)
    monkeypatch.setattr(fit.TransitFit, 'save_posterior_samples', lambda self: None)
    monkeypatch.setattr(fit.TransitFit, 'save_corrected', lambda self: None)
    return tf, data, amps


def test_edf_is_measured_at_the_draw_the_likelihood_came_from(tmp_path, monkeypatch):
    """A criterion and its penalty have to describe one parameter vector.

    max_loglike is taken at the likelihood maximizing draw, so the edf has to
    be too. Reading it from map_soln instead mixes a likelihood from one draw
    with a penalty from another; the edf varies by tens of units across a real
    posterior and correlates with the likelihood, so the mismatch systematically
    under-penalizes the GP.
    """
    from timer import model

    tf, data, amps = _fit_with_two_draws(tmp_path, monkeypatch, ll_index=(0, 1))
    tf.save_results()

    rows = _ic_rows(tmp_path)
    at_ll_draw = sum(model.compute_gp_edf(
        _soln(amps[1], -2.0), data, {'g': None}, None).values())
    at_map_draw = sum(model.compute_gp_edf(
        _soln(amps[0], -2.0), data, {'g': None}, None).values())

    assert at_ll_draw - at_map_draw > 1.0, 'fixture must separate the two draws'
    assert rows['edf'] == pytest.approx(at_ll_draw, abs=0.005)


def test_edf_follows_the_likelihood_index_when_it_is_the_first_draw(
        tmp_path, monkeypatch):
    """The control: with the likelihood peaking at draw 0 the edf must be
    draw 0's, so the test above is not simply reading a fixed draw."""
    from timer import model

    tf, data, amps = _fit_with_two_draws(tmp_path, monkeypatch, ll_index=(0, 0))
    tf.save_results()

    rows = _ic_rows(tmp_path)
    expected = sum(model.compute_gp_edf(
        _soln(amps[0], -2.0), data, {'g': None}, None).values())
    assert rows['edf'] == pytest.approx(expected, abs=0.005)


def _ic_rows(tmp_path):
    rows = {}
    for line in (tmp_path / 'ic.txt').read_text().splitlines():
        key, value = line.split()
        rows[key] = float(value)
    return rows


def test_ic_reports_edf_corrected_criteria_for_a_gp_fit(tmp_path, monkeypatch):
    """The uncorrected rows stay, and the corrected ones swap the GP's two
    hyperparameters for the degrees of freedom it actually absorbs."""
    from timer import model

    tf = _fit_for_save_results(tmp_path, use_gp=True, monkeypatch=monkeypatch)
    tf.save_results()

    rows = _ic_rows(tmp_path)
    edf = sum(model.compute_gp_edf(tf.map_soln, tf.data, tf.masks, None).values())

    # an independent floor first, so the row cannot be satisfied by writing
    # zero, or the hyperparameter count, into it
    assert rows['edf'] > 5
    # ic.txt is written to two decimal places, hence the tolerances
    assert rows['edf'] == pytest.approx(edf, abs=0.005)
    assert rows['nparams'] == 13
    assert rows['nparams_edf'] == pytest.approx(13 - 2 + edf, abs=0.005)
    # uncorrected BIC with nparams 13, ndata 50 and logp -100
    assert rows['BIC'] == pytest.approx(200 + 13 * np.log(50), abs=0.01)
    assert rows['BIC_edf'] == pytest.approx(200 + (11 + edf) * np.log(50), abs=0.01)
    assert rows['BIC_edf'] > rows['BIC'], (
        'the correction must penalize the GP, otherwise it is not measuring '
        'the flexibility it adds'
    )


def test_ic_reports_no_edf_rows_for_a_fit_without_a_gp(tmp_path, monkeypatch):
    tf = _fit_for_save_results(tmp_path, use_gp=False, monkeypatch=monkeypatch)
    tf.save_results()

    rows = _ic_rows(tmp_path)
    assert set(rows) == {'BIC', 'AIC', 'AICc'}


def test_save_results_survives_an_edf_failure(tmp_path, monkeypatch, caplog):
    """compute_gp_edf runs inside save_results' `with open(ic.txt)` block and
    reads GP hyperparameters back out of map_soln. If it raises, that must not
    truncate ic.txt and must not skip the saves that follow the block.
    """
    from timer import fit

    tf = _fit_for_save_results(tmp_path, use_gp=True, monkeypatch=monkeypatch)

    def _raise_stale_hyperparam_lookup(*args, **kwargs):
        raise KeyError('gp_log_amp')

    monkeypatch.setattr(fit.model, 'compute_gp_edf', _raise_stale_hyperparam_lookup)

    calls = []
    monkeypatch.setattr(fit.TransitFit, 'save_posterior_samples',
                        lambda self: calls.append('posterior'))
    monkeypatch.setattr(fit.TransitFit, 'save_corrected',
                        lambda self: calls.append('corrected'))

    with caplog.at_level(logging.WARNING):
        tf.save_results()

    rows = _ic_rows(tmp_path)
    assert set(rows) == {'BIC', 'AIC', 'AICc'}, (
        'the uncorrected rows must survive, and no corrected row may be written'
    )
    assert calls == ['posterior', 'corrected']
    assert 'KeyError' in caplog.text
