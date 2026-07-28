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
    tf.trace = az.from_dict(
        posterior={'t0': np.full((1, 4, 1), 0.5)},
        sample_stats={'lp': np.array([[-101.0, -100.0, -102.0, -103.0]])})

    monkeypatch.setattr(fit.util, 'get_map_soln', lambda trace: ({}, -100.0))
    monkeypatch.setattr(fit.TransitFit, '_max_loglike', lambda self: -100.0)
    monkeypatch.setattr(fit.TransitFit, '_count_params', lambda self: nparams)
    monkeypatch.setattr(fit.TransitFit, '_count_data', lambda self: ndata)
    monkeypatch.setattr(fit.TransitFit, '_count_gp_hyperparams', lambda self: 2)
    monkeypatch.setattr(fit.TransitFit, 'save_posterior_samples', lambda self: None)
    monkeypatch.setattr(fit.TransitFit, 'save_corrected', lambda self: None)
    return tf


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
        'the correction must penalise the GP, otherwise it is not measuring '
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
