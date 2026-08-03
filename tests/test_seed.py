"""Reproducibility of the sampler.

Without a seed every fit differs, so any assertion about a fitted value is a
coin flip. Two tests in test_config_branches.py were tuned to tolerances that
happened to pass and then failed on a later run; the tolerances now come from
measurement, but the underlying variability is what made them fragile.
"""
import numpy as np
import pymc as pm
import pytest

from timer import cache, model


def _tiny_model():
    with pm.Model() as m:
        pm.Normal('mu', 0.0, 1.0)
        pm.Normal('obs', mu=0.0, sigma=1.0, observed=np.zeros(5))
    return m


def _draws(seed):
    trace = model.sample(_tiny_model(), {}, tune=5, draws=5, chains=1, cores=1,
                         random_seed=seed)
    return np.asarray(trace.posterior['mu'].values)


@pytest.mark.slow
def test_the_same_seed_gives_the_same_draws():
    """The property the seed exists for."""
    assert _draws(1234) == pytest.approx(_draws(1234))


@pytest.mark.slow
def test_a_different_seed_gives_different_draws():
    """The control: a seed that pins everything to one fixed chain would pass
    the test above while making the seed meaningless."""
    assert not np.allclose(_draws(1234), _draws(5678))


@pytest.mark.slow
def test_no_seed_still_samples():
    """random_seed is optional, so leaving it out must not break anything."""
    trace = model.sample(_tiny_model(), {}, tune=5, draws=5, chains=1, cores=1)
    assert trace.posterior['mu'].values.shape == (1, 5)


def test_the_configured_seed_reaches_the_sampler(monkeypatch):
    """fit.yaml's random_seed has to travel from the config to pm.sample.
    Reading it but never passing it on leaves every run non-reproducible while
    looking configured."""
    from timer import fit

    captured = {}

    def fake_sample(*args, **kwargs):
        captured.update(kwargs)
        return 'trace'

    monkeypatch.setattr(fit.model, 'sample', fake_sample)

    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.outdir, tf.clobber = '.', False
    tf._stale_force_loaded, tf._cache_keys = set(), {'model': 'M', 'run': 'R'}
    tf.trace, tf.model, tf.map_soln = None, object(), {}
    tf.tune, tf.draws, tf.chains, tf.cores = 5, 5, 1, 1
    tf.random_seed = 4321

    with pytest.raises(Exception):
        # everything after model.sample needs a real trace; the call itself is
        # what this test is about
        tf.sample(plot_fit=False, plot_systematics=False)

    assert captured.get('random_seed') == 4321


def test_the_seed_defaults_to_none():
    """Unset means unseeded, so ordinary runs keep their present behavior."""
    from timer import fit

    assert fit.defaults['model']['random_seed'] is None


def test_the_sampler_defaults_are_exactly_the_run_tier_and_no_effect_keys():
    """Which block a default lives in is documentation, and this is what keeps
    it honest.

    random_seed sat under `sampler` while the cache classified it as run tier,
    and the two agreed with each other while both were wrong. Pinning the block
    to the tiers means the next key that reaches the model cannot be filed as a
    sampler setting without something failing here.
    """
    from timer import fit

    assert set(fit.defaults['sampler']) == cache.RUN_TIER | cache.NO_EFFECT


def _priors_with_seed(tmp_path, make_project, seed, name):
    from timer import fit

    root = tmp_path / name
    fit_params, sys_params = make_project(str(root))
    fit_params['random_seed'] = seed
    return fit.TransitFit(sys_params, fit_params, wd=str(root)).priors


def test_the_seed_makes_the_limb_darkening_priors_reproducible(tmp_path,
                                                               make_project_fn):
    """ld.claret marginalizes over the stellar parameter uncertainties using
    numpy's global RNG, so two runs of the same config otherwise get different
    limb darkening priors and therefore different fits.

    A random_seed that reached only the sampler would leave ic.txt varying run
    to run, which is not what the option promises.
    """
    a = _priors_with_seed(tmp_path, make_project_fn, 99, 'a')
    b = _priors_with_seed(tmp_path, make_project_fn, 99, 'b')

    assert a['u_star']['g'] == pytest.approx(b['u_star']['g'])
    assert a['u_star_unc']['g'] == pytest.approx(b['u_star_unc']['g'])


def test_a_different_seed_gives_different_limb_darkening_priors(tmp_path,
                                                                make_project_fn):
    """The control: pinning the priors to one fixed draw regardless of the seed
    would satisfy the test above while hiding the marginalization entirely."""
    a = _priors_with_seed(tmp_path, make_project_fn, 99, 'a')
    b = _priors_with_seed(tmp_path, make_project_fn, 100, 'b')

    assert not np.allclose(a['u_star']['g'], b['u_star']['g'])


def test_seeding_restores_the_callers_global_random_state(tmp_path,
                                                          make_project_fn):
    """Seeding numpy's global RNG is a process wide side effect. Building a fit
    must not silently reset the random stream the caller was using."""
    np.random.seed(7)
    before = np.random.random()
    np.random.seed(7)

    _priors_with_seed(tmp_path, make_project_fn, 99, 'a')

    assert np.random.random() == pytest.approx(before)

