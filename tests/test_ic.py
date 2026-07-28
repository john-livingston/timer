import numpy as np
import pytest

import arviz as az
import pymc as pm

from timer import util
from timer.fit import TransitFit


@pytest.fixture
def gp_shaped_model():
    """A PyMC model with the shape of a GP fit: a handful of free parameters
    and three long deterministic sites.

    13 free parameters against 620 deterministic elements, so counting
    deterministics is off by a factor of 50 rather than by a rounding error.
    """
    n = 60
    with pm.Model() as model:
        pm.Normal('t0', 0, 1)
        pm.Normal('period', 3, 1)
        pm.Normal('ror', 0.05, 0.01)
        pm.Normal('b', 0.3, 0.1)
        pm.Normal('dur', 0.1, 0.01)
        pm.Normal('u_star_g', 0.3, 0.1, shape=2)
        pm.Normal('gp_log_amp', 0, 1)
        pm.Normal('gp_log_scale', 0, 1)
        pm.Normal('g_log_sigma_lc', -5, 1)
        w = pm.Normal('g_weights', 0, 1, shape=3)
        lm = pm.Deterministic('g_lm', w.sum() * np.ones(n))
        pm.Deterministic('g_light_curves', w.sum() * np.ones(n))
        pm.Deterministic('g_light_curves_hr', w.sum() * np.ones(500))
        pm.Normal('g_y_observed', mu=lm, sigma=1, observed=np.zeros(n))
    return model


def test_count_free_params_excludes_deterministics_and_observed(gp_shaped_model):
    """Deterministic sites are functions of the free parameters, not extra
    freedom. Counting the GP conditional mean and the light curves as
    parameters inflates nparams by the number of data points and reverses
    every model comparison in ic.txt.
    """
    assert util.count_free_params(gp_shaped_model) == 13


def test_count_free_params_can_count_the_gp_hyperparameters_alone(gp_shaped_model):
    """The edf correction replaces the GP's hyperparameters with the degrees
    of freedom it actually absorbs, so it has to subtract exactly what the
    full count added: 2 of the 13, not all 13 and not 0."""
    assert util.count_free_params(gp_shaped_model, prefix='gp_log_') == 2


def test_gp_hyperparameter_count_excludes_a_dataset_named_gp():
    """A dataset's jitter site is {name}_log_sigma_lc, so a dataset named 'gp'
    produces 'gp_log_sigma_lc'. Matching on the prefix 'gp_log_' swallows it,
    the edf correction then subtracts one parameter too many, and BIC_edf comes
    out low by log(ndata).
    """
    with pm.Model() as model:
        pm.Normal('gp_log_amp', 0, 1)
        pm.Normal('gp_log_scale', 0, 1)
        pm.Normal('gp_log_sigma_lc', -5, 1)     # the dataset's jitter
    assert util.count_free_params(model, prefix=util.GP_HYPERPARAM_PREFIXES) == 2


def test_gp_hyperparameter_count_covers_per_dataset_sites():
    """With gp.per_dataset the sites gain a dataset suffix, and those are still
    GP hyperparameters."""
    with pm.Model() as model:
        pm.Normal('gp_log_amp_g', 0, 1)
        pm.Normal('gp_log_amp_r', 0, 1)
        pm.Normal('gp_log_scale', 0, 1)
        pm.Normal('g_log_sigma_lc', -5, 1)
    assert util.count_free_params(model, prefix=util.GP_HYPERPARAM_PREFIXES) == 3


def test_compute_ic_rejects_an_unknown_method():
    """An unrecognised method falls through every branch and raises
    UnboundLocalError from the return, which names nothing useful."""
    with pytest.raises(ValueError, match='WAIC'):
        util.compute_ic(None, -100.0, 3, 100, method='WAIC', verbose=False)


def test_count_free_params_counts_every_element_of_a_vector_parameter():
    """A shape (3,) weight vector is three parameters, not one."""
    with pm.Model() as model:
        pm.Normal('scalar', 0, 1)
        pm.Normal('vector', 0, 1, shape=3)
    assert util.count_free_params(model) == 4


def _bare_fit(data, masks):
    """A TransitFit with only the attributes _count_data reads.

    The constructor loads YAML, reads data files and builds priors, none of
    which this counting depends on.
    """
    fit = TransitFit.__new__(TransitFit)
    fit.data = data
    fit.masks = masks
    return fit


def test_count_data_excludes_clipped_points():
    """Clipped outliers never enter the likelihood, so counting them inflates
    the sample size in BIC and shrinks the AICc correction."""
    fit = _bare_fit(
        data={'g': {'x': np.arange(10.)}, 'r': {'x': np.arange(6.)}},
        masks={'g': np.array([True] * 8 + [False] * 2), 'r': None},
    )
    assert fit._count_data() == 14


def test_count_data_counts_everything_when_nothing_is_clipped():
    fit = _bare_fit(
        data={'g': {'x': np.arange(10.)}, 'r': {'x': np.arange(6.)}},
        masks={'g': None, 'r': None},
    )
    assert fit._count_data() == 16


def _one_parameter_fit(prior_sigma, obs=np.zeros(4)):
    """A model with one free parameter and a Normal likelihood, plus a trace.

    The draws are mu = 0.5 then mu = 0, so the best one is deliberately not
    the first: taking draw zero rather than the maximum has to be visible.
    """
    with pm.Model() as model:
        mu = pm.Normal('mu', 0.0, prior_sigma)
        pm.Normal('obs', mu=mu, sigma=1.0, observed=obs)
    draws = np.array([[0.5, 0.0]])
    # real lp values, so that a criterion built on them genuinely moves with
    # prior_sigma and the invariance test below can fail
    logp = model.compile_logp()
    trace = az.from_dict(
        posterior={'mu': draws},
        sample_stats={'lp': np.array([[float(logp({'mu': v})) for v in draws[0]]])})
    fit = _bare_fit(data={'g': {'x': np.arange(4.)}}, masks={'g': None})
    fit.model = model
    fit.trace = trace
    return fit


def test_bic_is_built_from_the_likelihood_not_the_log_posterior():
    """BIC is defined with the maximized likelihood. PyMC's sample_stats['lp']
    is the joint density in the unconstrained space, so it also carries every
    prior term and the transform Jacobian.

    Hand derived: 4 observations at 0 with sigma 1 and mu 0 give a log
    likelihood of 4 * -0.5*ln(2*pi) = -3.67575413, so
    BIC = -2*(-3.67575413) + 1*ln(4) = 7.35150827 + 1.38629436 = 8.73780262.
    Using lp instead would add the mu prior, about +15.7 for sigma=1e3.
    """
    fit = _one_parameter_fit(prior_sigma=1e3)
    assert fit.get_ic(ic='BIC') == pytest.approx(8.73780262, abs=1e-6)


def test_the_information_criteria_do_not_move_with_the_prior_width():
    """The property the fix exists for. The weight prior in model.build is
    sigma=1e3, chosen to be uninformative; if it reaches the criteria then
    widening it to 1e4 makes every model look worse by 4.6 per parameter, and
    a detrending comparison is decided by a number nobody meant to set.
    """
    narrow = _one_parameter_fit(prior_sigma=1e3).get_ic(ic='BIC')
    wide = _one_parameter_fit(prior_sigma=1e4).get_ic(ic='BIC')
    assert narrow == pytest.approx(wide, abs=1e-9)


def test_the_likelihood_is_maximized_over_the_draws():
    """The best draw is the second one. Hand derived: at mu = 0.5 the log
    likelihood is -3.67575413 - 4*0.125 = -4.17575413, so taking the first
    draw, or any fixed one, would give BIC 9.73780262 instead of 8.73780262.
    """
    fit = _one_parameter_fit(prior_sigma=1e3)
    assert fit.get_ic(ic='BIC') == pytest.approx(8.73780262, abs=1e-6)


def test_get_ic_uses_the_masked_point_count_in_the_penalty(gp_shaped_model):
    """The penalty is k*ln(n) over the points that entered the likelihood."""
    fit = _one_parameter_fit(prior_sigma=1e3)
    fit.masks = {'g': np.array([True, True, True, False])}
    # nparams 1, ndata 3: -2*(-3.67575413) + ln(3)
    assert fit.get_ic(ic='BIC') == pytest.approx(7.35150827 + np.log(3), abs=1e-6)
