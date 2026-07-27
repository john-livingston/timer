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


@pytest.fixture
def tiny_trace():
    """A two chain, three draw trace whose best sample has lp = -100."""
    lp = np.array([[-101.0, -100.0, -102.0], [-103.0, -104.0, -105.0]])
    posterior = {'t0': np.full((2, 3), 0.5)}
    return az.from_dict(posterior=posterior, sample_stats={'lp': lp})


def test_get_ic_reports_bic_from_the_model_and_the_unmasked_points(
        gp_shaped_model, tiny_trace):
    """Exercises the whole ic.txt path: best log probability from the trace,
    free parameter count from the model, sample size from the masks.

    Hand derived with logp -100, nparams 13 and ndata 14:
    -2*(-100) + 13*ln(14) = 200 + 34.30607... = 234.30607.
    """
    fit = _bare_fit(
        data={'g': {'x': np.arange(10.)}, 'r': {'x': np.arange(6.)}},
        masks={'g': np.array([True] * 8 + [False] * 2), 'r': None},
    )
    fit.model = gp_shaped_model
    fit.trace = tiny_trace

    assert fit.get_ic(ic='BIC') == pytest.approx(200 + 13 * np.log(14))
