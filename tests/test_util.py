import arviz as az
import numpy as np
import pytest

from timer import util


# --------------------------------------------------------------- map soln

def _trace(lp, n=6):
    """An InferenceData with two chains, three draws and per-draw values that
    identify which sample was selected.

    Every posterior variable is filled with its flat sample index, so the
    returned MAP solution says unambiguously which draw it came from.
    """
    chains, draws = lp.shape
    idx = np.arange(chains * draws, dtype=float).reshape(chains, draws)
    posterior = {
        't0': idx[..., None],
        'g_weights': idx[..., None] * 10,
        'g_mean': idx,
        'g_lm': np.broadcast_to(idx[..., None], (chains, draws, n)).copy(),
        'g_light_curves': np.broadcast_to(
            idx[..., None, None], (chains, draws, n, 1)).copy(),
    }
    return az.from_dict(posterior=posterior, sample_stats={'lp': lp})


def test_get_map_soln_picks_one_sample_when_log_probabilities_tie():
    """Short chains repeat samples, so more than one draw can hold the maximum.

    Selecting by equality then keeps every tie, and each model component comes
    back with a trailing sample axis: the systematics model becomes (n, 2) and
    no longer broadcasts against the data.
    """
    lp = np.array([[-101.0, -100.0, -102.0], [-100.0, -103.0, -104.0]])
    soln, _ = util.get_map_soln(_trace(lp))
    assert soln['g_lm'].shape == (6,)
    # nanargmax takes the first maximum in flat order, which is chain 0 draw 1
    assert soln['g_lm'] == pytest.approx(np.full(6, 1.0))
    assert soln['g_mean'] == pytest.approx(1.0)


def test_get_map_soln_preserves_free_parameter_shapes():
    """The MAP is fed back to pm.sample as initvals, which type checks against
    the model's declared shapes. Collapsing a shape (1,) site to a Python float
    makes every resumed run fail with a pytensor conversion error."""
    lp = np.array([[-101.0, -100.0, -102.0], [-103.0, -104.0, -105.0]])
    soln, _ = util.get_map_soln(_trace(lp))
    assert soln['t0'].shape == (1,)
    assert soln['g_weights'].shape == (1,)
    assert np.shape(soln['g_mean']) == ()


def test_get_map_soln_squeezes_derived_quantities():
    """Derived arrays are consumed as flat vectors, so a single planet's light
    curves must come back (n,) and not (n, 1)."""
    lp = np.array([[-101.0, -100.0, -102.0], [-103.0, -104.0, -105.0]])
    soln, _ = util.get_map_soln(_trace(lp))
    assert soln['g_light_curves'].shape == (6,)


def test_get_map_soln_returns_the_best_log_probability():
    lp = np.array([[-101.0, -100.0, -102.0], [-103.0, -104.0, -105.0]])
    _, max_logp = util.get_map_soln(_trace(lp))
    assert max_logp == pytest.approx(-100.0)


def test_get_map_soln_ignores_nan_log_probabilities():
    """A divergent draw can record lp as nan, and a plain max then returns nan
    and selects that draw as the best sample."""
    lp = np.array([[-101.0, np.nan, -102.0], [-100.0, -103.0, -104.0]])
    soln, max_logp = util.get_map_soln(_trace(lp))
    assert max_logp == pytest.approx(-100.0)
    assert soln['g_mean'] == pytest.approx(3.0)


# ---------------------------------------------------------------- sys model

def _soln_with_every_component(n):
    """A MAP solution carrying every non-transit component, each with a
    distinct constant so a dropped term changes the answer by a known amount."""
    return {
        't0': np.array(0.5),
        'g_mean': np.array(0.5),
        'g_lm': np.full(n, 0.25),
        'g_flare': np.full(n, 0.125),
        'g_bump': np.full(n, 0.0625),
        'g_gp_pred': np.arange(n) * 0.1,
        'g_light_curves': np.zeros(n),
        'g_light_curves_hr': np.zeros(500),
    }


def test_get_corrected_subtracts_gp_flare_and_bump():
    """The corrected light curve is the data minus every non-transit component.

    Summing only the linear model and the mean leaves the GP trend, the flare
    and the bump in every *-cor.csv, which is the file used for downstream
    analysis.
    """
    n = 8
    data = dict(x=np.arange(n, dtype=float), y=np.arange(n, dtype=float),
                yerr=np.full(n, 0.01), x_hr=np.linspace(0, 7, 500))
    soln = _soln_with_every_component(n)

    cor = util.get_corrected(data, 'g', soln, 1, subtract_tc=False)

    # hand derived: mean 0.5 + lm 0.25 + flare 0.125 + bump 0.0625 = 0.9375,
    # plus the GP's 0.1*i, subtracted from y = i
    expected = np.arange(n) * 0.9 - 0.9375
    assert cor['y'] == pytest.approx(expected)
    assert cor['x'] == pytest.approx(np.arange(n, dtype=float))


def test_get_corrected_applies_the_mask_to_the_data_only():
    """Model components come out of the MAP solution already masked, so they
    are the length of the surviving points, not of the raw series."""
    n = 8
    mask = np.array([True, True, False, True, True, False, True, True])
    k = int(mask.sum())
    data = dict(x=np.arange(n, dtype=float), y=np.arange(n, dtype=float),
                yerr=np.full(n, 0.01), x_hr=np.linspace(0, 7, 500))
    soln = _soln_with_every_component(k)

    cor = util.get_corrected(data, 'g', soln, 1, mask=mask, subtract_tc=False)

    kept = np.array([0., 1., 3., 4., 6., 7.])
    assert cor['y'] == pytest.approx(kept - 0.9375 - np.arange(k) * 0.1)
    assert cor['x'] == pytest.approx(kept)


def test_get_corrected_subtracts_the_transit_time():
    """subtract_tc recenters the output on t0, which is what the plots expect."""
    n = 8
    data = dict(x=np.arange(n, dtype=float), y=np.arange(n, dtype=float),
                yerr=np.full(n, 0.01), x_hr=np.linspace(0, 7, 500))
    soln = _soln_with_every_component(n)

    cor = util.get_corrected(data, 'g', soln, 1, subtract_tc=True)

    assert cor['x'] == pytest.approx(np.arange(n) - 0.5)
    assert cor['x_hr'] == pytest.approx(np.linspace(0, 7, 500) - 0.5)


def test_get_residuals_subtracts_the_gp_without_being_told_about_it():
    """A GP fit's residuals must not still contain the GP trend. The component
    is present in the MAP solution, so no use_gp flag is needed to find it."""
    n = 8
    y = np.arange(n, dtype=float)
    soln = _soln_with_every_component(n)
    soln['g_light_curves'] = np.full(n, -2.0)

    resid = util.get_residuals('g', y, soln)

    expected = y + 2.0 - 0.9375 - np.arange(n) * 0.1
    assert resid == pytest.approx(expected)


def test_get_residuals_without_a_mean_treats_it_as_zero():
    """{name}_mean is only a model site when include_mean=True, so reading it
    unguarded makes every include_mean=False fit fail at the residual report."""
    n = 8
    y = np.arange(n, dtype=float)
    soln = _soln_with_every_component(n)
    del soln['g_mean']

    resid = util.get_residuals('g', y, soln)

    expected = y - 0.4375 - np.arange(n) * 0.1
    assert resid == pytest.approx(expected)


def test_get_residuals_sums_over_the_planet_axis():
    """Two planets keep the light curves 2-D; picking one column instead of
    summing would halve the depth removed."""
    n = 8
    y = np.zeros(n)
    soln = {
        'g_mean': np.array(0.0),
        'g_light_curves': np.column_stack([np.full(n, -1.0), np.full(n, -0.25)]),
    }

    resid = util.get_residuals('g', y, soln)

    assert resid == pytest.approx(np.full(n, 1.25))


# ------------------------------------------------------------- claret bands

@pytest.mark.parametrize('band,expected', [
    ('g', 'g*'),
    ('r', 'r*'),
    ('i', 'i*'),
    ('z', 'z*'),
    # `band in 'griz'` is a substring test, so these all wrongly gain an
    # asterisk and claret is then asked for a band it does not have
    ('gr', 'gr'),
    ('ri', 'ri'),
    ('iz', 'iz'),
    ('griz', 'griz'),
    ('', ''),
    # bands that were never Sloan
    ('B', 'B'),
    ('J', 'J'),
])
def test_claret_band_maps_only_exact_sloan_filters(band, expected):
    assert util.claret_band(band) == expected


# --------------------------------------------------- information criteria

def test_bic_uses_the_number_of_data_points_in_the_penalty():
    """Hand derived: -2*(-100) + 3*ln(100) = 200 + 13.8155105579643."""
    assert util.compute_ic(None, -100.0, 3, 100, method='BIC', verbose=False) == \
        pytest.approx(213.8155105579643)


def test_aic_penalty_is_twice_the_parameter_count():
    """Hand derived: 2*3 - 2*(-100) = 206."""
    assert util.compute_ic(None, -100.0, 3, 100, method='AIC', verbose=False) == \
        pytest.approx(206.0)


def test_aicc_adds_the_small_sample_correction():
    """Hand derived: AIC 206 + 2*(9+3)/(100-3-1) = 206 + 24/96 = 206.25."""
    assert util.compute_ic(None, -100.0, 3, 100, method='AICc', verbose=False) == \
        pytest.approx(206.25)


@pytest.mark.parametrize('ndata,nparams', [(4, 3), (3, 3), (10, 20)])
def test_aicc_is_nan_when_the_correction_denominator_is_not_positive(ndata, nparams):
    """ndata - nparams - 1 <= 0 makes the correction undefined. Left unguarded
    it silently flips sign, which is how a real run recorded AICc -17673.68 and
    looked like the best model in the comparison.
    """
    val = util.compute_ic(None, -100.0, nparams, ndata, method='AICc', verbose=False)
    assert np.isnan(val)


def test_aicc_is_finite_just_above_the_denominator_boundary():
    """The control at ndata - nparams - 1 = 1, so the guard cannot be satisfied
    by returning nan for every small sample. Hand derived:
    2*3 - 2*(-100) + 2*12/1 = 206 + 24 = 230."""
    assert util.compute_ic(None, -100.0, 3, 5, method='AICc', verbose=False) == \
        pytest.approx(230.0)


# ---------------------------------------------------------------- var names

def _var_name_args(fixed):
    data = {'g': {}, 'r': {}}
    return dict(data=data, bands=['g', 'r'], fit_basis='duration',
                use_gp=False, fixed=fixed)


def test_get_var_names_drops_a_fixed_t0():
    """t0 is not a posterior variable when it is fixed, so leaving it in the
    list makes az.summary raise after sampling has already finished."""
    names = util.get_var_names(**_var_name_args(['t0']))
    assert 't0' not in names


def test_get_var_names_keeps_t0_when_it_is_sampled():
    """The control: filtering must not drop t0 from an ordinary fit."""
    names = util.get_var_names(**_var_name_args([]))
    assert 't0' in names


def test_get_var_names_filters_each_transit_parameter_independently():
    names = util.get_var_names(**_var_name_args(['period', 'ror']))
    assert 'period' not in names
    assert 'ror' not in names
    assert 't0' in names
    assert 'b' in names
    assert 'dur' in names


# -------------------------------------------------------------- outlier mask

def _clip_fixture():
    """21 points alternating +/-1 with a single 20-sigma spike at index 10.

    Residual squares are 1 for the 20 ordinary points and 400 for the spike,
    so the median is 1 and the rms is 1. At nsig=7 the spike is the only point
    outside the threshold.
    """
    n = 21
    x = np.arange(n, dtype=float)
    y = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    y[10] = 20.0
    return x, y, n


def test_get_outlier_mask_without_a_mean_treats_it_as_zero():
    """{name}_mean only exists when include_mean=True, so reading it unguarded
    makes clipping crash on exactly the configuration that needs it most."""
    x, y, n = _clip_fixture()
    map_soln = {'g_light_curves': np.zeros(n)}

    mask = util.get_outlier_mask(x, y, 'g', map_soln, use_gp=False)

    assert mask.sum() == 20
    assert not mask[10]


def test_get_outlier_mask_uses_the_mean_when_it_is_present():
    """With a mean of 20 the spike becomes the typical point and the ordinary
    points become the large residuals: rms is then 19 and nothing is clipped.
    A guard that ignored the mean instead of defaulting it would clip index 10.
    """
    x, y, n = _clip_fixture()
    map_soln = {'g_light_curves': np.zeros(n), 'g_mean': np.array(20.0)}

    mask = util.get_outlier_mask(x, y, 'g', map_soln, use_gp=False)

    assert mask.all()
