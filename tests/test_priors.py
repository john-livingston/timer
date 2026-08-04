"""Coverage for get_priors' uniform-bounds arithmetic.

get_priors stores a uniform prior as a center and a width, and get_rv
reconstructs the bounds from them as `center -/+ width/2`. Nothing tested that
round trip, so any error in the halving, the centering or the per-planet
indexing would have shipped: the model would sample a parameter over bounds the
user never wrote, silently.

Every expectation here is the literal the user wrote in fit.yaml, so these are
not mirrors of the implementation.
"""
import numpy as np
import pytest

from timer import util


STAR = {'teff': [5675, 75], 'logg': [4.2, 0.2], 'feh': [0.0, 0.5],
        'radius': [1.0, 0.05]}


def _planet(period=14.3, dur=0.2, ror=0.06, b=0.15):
    return {'period': [period, 1e-4], 'dur': [dur, 0.005],
            'ror': [ror, 0.005], 'b': [b, 0.1]}


def _priors(uniform, planets=None, fixed=(), bands=('g',)):
    return util.get_priors(
        'duration', STAR, planets or [_planet()], list(fixed), list(bands),
        np.atleast_1d(0.5), np.atleast_1d(0.02), uniform=uniform)


def _bounds_the_model_will_use(priors, key):
    """The bounds get_rv reconstructs, which is what pm.Uniform receives."""
    return (priors[key] - priors[f'{key}_unc'] / 2,
            priors[key] + priors[f'{key}_unc'] / 2)


def test_uniform_bounds_survive_the_round_trip():
    """The user writes [0.01, 0.15]; the model must sample over [0.01, 0.15].

    get_priors stores the center and the width, so a missing factor of two, or
    a center computed as a difference, would move the bounds without any error.
    """
    priors = _priors({'ror': [0.01, 0.15]})

    lower, upper = _bounds_the_model_will_use(priors, 'ror')

    assert priors['ror_prior'] == 'uniform'
    assert lower == pytest.approx(0.01)
    assert upper == pytest.approx(0.15)


def test_uniform_bounds_are_not_symmetric_about_the_sys_yaml_value():
    """Asymmetric bounds are the discriminating case: the center has to come
    from the bounds, not from the sys.yaml mean. Here sys.yaml says b = 0.15
    while the bounds are [0.0, 1.0], whose center is 0.5."""
    priors = _priors({'b': [0.0, 1.0]}, planets=[_planet(b=0.15)])

    lower, upper = _bounds_the_model_will_use(priors, 'b')

    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(1.0)


def test_planet_indexed_uniform_bounds_stay_with_their_planet():
    """Two planets with different bounds. Transposing, averaging across planets
    or broadcasting the first pair would all keep the shapes valid."""
    priors = _priors(
        {'ror': [[0.03, 0.06], [0.10, 0.20]]},
        planets=[_planet(ror=0.04), _planet(ror=0.15)])

    lower, upper = _bounds_the_model_will_use(priors, 'ror')

    assert lower == pytest.approx([0.03, 0.10])
    assert upper == pytest.approx([0.06, 0.20])


def test_shared_uniform_bounds_are_broadcast_to_every_planet():
    priors = _priors({'ror': [0.01, 0.15]},
                     planets=[_planet(ror=0.04), _planet(ror=0.05)])

    lower, upper = _bounds_the_model_will_use(priors, 'ror')

    assert lower == pytest.approx([0.01, 0.01])
    assert upper == pytest.approx([0.15, 0.15])


def test_the_initial_value_is_the_sys_yaml_value_when_it_is_inside_the_bounds():
    """pm.Uniform rejects an initval outside its support, so the initval is the
    sys.yaml mean clipped into the bounds. Inside the bounds it must pass
    through untouched rather than being replaced by the center."""
    priors = _priors({'ror': [0.01, 0.15]}, planets=[_planet(ror=0.06)])

    assert priors['ror_initval'] == pytest.approx([0.06])


def test_the_initial_value_is_pulled_inside_bounds_that_exclude_it():
    """sys.yaml can disagree with the bounds. An unclipped initval makes
    pm.Uniform raise at model build time, so it is clipped, and strictly
    inside: exactly on the edge is still outside the open support.
    """
    priors = _priors({'ror': [0.01, 0.05]}, planets=[_planet(ror=0.06)])

    initval = priors['ror_initval']
    lower, upper = _bounds_the_model_will_use(priors, 'ror')
    assert (initval > lower).all() and (initval < upper).all()
    assert initval == pytest.approx([0.05], abs=1e-9)


def test_planet_indexed_initial_values_are_clipped_per_planet():
    """The clipping is per planet, so one planet's sys.yaml value can pass
    through while another's is pulled in. Using the bounds' center instead would
    be a valid initval for both and hide the difference.

    Planet 0's ror of 0.04 sits inside [0.03, 0.06]; planet 1's 0.30 is outside
    [0.10, 0.20] and must come back at the upper edge.
    """
    priors = _priors(
        {'ror': [[0.03, 0.06], [0.10, 0.20]]},
        planets=[_planet(ror=0.04), _planet(ror=0.30)])

    initval = priors['ror_initval']

    assert initval[0] == pytest.approx(0.04)
    assert initval[1] == pytest.approx(0.20, abs=1e-9)
    lower, upper = _bounds_the_model_will_use(priors, 'ror')
    assert (initval > lower).all() and (initval < upper).all()


def test_u_star_uniform_bounds_round_trip_per_band():
    """u_star is stored per band as a dict, so the bounds have to be rebuilt
    for every band rather than for the first one only."""
    priors = _priors({'u_star': [0.0, 1.0]}, bands=('g', 'r'))

    assert priors['u_star_prior'] == 'uniform'
    for band in ('g', 'r'):
        center, width = priors['u_star'][band], priors['u_star_unc'][band]
        assert center - width / 2 == pytest.approx(0.0)
        assert center + width / 2 == pytest.approx(1.0)


def test_u_star_uniform_keeps_the_claret_value_as_the_initial_value():
    """The claret coefficients are the physically motivated starting point, so
    a uniform prior must widen the support without discarding them.

    Asserted within one call. ld.claret marginalizes over the stellar parameter
    uncertainties and returns a different draw every time, so comparing two
    get_priors calls would compare two independent samples.
    """
    priors = _priors({'u_star': [0.0, 1.0]})

    initval = np.asarray(priors['u_star_initval']['g'])
    center = priors['u_star']['g']

    assert initval.shape == (2,), 'both limb darkening coefficients are kept'
    # the initval is the claret draw, not the midpoint of the bounds
    assert not np.allclose(initval, center)
    assert (initval > 0.0).all() and (initval < 1.0).all()


def test_u_star_uniform_clips_the_claret_initial_value_into_the_bounds():
    """A configured range narrower than the claret draw is the whole point of
    writing one, and the two need not overlap.

    pm.Uniform does not reject an initval outside its support: it writes nan
    into the initial point, and the optimizer starts the MAP from there. Every
    other uniform parameter clips, so u_star does too. The bounds here exclude
    the second claret coefficient, which sits near 0.18 for this star.
    """
    lower, upper = 0.45, 0.55
    priors = _priors({'u_star': [lower, upper]})

    initval = np.asarray(priors['u_star_initval']['g'])

    assert initval.shape == (2,)
    assert (initval > lower).all() and (initval < upper).all()


def test_a_parameter_not_listed_in_uniform_keeps_its_gaussian_prior():
    """The control: `uniform` must not turn every parameter uniform."""
    priors = _priors({'ror': [0.01, 0.15]})

    assert priors['ror_prior'] == 'uniform'
    assert priors['b_prior'] == 'gaussian'
    assert priors['dur_prior'] == 'gaussian'
    # the gaussian branch keeps the sys.yaml value and uncertainty
    assert priors['b'] == pytest.approx([0.15])
    assert priors['b_unc'] == pytest.approx([0.1])


def test_a_fixed_parameter_is_not_given_a_prior_at_all():
    """A fixed parameter is held at its sys.yaml value, so it gets no _prior
    key and the model never creates a site for it."""
    priors = _priors({}, fixed=('period',))

    assert 'period_prior' not in priors
    assert priors['period'] == pytest.approx([14.3])


def test_planet_indexed_bounds_must_match_the_planet_count():
    """Two bound pairs against one planet is a config error, and silently
    using the first pair would fit the wrong support."""
    with pytest.raises(ValueError, match='ror'):
        _priors({'ror': [[0.03, 0.06], [0.10, 0.20]]}, planets=[_planet()])
