"""End to end coverage for the configuration branches no fixture used to reach.

`clip`, `include_flare`, `include_bump` and `chromatic` each take a different
path through model.build and through every downstream consumer. Nothing
exercised them, so a mutation anywhere in those branches failed no test. These
run real fits, which is the only way to reach them.
"""
import os

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.slow


def _run(wd, fit_params, sys_params):
    from timer import fit
    tf = fit.TransitFit(sys_params, fit_params, wd=str(wd))
    tf.build_model(verbose=False, plot=False)
    tf.clip_outliers()
    tf.sample(plot_fit=False, plot_systematics=False)
    tf.save_results()
    return tf


def _cor(wd, name):
    prefix = os.path.basename(str(wd))
    return pd.read_csv(os.path.join(wd, 'out', f'{prefix}-{name}-cor.csv'))


# ------------------------------------------------------------------ clip

@pytest.fixture(scope='module')
def clip_run(tmp_path_factory, make_project_module):
    wd = tmp_path_factory.mktemp('branch') / 'clip'
    fit_params, sys_params = make_project_module(str(wd), clip=True)
    return wd, _run(wd, fit_params, sys_params)


def test_clipping_removes_exactly_the_injected_outlier(clip_run, outlier_index):
    """One 12 sigma point is planted at a known index. The clip has to take
    that point and only that point: taking nothing means the threshold is too
    loose, taking more means it is eating real data."""
    _, tf = clip_run
    mask = tf.masks['g']

    assert mask is not None, 'clip: true must produce a mask'
    assert not mask[outlier_index], 'the planted outlier must be clipped'
    assert mask.sum() == mask.size - 1, 'nothing else may be clipped'


def test_a_clipped_point_leaves_the_likelihood_and_the_criteria(clip_run):
    """ndata drives the BIC penalty, so a clipped point has to leave the count
    as well as the mask."""
    _, tf = clip_run

    assert tf._count_data() == len(tf.data['g']['x']) - 1


def test_a_clipped_point_is_absent_from_the_corrected_light_curve(clip_run,
                                                                 outlier_index):
    """*-cor.csv is written from the masked arrays, so the clipped point must
    not appear in the file at all."""
    wd, tf = clip_run
    cor = _cor(wd, 'g')
    dropped_x = tf.data['g']['x'][outlier_index] + tf.ref_time

    assert len(cor) == len(tf.data['g']['x']) - 1
    # atol in days: points are ~0.002 d apart, and np.isclose's default
    # rtol against a BJD of 2460423 is a tolerance of about 24 days
    assert not np.isclose(cor.x.values, dropped_x, rtol=0, atol=1e-4).any()


def test_clipping_refits_the_model_on_the_masked_data(clip_run):
    """clip_outliers refits when the mask changes, so the model the trace came
    from must have been built on the masked points, not the raw ones."""
    _, tf = clip_run
    n_kept = int(tf.masks['g'].sum())

    observed = tf.model.observed_RVs[0]
    assert int(observed.shape.eval()[0]) == n_kept


# ------------------------------------------------------------------ flare

@pytest.fixture(scope='module')
def flare_run(tmp_path_factory, make_project_module):
    wd = tmp_path_factory.mktemp('branch') / 'flare'
    fit_params, sys_params = make_project_module(str(wd), flare=True)
    return wd, _run(wd, fit_params, sys_params), fit_params


def test_the_flare_branch_creates_its_own_parameters(flare_run):
    """include_flare adds three sampled sites. Without them the flare model is
    not in the graph at all and everything below is vacuous."""
    from timer import util

    _, tf, _ = flare_run
    names = {rv.name for rv in tf.model.free_RVs}

    assert {'flare_tpeak', 'flare_fwhm', 'flare_ampl'} <= names
    assert util.count_free_params(tf.model) > 7, 'the flare adds free parameters'


def test_the_flare_component_is_localized_in_time(flare_run):
    """The flare model is a fast rise and exponential decay, so its component
    has to be large near the planted peak and negligible long before it. A
    constant, or a component evaluated on the wrong time array, fails both.
    """
    _, tf, fit_params = flare_run
    x = tf.data['g']['x'] + tf.ref_time
    flare = np.asarray(tf.map_soln['g_flare']).squeeze()
    tpeak = fit_params['flare']['tpeak']

    near = np.abs(x - tpeak) < 0.006
    well_before = x < tpeak - 0.03

    assert flare[near].max() > 1.0, 'the flare must be present at its peak'
    assert np.abs(flare[well_before]).max() < 0.5, 'and absent long before it'


def test_the_flare_is_removed_from_the_corrected_light_curve(flare_run):
    """A flare left in *-cor.csv would be read as astrophysical signal. The
    corrected flux at the flare peak must sit at the out of transit baseline.
    """
    wd, tf, fit_params = flare_run
    cor = _cor(wd, 'g')
    tpeak = fit_params['flare']['tpeak']

    # aflare1 decays slowly, so the comparison window sits well before the
    # peak where the flare model is identically zero, not after it
    at_peak = np.abs(cor.x.values - tpeak) < 0.006
    baseline = cor.x.values < tpeak - 0.03

    assert at_peak.sum() and baseline.sum(), 'fixture must span both regions'
    # the fixture samples five draws, so the recovered flare parameters are
    # rough and a residual of order 1e-3 survives. Measured over repeated
    # runs: 1.44e-3, 1.51e-3, 1.76e-3. Leaving the flare in shifts this by
    # the injected 6 ppt, so 3e-3 separates the two comfortably.
    assert cor.y.values[at_peak].mean() == pytest.approx(
        cor.y.values[baseline].mean(), abs=3e-3)


# ------------------------------------------------------------------- bump

@pytest.fixture(scope='module')
def bump_run(tmp_path_factory, make_project_module):
    wd = tmp_path_factory.mktemp('branch') / 'bump'
    fit_params, sys_params = make_project_module(str(wd), bump=True)
    return wd, _run(wd, fit_params, sys_params), fit_params


def test_the_bump_branch_creates_its_own_parameters(bump_run):
    _, tf, _ = bump_run
    names = {rv.name for rv in tf.model.free_RVs}

    assert {'bump_tcenter', 'bump_width', 'bump_ampl'} <= names


def test_the_bump_component_is_localized_in_time(bump_run):
    """A Gaussian in time, so it must peak at its center and vanish a few
    widths away."""
    _, tf, fit_params = bump_run
    x = tf.data['g']['x'] + tf.ref_time
    bump = np.asarray(tf.map_soln['g_bump']).squeeze()
    tcenter = fit_params['bump']['tcenter']

    near = np.abs(x - tcenter) < 0.004
    far = np.abs(x - tcenter) > 0.03

    assert bump[near].max() > 0.4, 'the bump must be present at its center'
    assert np.abs(bump[far]).max() < 0.2, 'and gone well away from it'


def test_the_bump_is_removed_from_the_corrected_light_curve(bump_run):
    """The bump sits inside the transit, so leaving it in *-cor.csv distorts
    the transit shape and therefore the fitted depth."""
    wd, tf, fit_params = bump_run
    cor = _cor(wd, 'g')
    tcenter = fit_params['bump']['tcenter']

    # both windows are inside the transit, so the transit depth cancels and
    # only the bump distinguishes them
    at_bump = np.abs(cor.x.values - tcenter) < 0.004
    in_transit_away = (np.abs(cor.x.values - 2460423.06) < 0.018) & \
                      (np.abs(cor.x.values - tcenter) > 0.008)

    assert at_bump.sum() and in_transit_away.sum()
    # measured over repeated runs: 5.3e-4, 8.3e-4, 1.05e-3. Leaving the
    # bump in shifts this by the injected 5 ppt.
    assert cor.y.values[at_bump].mean() == pytest.approx(
        cor.y.values[in_transit_away].mean(), abs=2.5e-3)


# -------------------------------------------------------------- chromatic

@pytest.fixture(scope='module')
def chromatic_run(tmp_path_factory, make_project_module):
    wd = tmp_path_factory.mktemp('branch') / 'chromatic'
    fit_params, sys_params = make_project_module(
        str(wd), bands=('g', 'r'), fit_u_star=True)
    return wd, _run(wd, fit_params, sys_params)


def test_chromatic_fits_one_radius_ratio_per_band(chromatic_run):
    """The whole point of chromatic: ror becomes ror_<band>, and the shared
    ror site must not exist alongside them."""
    _, tf = chromatic_run
    names = {rv.name for rv in tf.model.free_RVs}

    assert {'ror_g', 'ror_r'} <= names
    assert 'ror' not in names


def test_chromatic_limb_darkening_is_fitted_per_band(chromatic_run):
    """Each band gets its own limb darkening pair, since that is the other
    thing that varies with wavelength. Sharing one site across bands, or
    creating it once for the first band, both fail here.

    The values are only loosely bounded: ld.claret marginalizes over the
    stellar parameter uncertainties, so the priors these are drawn from differ
    from call to call.
    """
    _, tf = chromatic_run
    names = {rv.name for rv in tf.model.free_RVs}

    assert {'u_star_g', 'u_star_r'} <= names
    for band in ('g', 'r'):
        u = np.asarray(tf.map_soln[f'u_star_{band}']).squeeze()
        assert u.shape == (2,), 'a quadratic law has two coefficients'
        assert (u > -1.0).all() and (u < 1.5).all(), 'physically plausible'
        assert u.sum() < 1.5, 'the limb cannot be brighter than the disk center'



def test_chromatic_reports_both_bands_in_the_summary(chromatic_run):
    """get_var_names has a chromatic branch of its own; if it emitted the plain
    ror name, az.summary would raise after sampling had finished."""
    _, tf = chromatic_run
    index = set(tf.summary.index)

    assert any(n.startswith('ror_g') for n in index)
    assert any(n.startswith('ror_r') for n in index)


def test_chromatic_writes_a_corrected_light_curve_per_dataset(chromatic_run):
    wd, tf = chromatic_run

    for band in ('g', 'r'):
        cor = _cor(wd, band)
        assert len(cor) == len(tf.data[band]['x'])


# --------------------------------------------- uniform limb darkening bounds

# deliberately narrower than the [0, 1] model.build used to hardcode, so a
# point outside them is still inside the old range
U_STAR_BOUNDS = (0.3, 0.7)


def _u_star_model(tmp_path_factory, make_project, name, bounds):
    """A built model whose limb darkening is sampled under `bounds`."""
    import pymc as pm  # noqa: F401  (imported here to keep the module import light)
    from timer import fit

    wd = tmp_path_factory.mktemp('branch') / name
    fit_params, sys_params = make_project(
        str(wd), uniform={'u_star': list(bounds)}, fit_u_star=True)
    tf = fit.TransitFit(sys_params, fit_params, wd=str(wd))
    tf.build_model(verbose=False, plot=False)
    return tf.model


@pytest.fixture(scope='module')
def u_star_model(tmp_path_factory, make_project_module):
    return _u_star_model(tmp_path_factory, make_project_module, 'ustar',
                         U_STAR_BOUNDS)


@pytest.fixture(scope='module')
def u_star_model_full_range(tmp_path_factory, make_project_module):
    return _u_star_model(tmp_path_factory, make_project_module, 'ustar_full',
                         (0.0, 1.0))


def _logp_at(rv, value):
    """The model's own verdict on a value, elementwise: -inf means outside the
    support. Reading the bounds off the site would depend on how PyMC orders
    an Op's inputs; the support is the behavior that matters."""
    import numpy as np
    import pymc as pm

    return np.asarray(pm.logp(rv, np.full(2, value)).eval())


def test_configured_u_star_bounds_reach_the_model(u_star_model):
    """The bug: model.build hardcoded [0, 1] under a comment claiming the
    bounds came from fit.yaml, so `uniform: u_star: [0.3, 0.7]` sampled over
    [0, 1] with no error and no warning.

    get_priors already encodes the configured range as a center and a width per
    band, the same convention every other uniform parameter uses. Only the
    decoding was missing.
    """
    import numpy as np

    lower, upper = U_STAR_BOUNDS
    rv = u_star_model['u_star_g']

    assert np.isfinite(_logp_at(rv, (lower + upper) / 2)).all()
    # both of these sit inside the old hardcoded [0, 1], which is what makes
    # this discriminating rather than a restatement of pm.Uniform
    assert np.isneginf(_logp_at(rv, lower - 0.05)).all()
    assert np.isneginf(_logp_at(rv, upper + 0.05)).all()


def test_u_star_keeps_both_limb_darkening_coefficients(u_star_model):
    """The uniform branch stores one scalar center and width per band, not one
    per coefficient. Passing those through without a shape would collapse the
    site to a scalar, and the light curve code indexes u_star[0] and u_star[1].
    """
    assert tuple(u_star_model['u_star_g'].shape.eval()) == (2,)


def test_the_full_u_star_range_still_behaves_as_before(u_star_model_full_range):
    """The control: [0, 1] is what every config effectively got before the fix,
    so it has to come out unchanged rather than merely differently wrong."""
    import numpy as np

    rv = u_star_model_full_range['u_star_g']

    assert np.isfinite(_logp_at(rv, 0.1)).all()
    assert np.isfinite(_logp_at(rv, 0.9)).all()
    assert np.isneginf(_logp_at(rv, -0.1)).all()
    assert np.isneginf(_logp_at(rv, 1.1)).all()
