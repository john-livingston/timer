import logging

import numpy as np
import pandas as pd
from astropy.time import Time
import limbdark as ld
import arviz as az
from patsy import dmatrix

from .plot import plot_outliers

def get_spline_basis(x, degree=3, knots=None, n_knots=5, include_intercept=False):
    if knots is not None:
        dm_formula = "bs(x, knots={}, degree={}, include_intercept={}) - 1" "".format(
            knots, degree, include_intercept
        )
        spline_dm = np.asarray(dmatrix(dm_formula, {"x": x}))
    else:
        dm_formula = "bs(x, df={}, degree={}, include_intercept={}) - 1" "".format(
            n_knots, degree, include_intercept
        )
        spline_dm = np.asarray(dmatrix(dm_formula, {"x": x}))
    return spline_dm

def get_sys_model(name, soln, npoints):
    """Sum of every non-transit model component present in soln.

    Components are the mean flux, the linear (systematics) model, the flare
    and bump models, and the GP conditional mean. A component absent from
    soln contributes zero, so this is safe for any model configuration.
    """
    sys_mod = np.zeros(npoints)
    for key in (f'{name}_mean', f'{name}_lm', f'{name}_flare',
                f'{name}_bump', f'{name}_gp_pred'):
        if key in soln:
            sys_mod = sys_mod + np.asarray(soln[key]).squeeze().flatten()
    return sys_mod

def get_residuals(name, y, soln, mask=None):

    if mask is None:
        mask = np.ones(len(y), dtype=bool)

    tra_mod = soln[f"{name}_light_curves"]
    # Sum over planets axis if multiple planets
    if tra_mod.ndim > 1:
        tra_mod = np.sum(tra_mod, axis=1)

    # Apply mask to tra_mod if it has the same length as y
    if len(tra_mod) == len(y):
        tra_mod = tra_mod[mask]

    sys_mod = get_sys_model(name, soln, int(mask.sum()))

    return y[mask] - tra_mod - sys_mod

# PyMC deterministic sites and the observed site, plus the post-hoc GP
# prediction added by model._add_gp_predictions. None of these are free
# parameters, and all are consumed as flat arrays.
DERIVED_SUFFIXES = ('_light_curves', '_light_curves_hr', '_lc_pred', '_lm',
                    '_flare', '_bump', '_y_observed', '_gp_pred')


def get_map_soln(trace):
    # arviz trace is an InferenceData object.
    # Index the single best sample rather than masking a copy of the whole
    # posterior: selecting by equality keeps every draw that ties on lp, and
    # short chains repeat samples often. nanargmax/nanmax also skip nan log
    # probabilities rather than selecting a nan sample as the best one.
    lp_values = np.asarray(trace.sample_stats["lp"].values)
    chain, draw = np.unravel_index(np.nanargmax(lp_values), lp_values.shape)
    trace_map = trace.posterior.isel(chain=chain, draw=draw)
    soln = {}
    for k, v in trace_map.data_vars.items():
        val = np.asarray(v.values)
        if k.endswith(DERIVED_SUFFIXES):
            # derived quantities are consumed as flat arrays, so drop the
            # planet axis a single planet leaves behind
            soln[k] = np.squeeze(val)
        else:
            # free parameters go back to pm.sample as initvals, which type
            # checks them against the model's declared shapes, so a shape (1,)
            # site must not come back as a Python float
            soln[k] = val
    return soln, float(np.nanmax(lp_values))

def get_var_names(data, bands, fit_basis, use_gp, fixed,
                  chromatic=False, log_sigma=True, weights=False, gp_config=None):

    var_names = []
    for par in 't0 period b dur'.split():
        if par not in fixed:
            var_names += [par]
    if 'ror' not in fixed:
        if chromatic:
            for band in bands:
                var_names += [f'ror_{band}']
        else:
            var_names += ['ror']
    if (fit_basis == 'mstar/rstar') and not any(['m_star' in fixed, 'r_star' in fixed]):
        var_names += ['m_star', 'r_star']
    if use_gp:
        per_ds = gp_config.get('per_dataset', []) if gp_config else []
        for p in ['log_amp', 'log_scale']:
            if p in per_ds:
                for name in data.keys():
                    var_names += [f'gp_{p}_{name}']
            else:
                var_names += [f'gp_{p}']
    for name in data.keys():
        if weights:
            var_names += [f'{name}_weights']
        if log_sigma:
            var_names += [f'{name}_log_sigma_lc']
    return var_names

def get_summary(trace, data, bands, fit_basis, use_gp, fixed,
                chromatic=False, log_sigma=True, weights=False, gp_config=None):

    var_names = get_var_names(data, bands, fit_basis, use_gp, fixed,
                              chromatic=chromatic, log_sigma=log_sigma, weights=weights,
                              gp_config=gp_config)
    summary = az.summary(
        trace,
        var_names=var_names
    )
    return summary

def get_outlier_mask(x, y, name, map_soln, use_gp, nsig=7, include_flare=False, include_bump=False, fp=None):
    lcs = map_soln[f"{name}_light_curves"]
    # the mean is only a model site when include_mean=True
    mean = map_soln[f"{name}_mean"] if f"{name}_mean" in map_soln else 0.0
    mod = (
        + mean
        + (np.sum(lcs, axis=-1) if lcs.ndim > 1 else lcs)
    )
    if f"{name}_lm" in map_soln.keys():
        mod += map_soln[f"{name}_lm"]
    if use_gp:
        mod += map_soln[f"{name}_gp_pred"]
    if include_flare:
        mod += map_soln[f'{name}_flare']
    if include_bump:
        mod += map_soln[f'{name}_bump']
    resid = y - mod
    rms = np.sqrt(np.median(resid**2))
    mask = np.abs(resid) < nsig * rms

    if fp is not None and mask.sum() < mask.size:
        plot_outliers(x, resid, mask, fp=fp)

    return mask

# Sloan filters, which claret's tables distinguish from the Stromgren filters
# of the same letter by a trailing asterisk
SLOAN_BANDS = frozenset(['g', 'r', 'i', 'z'])


def claret_band(band):
    """The claret table name for a filter name.

    Membership has to be exact. `band in 'griz'` is a substring test, so it
    also matches 'gr', 'ri', 'iz' and '', appending an asterisk and asking
    claret for a band it does not have.
    """
    return f'{band}*' if band in SLOAN_BANDS else band


def get_priors(fit_basis, star, planets, fixed, bands, tc_guess, tc_guess_unc, uniform={}):

    priors = {}
    if fit_basis == 'mstar/rstar':
        priors['r_star'] = np.array(star['radius'][0])
        priors['r_star_unc'] = np.array(star['radius'][1])
        priors['m_star'] = np.array(star['mass'][0])
        priors['m_star_unc'] = np.array(star['mass'][1])
    elif fit_basis == 'duration':
        if 'radius' in star:
            priors['r_star'] = np.array(star['radius'][0])
            priors['r_star_unc'] = np.array(star['radius'][1])
    elif fit_basis == 'density':
        raise NotImplementedError
    else:
        raise ValueError(f"fit_basis={fit_basis} not supported")

    bands_ = [claret_band(band) for band in bands]
    ldp = [ld.claret(band, *star['teff'], *star['logg'], *star['feh']) for band in bands_]
    priors['u_star'] = {band:ld[::2] for band,ld in zip(bands, ldp)}
    priors['u_star_unc'] = {band:ld[1::2] for band,ld in zip(bands, ldp)}
    if 'u_star' in uniform:
        priors['u_star_prior'] = 'uniform'
        bounds = np.array(uniform['u_star'])
        priors['u_star_unc'] = {band:bounds[1] - bounds[0] for band in bands}
        priors['u_star_initval'] = priors['u_star']
        priors['u_star'] = {band:(bounds[0] + bounds[1]) / 2 for band in bands}
    else:
        priors['u_star_prior'] = 'gaussian'

    for par in 'period dur ror b'.split():
        # Always store the original mean value from sys.yaml
        original_mean = np.array([i[par][0] for i in planets])
        priors[par] = original_mean

        if par not in fixed:
            if par in uniform:
                priors[f'{par}_prior'] = 'uniform'
                # For uniform priors, we need to calculate the width from the bounds
                # The model expects: lower = priors[key] - priors[f'{key}_unc']/2
                #                   upper = priors[key] + priors[f'{key}_unc']/2
                # So: priors[f'{key}_unc'] = upper - lower

                bounds_input = uniform[par]

                # Check if we have planet-indexed bounds: [[low1,high1], [low2,high2], ...]
                # or single bounds for all planets: [low, high]
                if isinstance(bounds_input[0], (list, tuple)):
                    # Planet-indexed bounds
                    if len(bounds_input) != len(planets):
                        raise ValueError(f"Number of {par} bounds ({len(bounds_input)}) must match number of planets ({len(planets)})")

                    bounds_array = np.array(bounds_input)
                    priors[f'{par}_unc'] = bounds_array[:, 1] - bounds_array[:, 0]
                    # Store the center point as the parameter value for bounds calculation
                    priors[par] = (bounds_array[:, 0] + bounds_array[:, 1]) / 2
                    # Clip initval to be within bounds (with small epsilon to avoid boundary issues)
                    epsilon = 1e-10
                    clipped = np.clip(original_mean, bounds_array[:, 0] + epsilon, bounds_array[:, 1] - epsilon)
                    priors[f'{par}_initval'] = clipped
                else:
                    # Single bounds for all planets (backward compatibility)
                    bounds = np.array(bounds_input)
                    priors[f'{par}_unc'] = bounds[1] - bounds[0]
                    # Store the center point as the parameter value for bounds calculation
                    priors[par] = np.array([(bounds[0] + bounds[1]) / 2] * len(planets))
                    # Clip initval to be within bounds (with small epsilon to avoid boundary issues)
                    epsilon = 1e-10
                    clipped = np.clip(original_mean, bounds[0] + epsilon, bounds[1] - epsilon)
                    priors[f'{par}_initval'] = clipped
            else:
                # assume gaussian
                priors[f'{par}_prior'] = 'gaussian'
                priors[f'{par}_unc'] = np.array([i[par][1] for i in planets])

    priors['t0'] = tc_guess
    priors['t0_unc'] = tc_guess_unc
    priors['t0_prior'] = 'uniform'

    return priors

def get_tc_prior(fit_params, x, ref_time):

    if 'tc_pred' in fit_params.keys():
        tc_guess = np.array(fit_params['tc_pred']) - ref_time
    elif 'tc_pred_iso' in fit_params.keys():
        tc_guess = Time(np.array(fit_params['tc_pred_iso'])).jd - ref_time
    else:
        tc_guess = x.mean()
    if 'tc_pred_unc' in fit_params.keys():
        tc_guess_unc = fit_params['tc_pred_unc']
    else:
        tc_guess_unc = 0.04

    return np.atleast_1d(tc_guess), np.atleast_1d(tc_guess_unc)

def bin_df(df, timecol='time', errcol='flux_err', binsize=60/86400., kind='median'):
    """
    df : DataFrame
    timecol : name of column with measurement times
    errcol : name of column with measurement errors
    binsize : size of bins (same units as time column)
    kind : median of points in each bin if set to 'median', else mean
    """
    bins = np.arange(df[timecol].min(), df[timecol].max(), binsize)
    groups = df.groupby(np.digitize(df[timecol], bins))
    if kind == 'median':
        df_binned = groups.median()
    else:
        df_binned = groups.mean()
    yerr_binned = groups.median()[errcol] / np.sqrt(groups.size())
    df_binned[errcol] = yerr_binned
    return df_binned.dropna()

def count_free_params(model, prefix=None):
    """Number of free parameters in a PyMC model.

    PyMC's free_RVs is exactly the right set: deterministic sites (the light
    curves, the linear model, the GP conditional mean) and the observed
    likelihood are functions of the free parameters, not extra freedom, and
    PyMC keeps them in separate collections. That is why this counts the
    model rather than the MAP solution dict, which mixes the two.

    Pass `prefix` to count only the parameters whose site name starts with it,
    which is how the GP's own hyperparameters are counted for the effective
    degrees of freedom correction.
    """
    return sum(int(rv.size.eval()) for rv in model.free_RVs
               if prefix is None or rv.name.startswith(prefix))


def compute_ic(map_soln, max_logp, nparams, ndata, method='BIC', verbose=True):

    if method == 'BIC':
        ic = -2 * max_logp + nparams * np.log(ndata)
    elif method == 'AIC':
        ic = 2 * nparams - 2 * max_logp
    elif method == 'AICc':
        ic = 2 * nparams - 2 * max_logp
        denom = ndata - nparams - 1
        if denom <= 0:
            logging.warning(
                f'AICc is undefined for nparams={nparams} and ndata={ndata}: '
                f'the correction denominator is {denom}; returning nan'
            )
            return float('nan')
        ic += 2 * (nparams**2 + nparams) / denom

    if verbose:
        print('Number of datapoints: {}'.format(ndata))
        print('Number of parameters: {}'.format(nparams))
        print('Max logp = {}'.format(max_logp))
        print('{} = {}'.format(method, ic))

    return float(ic)

def get_corrected(data, name, soln, nplanets, mask=None, subtract_tc=True):

    if subtract_tc:
        offset = soln['t0']
        if nplanets > 1:
            offset = offset[0]
    else:
        offset = 0
    
    if isinstance(offset, np.ndarray):
        offset = offset.item()
        
    x, y, yerr, x_hr = [data.get(i) for i in 'x y yerr x_hr'.split()]
    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    lcs_hr = soln[f"{name}_light_curves_hr"]
    tra_mod_hr = np.sum(lcs_hr, axis=-1) if lcs_hr.ndim > 1 else lcs_hr

    # subtract every non-transit component, not just the linear model
    sys_mod = get_sys_model(name, soln, int(mask.sum()))

    cor = dict(
        x=x[mask]-offset, 
        y=y[mask]-sys_mod,
        yerr=yerr[mask], 
        x_hr=x_hr-offset, 
        tra_mod_hr=tra_mod_hr
    )
    
    return cor
