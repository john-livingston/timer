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

def get_residuals(name, y, soln, mask=None, use_gp=False):

    if mask is None:
        mask = np.ones(len(y), dtype=bool)

    mean = soln[f"{name}_mean"]
    lin_mod = soln[f'{name}_lm'] if f'{name}_lm' in soln.keys() else np.zeros(mask.sum())
    tra_mod = soln[f"{name}_light_curves"]
    # Sum over planets axis if multiple planets
    if tra_mod.ndim > 1:
        tra_mod = np.sum(tra_mod, axis=1)

    # Apply mask to tra_mod if it has the same length as y
    if len(tra_mod) == len(y):
        tra_mod = tra_mod[mask]

    # Add flare and bump components if they exist
    flare_mod = soln[f"{name}_flare"] if f"{name}_flare" in soln.keys() else 0
    bump_mod = soln[f"{name}_bump"] if f"{name}_bump" in soln.keys() else 0

    sys_mod = lin_mod + mean + flare_mod + bump_mod
    if use_gp:
        gp_mod = soln[f"{name}_gp_pred"]
        sys_mod += gp_mod

    return y[mask] - tra_mod - sys_mod

def get_map_soln(trace):
    # arviz trace is an InferenceData object
    max_lp = trace.sample_stats["lp"].max()
    ix = trace.sample_stats["lp"] == max_lp
    trace_map = trace.posterior.where(ix, drop=True)
    flat_samps_map = trace_map.stack(sample=("chain", "draw"))
    soln = {k: v.values.item() if v.values.size == 1 else v.values for k, v in flat_samps_map.data_vars.items()}
    return soln, max_lp.values.item()

def get_var_names(data, bands, fit_basis, use_gp, fixed,
                  chromatic=False, log_sigma=True, weights=False):

    var_names = ['t0']
    for par in 'period b dur'.split():
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
    for name in data.keys():
        if use_gp:
            var_names += [f'{name}_log_rho_gp', f'{name}_log_sigma_gp']
        if weights:
            var_names += [f'{name}_weights']
        if log_sigma:
            var_names += [f'{name}_log_sigma_lc']
    return var_names

def get_summary(trace, data, bands, fit_basis, use_gp, fixed,
                chromatic=False, log_sigma=True, weights=False):

    var_names = get_var_names(data, bands, fit_basis, use_gp, fixed,
                              chromatic=chromatic, log_sigma=log_sigma, weights=weights)
    summary = az.summary(
        trace,
        var_names=var_names
    )
    return summary

def get_outlier_mask(x, y, name, map_soln, use_gp, nsig=7, include_flare=False, include_bump=False, fp=None):
    mod = (
        + map_soln[f"{name}_mean"]
        + np.sum(map_soln[f"{name}_light_curves"], axis=-1)
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

    bands_ = [f'{band}*' if band in 'griz' else band for band in bands]
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

def compute_ic(map_soln, max_logp, nparams, ndata, method='BIC', verbose=True):

    if method == 'BIC':
        ic = -2 * max_logp + nparams * np.log(ndata)
    elif method == 'AIC':
        ic = 2 * nparams - 2 * max_logp
    elif method == 'AICc':
        ic = 2 * nparams - 2 * max_logp
        ic += 2 * (nparams**2 + nparams) / (ndata - nparams - 1)

    if verbose:
        print('Number of datapoints: {}'.format(ndata))
        print('Number of parameters: {}'.format(nparams))
        print('Max logp = {}'.format(max_logp))
        print('{} = {}'.format(method, ic))

    return float(ic)

def get_corrected(data, name, soln, nplanets, 
                  mask=None, trace=None, use_gp=False, median=True, subtract_tc=True):
    
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

    if trace is None or not median:
        if f'{name}_mean' in soln.keys():
            mean = soln[f"{name}_mean"]
        else:
            mean = 0
        lcjit = np.exp(soln[f'{name}_log_sigma_lc'])
        lin_mod = soln[f'{name}_lm']
        tra_mod = np.sum(soln[f"{name}_light_curves"], axis=1)
        tra_mod_hr = np.sum(soln[f"{name}_light_curves_hr"], axis=1)
    else:
        if f'{name}_mean' in soln.keys():
            mean = np.median(trace[f"{name}_mean"])
        else:
            mean = 0
        lcjit = np.exp(np.median(trace[f'{name}_log_sigma_lc']))
        lin_mod = np.median(trace[f'{name}_lm'], axis=0)
        tra_mod = np.sum(np.median(trace[f"{name}_light_curves"], axis=0), axis=1)
        tra_mod_hr = np.sum(np.median(trace[f"{name}_light_curves_hr"], axis=0), axis=1)
    
    sys_mod = lin_mod.flatten() + mean
    
    cor = dict(
        x=x[mask]-offset, 
        y=y[mask]-sys_mod,
        yerr=yerr[mask], 
        x_hr=x_hr-offset, 
        tra_mod_hr=tra_mod_hr
    )
    
    return cor
