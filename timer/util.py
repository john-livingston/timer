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
    tra_mod = np.sum(soln[f"{name}_light_curves"], axis=-1)

    sys_mod = lin_mod + mean
    if use_gp:
        gp_mod = soln[f"{name}_gp_pred"]
        sys_mod += gp_mod

    return y[mask] - tra_mod - sys_mod

def get_map_soln(trace, inferencedata=False):
    if inferencedata:
        # doesn't work!
        max_lp = trace.sample_stats['lp'].max()
        ix = trace.sample_stats['lp'] == max_lp
        trace_map = trace.posterior.where(ix, drop=True)
        flat_samps_map = trace_map.stack(sample=("chain", "draw"))
        soln = {k:np.array(v['data']).flatten() for k,v in flat_samps_map.to_dict()['data_vars'].items()}
    else:
        idx = trace.get_sampler_stats('model_logp').argmax()
        max_lp = trace.get_sampler_stats('model_logp')[idx]
        vals = [trace.get_values(vn)[idx] for vn in trace.varnames]
        soln = dict(zip(trace.varnames, vals))
    return soln, max_lp

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

def get_outlier_mask(x, y, name, map_soln, use_gp, nsig=7, include_flare=False, fp=None):
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
    resid = y - mod
    rms = np.sqrt(np.median(resid**2))
    mask = np.abs(resid) < nsig * rms

    if fp is not None and mask.sum() < mask.size:
        plot_outliers(x, resid, mask, fp=fp)

    return mask

def get_priors(fit_basis, star, planets, fixed, bands, tc_guess, tc_guess_unc, unif=[], unif_nsig=10):

    priors = {}
    priors['r_star'] = np.array(star['radius'][0])
    priors['r_star_unc'] = np.array(star['radius'][1])
    if fit_basis == 'mstar/rstar':
        priors['m_star'] = np.array(star['mass'][0])
        priors['m_star_unc'] = np.array(star['mass'][1])
    elif fit_basis == 'duration':
        pass
    elif fit_basis == 'density':
        raise NotImplementedError
    else:
        print("basis not supported")

    bands_ = [f'{band}*' if band in 'griz' else band for band in bands]
    ldp = [ld.claret(band, *star['teff'], *star['logg'], *star['feh']) for band in bands_]
    priors['u_star'] = {band:ld[::2] for band,ld in zip(bands, ldp)}
    priors['u_star_unc'] = {band:ld[1::2] for band,ld in zip(bands, ldp)}

    for par in 'period dur ror b'.split():
        priors[par] = np.array([i[par][0] for i in planets])
        if par not in fixed:
            if par in unif:
                priors[f'{par}_prior'] = 'uniform'
                priors[f'{par}_unc'] = unif_nsig * np.array([i[par][1] for i in planets])
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
