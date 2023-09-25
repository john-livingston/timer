import numpy as np
import exoplanet as xo
import pymc3 as pm
import aesara_theano_fallback.tensor as tt
import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess


def aflare1(t, tpeak, fwhm, ampl, theano=True):
    # adapted from: https://github.com/MNGuenther/allesfitter/blob/master/allesfitter/flares/aflare.py
    '''
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Use this function for fitting classical flares with most curve_fit
    tools.
    Note: this model assumes the flux before the flare is zero centered
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The time of the flare peak
    fwhm : float
        The "Full Width at Half Maximum", timescale of the flare
    ampl : float
        The amplitude of the flare
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    '''

    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]
    
    if theano:
        fun1 = lambda x: (  _fr[0]+                       # 0th order
                            _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                            _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                            _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                            _fr[4]*((x-tpeak)/fwhm)**4. ) # 4th order
        fun2 = lambda x: (  _fd[0]*tt.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                            _fd[2]*tt.exp( ((x-tpeak)/fwhm)*_fd[3] ) )
        part1 = tt.switch((t > tpeak-fwhm) & (t <= tpeak), fun1(t), 0)
        part2 = tt.switch(t > tpeak, fun2(t), 0)
        flare = part1 + part2
    else:
        fun1 = lambda x: (  _fr[0]+                       # 0th order
                            _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                            _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                            _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                            _fr[4]*((x-tpeak)/fwhm)**4. ) # 4th order
        fun2 = lambda x: (  _fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                            _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ) )
        ix1 = (t > tpeak-fwhm) & (t <= tpeak) 
        ix2 = t > tpeak
        flare = np.zeros_like(t)
        flare[ix1] = fun1(t[ix1])
        flare[ix2] = fun2(t[ix2])

    return flare * ampl


def get_rv(key=None, priors=None, dist=None, shape=None, name=None, bounded=None, 
           mu=None, sd=None, lower=None, upper=None, verbose=False, testval=None):
    if priors is not None:
        dist = priors[f'{key}_prior']
    if name is None:
        name = key
    if dist == 'gaussian':
        if priors is not None:
            mu, sd = priors[key], priors[f'{key}_unc']
        fun = pm.Normal if bounded is None else bounded
        if testval is None:
            testval = mu
        rv = fun(name, mu=mu, sd=sd, shape=shape, testval=testval)
        spec = f'{dist}({mu},{sd})'
    elif dist == 'uniform':
        if priors is not None:
            lower = priors[key] - priors[f'{key}_unc']/2
            upper = priors[key] + priors[f'{key}_unc']/2
        if bounded is not None:
            ix = lower < bounded.lower
            if ix.any():
                lower[ix] = bounded.lower
            ix = upper > bounded.upper
            if ix.any():
                upper[ix] = bounded.upper
        if testval is None:
            testval = priors[key]
        rv = pm.Uniform(name, lower=lower, upper=upper, shape=shape, testval=testval)
        spec = f'{dist}({lower},{upper})'
    else:
        print(f'dist={dist} not supported')
    if verbose:
        print(f'{name} ~ {spec}')
    return rv

def sample(
    model,
    map_soln,
    tune=1000,
    draws=1000,
    chains=2,
    cores= 2,
    inferencedata=False
):
    with model:
        trace = pm.sample(
            tune=tune,
            draws=draws,
            start=map_soln,
            cores=cores,
            chains=chains,
            target_accept=0.95,
            return_inferencedata=inferencedata,
            init="adapt_full",
        )

    return trace

def build(
    datasets,
    priors,
    nplanets,
    masks={},
    start=None,
    basis='duration',
    chromatic=False,
    use_gp=False,
    include_mean=True,
    include_flare=False,
    fixed=[],
    verbose=False
):

    with pm.Model() as model:

        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
        BoundedUniform = pm.Bound(pm.Uniform, lower=0, upper=1)

        v = {}

        bands = set([i['band'] for i in datasets.values()])

        # Parameters for the stellar properties
        for band in bands:
            p = f'u_star_{band}'
            if 'u_star' in priors.keys():
                if 'u_star' in fixed:
                    v[p] = priors['u_star'][band]
                else:
                    v[p] = get_rv(
                        name=p,
                        dist='gaussian',
                        shape=2,
                        mu=priors['u_star'][band],
                        sd=priors['u_star_unc'][band],
                        verbose=verbose
                    )
            else:
                v[p] = xo.QuadLimbDark(p)
            v[f'star_{band}'] = xo.LimbDarkLightCurve(v[p])

        if basis == 'duration':
            v['r_star'] = priors['r_star']
            p = "dur"
            if p in fixed:
                v[p] = priors[p]
            else:
                bounded = BoundedNormal if priors[f'{p}_prior'] == 'gaussian' else BoundedUniform
                v[p] = get_rv(key=p, priors=priors, shape=nplanets, verbose=verbose, bounded=bounded)
        elif basis == 'density':
            raise NotImplementedError

        # flare parameters
        if include_flare:
            flare_tpeak = get_rv(
                key='flare_tpeak',
                priors=priors,
                shape=1,
                verbose=verbose
            )
            flare_fwhm = get_rv(
                key='flare_fwhm',
                priors=priors,
                shape=1,
                verbose=verbose
            )
            flare_ampl = get_rv(
                key='flare_ampl',
                priors=priors,
                shape=1,
                verbose=verbose
            )

        # parameters for the planets
        for p in "t0 period ror b".split():
            if p in fixed:
                v[p] = priors[p]
            else:
                if p == 'ror' and chromatic:
                    for band in bands:
                        name = f'ror_{band}'
                        v[name] = get_rv(
                            name=name,
                            dist='gaussian',
                            shape=nplanets,
                            mu=priors[p],
                            sd=priors[f'{p}_unc'],
                            bounded=BoundedNormal,
                            verbose=verbose
                        )
                elif p in ['ror', 'b']:
                    bounded = BoundedNormal if priors[f'{p}_prior'] == 'gaussian' else BoundedUniform
                    v[p] = get_rv(
                        key=p,
                        priors=priors,
                        shape=nplanets, 
                        bounded=bounded,
                        verbose=verbose
                    )
                else:
                    v[p] = get_rv(
                        key=p,
                        priors=priors,
                        shape=nplanets, 
                        verbose=verbose
                    )
        
        # Orbit model
        if basis == 'duration':
            if chromatic:
                ror = tt.mean([v[f'ror_{band}'] for band in bands])
            else:
                ror = v['ror']
            orbit = xo.orbits.KeplerianOrbit(
                duration=v['dur'],
                period=v['period'],
                t0=v['t0'],
                b=v['b'],
                ror=ror
            )
        else:
            print('basis not supported')
            
        # loop over the datasets
        parameters = dict()
        for n,(name,data) in enumerate(datasets.items()):

            x, y, yerr, X, texp, x_hr, band = [data.get(i) for i in 'x y yerr X texp x_hr band'.split()]
            mask = masks[name]
            if mask is None:
                mask = np.ones(len(x), dtype=bool)

            if include_mean:
                # mean flux of the light curve (i.e. the bias term)
                mean = pm.Normal(f"{name}_mean", mu=0.0, sd=10.0, testval=0)
            else:
                mean = 0

            # linear systematics model
            if X is not None:
                mu = np.zeros(X[mask].shape[1])
                sd = mu + 1e3
                shape = X[mask].shape[1]
                weights = get_rv(
                    name=f'{name}_weights', 
                    dist='gaussian', 
                    shape=shape, 
                    mu=mu, 
                    sd=sd, 
                    verbose=verbose
                )
                parameters[name] = weights
                lm = pm.Deterministic(f"{name}_lm", tt.dot(X[mask], weights))
            else:
                lm = 0

            # Transit jitter & GP parameters
            lower = -10
            upper = np.log(10*np.std(y[mask]))
            log_sigma_lc = get_rv(
                name=f'{name}_log_sigma_lc',
                dist='uniform',
                shape=1,
                lower=lower,
                upper=upper,
                testval=-9,
                verbose=verbose
            )
            priors[f'{name}_log_sigma_lc'] = (upper+lower)/2
            priors[f'{name}_log_sigma_lc_unc'] = upper-lower
            priors[f'{name}_log_sigma_lc_prior'] = 'uniform'
            parameters[f'{name}_noise'] = log_sigma_lc
            if use_gp:
                log_rho_gp = get_rv(
                    name=f'{name}_log_rho_gp', 
                    dist='gaussian', 
                    mu=0, 
                    sd=10, 
                    verbose=verbose
                )
                log_sigma_gp = get_rv(
                    name=f'{name}_log_sigma_gp', 
                    dist='gaussian', 
                    mu=np.log(np.std(y[mask])), 
                    sd=10, 
                    verbose=verbose
                )
                parameters[f'{name}_gp'] = [log_rho_gp, log_sigma_gp]

            if include_flare:
                flare = aflare1(x[mask], tpeak=flare_tpeak, fwhm=flare_fwhm, ampl=flare_ampl)
                pm.Deterministic(f"{name}_flare", flare)
            else:
                flare = 0

            # Compute the model light curve
            if chromatic:
                ror = v[f'ror_{band}']
            else:
                ror = v['ror']
            light_curves = (
                v[f'star_{band}'].get_light_curve(orbit=orbit, r=ror, t=x[mask], texp=texp)
                * 1e3
            )
            pm.Deterministic(f"{name}_light_curves", light_curves)
            light_curve = tt.sum(light_curves, axis=-1) + mean + lm + flare
            resid = y[mask] - light_curve

            # Compute high-res model light curve
            light_curves_hr = (
                v[f'star_{band}'].get_light_curve(orbit=orbit, r=ror, t=x_hr, texp=texp)
                * 1e3
            )
            pm.Deterministic(f"{name}_light_curves_hr", light_curves_hr)

            # GP likelihood
            if use_gp:
                kernel = terms.SHOTerm(
                    sigma=tt.exp(log_sigma_gp),
                    rho=tt.exp(log_rho_gp),
                    Q=1 / np.sqrt(2),
                )
                gp = GaussianProcess(kernel, t=x[mask], yerr=np.sqrt(tt.exp(2*log_sigma_lc) + yerr[mask]**2))
                gp.marginal(f"{name}_gp", observed=resid)
                pm.Deterministic(f"{name}_gp_pred", gp.predict(resid))
            else:
                y_observed = pm.Normal(
                    f"{name}_y_observed", 
                    mu=light_curve,
                    sd=np.sqrt(tt.exp(2*log_sigma_lc) + yerr[mask]**2), 
                    observed=y[mask]
                )

            # Compute and save the phased light curve models
            pm.Deterministic(
                f"{name}_lc_pred",
                1e3
                * v[f'star_{band}'].get_light_curve(
                    orbit=orbit, r=ror, t=x[mask], texp=texp
                )[..., 0],
            )

        # track the implied density
        pm.Deterministic("rho_circ", orbit.rho_star)
        
        print(model.check_test_point())

        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.test_point
        # all
        map_soln = pmx.optimize(start=start)

#         # lm
#         map_soln = pmx.optimize(
#             start=map_soln, vars=[parameters[name] for name in datasets.keys()]
#         )
#         # transit
#         pnames = [i for i in 't0 b dur'.split() if i not in fixed]
#         if 'ror' not in fixed:
#             if chromatic:
#                 pnames += [f'ror_{band}' for band in bands]
#             else:
#                 pnames += ['ror']
#         map_soln = pmx.optimize(
#             start=map_soln, vars=[v[p] for p in pnames]
#         )
#         # gp
#         if use_gp:
#             for name in datasets.keys():
#                 map_soln = pmx.optimize(
#                     start=map_soln, vars=parameters[f'{name}_gp']
#                 )
#         # noise
#         map_soln = pmx.optimize(
#             start=map_soln, vars=[parameters[f'{name}_noise'] for name in datasets.keys()]
#         )

#         # # sequential optimization
#         # pnames = 't0 dur b'.split()
#         # for p in [i for i in pnames if i not in fixed]:
#         #     map_soln = pmx.optimize(start=map_soln, vars=[v[p]])
#         # # simultaneous optimization
#         # map_soln = pmx.optimize(
#         #     start=map_soln, vars=[v[p] for p in pnames]
#         # )

#         # simultaneous optimization of all parameters
#         map_soln = pmx.optimize(start=map_soln)

    return model, map_soln
