import numpy as np
import exoplanet as xo
import pymc3 as pm
import aesara_theano_fallback.tensor as tt
import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess

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
            light_curve = tt.sum(light_curves, axis=-1) + mean + lm
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
