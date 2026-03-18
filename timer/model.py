import numpy as np
import exoplanet as xo
import pymc as pm
import pytensor.tensor as pt
import logging
from . import optim

def bump_model(t, t_center, width, amplitude, theano=True):
    """
    Model a "bump" in a light curve using a simple exponential profile.
    This could be used to model phenomena like spot-crossing during a transit.

    Parameters:
    -----------
    t : array-like
        The time array to evaluate the bump over
    t_center : float
        The time at the center of the bump
    width : float
        The width of the bump (similar to standard deviation in a Gaussian)
    amplitude : float
        The amplitude of the bump
    theano : bool, optional
        If True, use pytensor tensors for compatibility with PyMC (default is True)

    Returns:
    --------
    bump : array-like
        The flux of the bump model evaluated at each time point
    """
    
    if theano:
        # Use pytensor tensors for PyMC compatibility
        bump = amplitude * pt.exp(-(t - t_center)**2 / (2 * width**2))
    else:
        # Use NumPy for standard Python calculations
        bump = amplitude * np.exp(-(t - t_center)**2 / (2 * width**2))
    
    return bump


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
        x = (t - tpeak) / fwhm
        f1 = _fr[0] + _fr[1] * x + _fr[2] * x**2 + _fr[3] * x**3 + _fr[4] * x**4
        f2 = _fd[0] * pt.exp(x * _fd[1]) + _fd[2] * pt.exp(x * _fd[3])
        part1 = pt.switch((t > tpeak - fwhm) & (t <= tpeak), f1, 0)
        part2 = pt.switch(t > tpeak, f2, 0)
        flare = part1 + part2
    else:
        x = (t - tpeak) / fwhm
        f1 = _fr[0] + _fr[1] * x + _fr[2] * x**2 + _fr[3] * x**3 + _fr[4] * x**4
        f2 = _fd[0] * np.exp(x * _fd[1]) + _fd[2] * np.exp(x * _fd[3])
        ix1 = (t > tpeak - fwhm) & (t <= tpeak)
        ix2 = t > tpeak
        flare = np.zeros_like(t)
        flare[ix1] = f1[ix1]
        flare[ix2] = f2[ix2]

    return flare * ampl


def get_rv(key=None, priors=None, dist=None, shape=None, name=None, bounded=None, 
           mu=None, sd=None, lower=None, upper=None, verbose=False, initval=None, bounds=None):
    if priors is not None:
        dist = priors[f'{key}_prior']
    if name is None:
        name = key
    if dist == 'gaussian':
        if priors is not None:
            mu, sd = priors[key], priors[f'{key}_unc']
        if initval is None:
            initval = mu
        if bounds is not None:
            lower_bound, upper_bound = bounds
            rv = BoundedNormal(name, mu=mu, sd=sd, shape=shape, lower=lower_bound, upper=upper_bound)
        else:
            rv = pm.Normal(name, mu=mu, sigma=sd, shape=shape, initval=initval)
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
        if initval is None:
            # Use the original mean value from sys.yaml if available, otherwise use priors[key]
            if priors is not None and f'{key}_initval' in priors:
                initval = priors[f'{key}_initval']
            else:
                initval = priors[key]
        # Ensure initval shape matches expected shape
        if shape is not None and np.isscalar(initval):
            initval = np.full(shape, initval)
        rv = pm.Uniform(name, lower=lower, upper=upper, shape=shape, initval=initval)
        spec = f'{dist}({lower},{upper})'
    else:
        raise ValueError(f'dist={dist} not supported')
    if verbose:
        print(f'{name} ~ {spec}')
    return rv

def BoundedNormal(name, mu, sd, shape, lower=0, upper=1):
    return pm.TruncatedNormal(name, mu=mu, sigma=sd, lower=lower, upper=upper, shape=shape)

def sample(
    model,
    map_soln,
    tune=1000,
    draws=1000,
    chains=2,
    cores= 2
):
    with model:
        trace = pm.sample(
            tune=tune,
            draws=draws,
            initvals=map_soln,
            cores=cores,
            chains=chains,
            target_accept=0.95,
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
    chromatic_flare=False,
    include_bump=False,
    chromatic_bump=False,
    fixed=[],
    verbose=False,
    logp_threshold=1,
    sequential_opt=False,
    use_custom_optimizer=True
):
    logging.info("Building model with optimizer: %s", 'custom' if use_custom_optimizer else 'pymc')

    with pm.Model() as model:

        v = {}

        bands = set([i['band'] for i in datasets.values()])

        # Parameters for the stellar properties
        for band in bands:
            p = f'u_star_{band}'
            if 'u_star' in priors.keys():
                if 'u_star' in fixed:
                    # Convert fixed u_star values to proper tensor for exoplanet
                    u_star_vals = priors['u_star'][band]
                    # Ensure it's a tensor that can be indexed
                    v[p] = pt.as_tensor_variable(u_star_vals)
                else:
                    if priors['u_star_prior'] == 'uniform':
                        # For uniform priors, calculate bounds directly
                        bounds = np.array([0, 1])  # u_star bounds from fit.yaml
                        lower = bounds[0]
                        upper = bounds[1]
                        initval = priors['u_star_initval'][band] if 'u_star_initval' in priors else priors['u_star'][band]
                        v[p] = pm.Uniform(p, lower=lower, upper=upper, shape=2, initval=initval)
                        if verbose:
                            print(f'{p} ~ uniform({lower},{upper})')
                    else:
                        # For Gaussian priors, use the original approach
                        initval = priors['u_star_initval'][band] if 'u_star_initval' in priors else priors['u_star'][band]
                        v[p] = get_rv(
                            name=p,
                            dist=priors['u_star_prior'],
                            shape=2,
                            mu=priors['u_star'][band],
                            sd=priors['u_star_unc'][band],
                            initval=initval,
                            verbose=verbose
                        )
            else:
                v[p] = xo.QuadLimbDark(p)
            v[f'star_{band}'] = xo.LimbDarkLightCurve(v[p])

        if basis == 'duration':
            if 'r_star' in priors:
                v['r_star'] = priors['r_star']
            p = "dur"
            if p in fixed:
                v[p] = priors[p]
            else:
                v[p] = get_rv(key=p, priors=priors, shape=nplanets, verbose=verbose)
        elif basis == 'density':
            raise NotImplementedError

        # flare parameters
        if include_flare:
            # Determine number of flares from priors
            nflares = len(priors['flare_tpeak']) if isinstance(priors['flare_tpeak'], np.ndarray) else 1

            # Shared flare parameters (tpeak and fwhm)
            flare_tpeak = get_rv(
                key='flare_tpeak',
                priors=priors,
                shape=nflares,
                verbose=verbose
            )
            flare_fwhm = get_rv(
                key='flare_fwhm',
                priors=priors,
                shape=nflares,
                verbose=verbose
            )

            # Band-dependent flare amplitude (chromatic_flare)
            if chromatic_flare:
                for band in bands:
                    name = f'flare_ampl_{band}'
                    v[name] = get_rv(
                        key='flare_ampl',
                        name=name,
                        priors=priors,
                        shape=nflares,
                        verbose=verbose
                    )
            else:
                flare_ampl = get_rv(
                    key='flare_ampl',
                    priors=priors,
                    shape=nflares,
                    verbose=verbose
                )

        # bump parameters
        if include_bump:
            # Determine number of bumps from priors
            nbumps = len(priors['bump_tcenter']) if isinstance(priors['bump_tcenter'], np.ndarray) else 1

            bump_tcenter = get_rv(
                key='bump_tcenter',
                priors=priors,
                shape=nbumps,
                verbose=verbose
            )
            bump_width = get_rv(
                key='bump_width',
                priors=priors,
                shape=nbumps,
                verbose=verbose
            )
            # Band-dependent bump amplitude (chromatic_bump)
            if chromatic_bump:
                for band in bands:
                    name = f'bump_ampl_{band}'
                    v[name] = get_rv(
                        key='bump_ampl',
                        name=name,
                        priors=priors,
                        shape=nbumps,
                        verbose=verbose
                    )
            else:
                bump_ampl = get_rv(
                    key='bump_ampl',
                    priors=priors,
                    shape=nbumps,
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
                            key=p,
                            name=name,
                            priors=priors,
                            shape=nplanets,
                            verbose=verbose,
                            bounds=[0, 1]
                        )
                elif p in ['ror', 'b']:
                    v[p] = get_rv(
                        key=p,
                        priors=priors,
                        shape=nplanets, 
                        verbose=verbose,
                        bounds=[0, 1]
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
                ror = pt.mean([v[f'ror_{band}'] for band in bands])
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
            raise ValueError(f'basis={basis} not supported')
            
        # loop over the datasets
        parameters = dict()
        for n,(name,data) in enumerate(datasets.items()):

            x, y, yerr, X, texp, x_hr, band = [data.get(i) for i in 'x y yerr X texp x_hr band'.split()]
            mask = masks[name]
            if mask is None:
                mask = np.ones(len(x), dtype=bool)

            if include_mean:
                # mean flux of the light curve (i.e. the bias term)
                mean = pm.Normal(f"{name}_mean", mu=0.0, sigma=10.0, initval=0)
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
                lm = pm.Deterministic(f"{name}_lm", pt.dot(X[mask], weights))
            else:
                lm = 0

            # Transit jitter & GP parameters
            lower = -10
            upper = np.log(10*np.std(y[mask]))
            log_sigma_lc = get_rv(
                name=f'{name}_log_sigma_lc',
                dist='uniform',
                lower=lower,
                upper=upper,
                initval=-9,
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
                # Get band-specific amplitude if chromatic flare is enabled
                if chromatic_flare:
                    flare_ampl_band = v[f'flare_ampl_{band}']
                else:
                    flare_ampl_band = flare_ampl

                # Handle multiple flares by summing individual flare components
                if nflares == 1:
                    # Single flare - extract scalar values
                    tpeak_val = flare_tpeak[0] if hasattr(flare_tpeak, '__getitem__') else flare_tpeak
                    fwhm_val = flare_fwhm[0] if hasattr(flare_fwhm, '__getitem__') else flare_fwhm
                    ampl_val = flare_ampl_band[0] if hasattr(flare_ampl_band, '__getitem__') else flare_ampl_band
                    flare = aflare1(x[mask], tpeak=tpeak_val, fwhm=fwhm_val, ampl=ampl_val)
                else:
                    # Multiple flares - sum all components
                    flare_total = pt.zeros_like(x[mask])
                    for i in range(nflares):
                        flare_component = aflare1(x[mask], tpeak=flare_tpeak[i], fwhm=flare_fwhm[i], ampl=flare_ampl_band[i])
                        flare_total += flare_component
                    flare = flare_total
                pm.Deterministic(f"{name}_flare", flare)
            else:
                flare = 0

            if include_bump:
                # Get band-specific amplitude if chromatic bump is enabled
                if chromatic_bump:
                    bump_ampl_band = v[f'bump_ampl_{band}']
                else:
                    bump_ampl_band = bump_ampl
                # Handle multiple bumps by summing individual bump components
                if nbumps == 1:
                    # Single bump - extract scalar values
                    tcenter_val = bump_tcenter[0] if hasattr(bump_tcenter, '__getitem__') else bump_tcenter
                    width_val = bump_width[0] if hasattr(bump_width, '__getitem__') else bump_width
                    ampl_val = bump_ampl_band[0] if hasattr(bump_ampl_band, '__getitem__') else bump_ampl_band
                    bump = bump_model(x[mask], t_center=tcenter_val, width=width_val, amplitude=ampl_val)
                else:
                    # Multiple bumps - sum all components
                    bump_total = pt.zeros_like(x[mask])
                    for i in range(nbumps):
                        bump_component = bump_model(x[mask], t_center=bump_tcenter[i], width=bump_width[i], amplitude=bump_ampl_band[i])
                        bump_total += bump_component
                    bump = bump_total
                pm.Deterministic(f"{name}_bump", bump)
            else:
                bump = 0

            # Compute the model light curve
            if chromatic:
                ror = v[f'ror_{band}']
            else:
                ror = v['ror']
            light_curves = (
                v[f'star_{band}'].get_light_curve(orbit=orbit, r=ror, t=x[mask], texp=texp, use_in_transit=False)
                * 1e3
            )
            pm.Deterministic(f"{name}_light_curves", light_curves)
            light_curve = pt.sum(light_curves, axis=-1) + mean + lm + flare + bump
            resid = y[mask] - light_curve

            # Compute high-res model light curve
            light_curves_hr = (
                v[f'star_{band}'].get_light_curve(orbit=orbit, r=ror, t=x_hr, texp=texp, use_in_transit=False)
                * 1e3
            )
            pm.Deterministic(f"{name}_light_curves_hr", light_curves_hr)

            # GP likelihood
            if use_gp:
                raise NotImplementedError("GP support is temporarily disabled")
            else:
                y_observed = pm.Normal(
                    f"{name}_y_observed", 
                    mu=light_curve,
                    sigma=np.sqrt(pt.exp(2*log_sigma_lc) + yerr[mask]**2), 
                    observed=y[mask]
                )

            # Compute and save the phased light curve models
            pm.Deterministic(
                f"{name}_lc_pred",
                1e3
                * v[f'star_{band}'].get_light_curve(
                    orbit=orbit, r=ror, t=x[mask], texp=texp, use_in_transit=False
                )[..., 0],
            )

        # track the implied density
        pm.Deterministic("rho_circ", orbit.rho_star)
        
        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.initial_point()

        # Get initial log probability - filter to only include value variables
        start_filtered = {k: v for k, v in start.items() if k in [var.name for var in model.value_vars]}
        
        # Log initial parameter values for debugging
        logging.info("Initial parameter values:")
        for k, v in start.items():
            if isinstance(v, np.ndarray):
                if v.size > 10:
                    logging.info(f"  {k}: shape={v.shape}, mean={np.mean(v):.6f}, std={np.std(v):.6f}")
                else:
                    logging.info(f"  {k}: {v}")
            else:
                logging.info(f"  {k}: {v}")
        
        try:
            logp_init = model.point_logps(start_filtered)
            logging.info("Initial logp evaluation successful:")
            for k, v in logp_init.items():
                logging.info(f"  {k}: {v}")
        except Exception as e:
            logging.error(f"Initial logp evaluation failed: {e}")
            raise
        
        # optimize all parameters
        if use_custom_optimizer:
            logging.info("Using custom optimizer")
            try:
                map_soln = optim.optimize(start=start, model=model)
                logging.info("Custom optimizer completed successfully")
            except Exception as e:
                logging.warning(f"Custom optimizer failed: {e}")
                logging.info("Falling back to PyMC find_MAP")
                map_soln = pm.find_MAP(start=start)
        else:
            logging.info("Using PyMC find_MAP")
            map_soln = pm.find_MAP(start=start)

        # Get final log probability after MAP optimization - filter to only include value variables
        map_soln_filtered = {k: v for k, v in map_soln.items() if k in [var.name for var in model.value_vars]}
        logp_final = model.point_logps(map_soln_filtered)
        
        # Sum up the individual log probabilities to get total
        logp_init_total = sum(logp_init.values())
        logp_final_total = sum(logp_final.values())
        
        print(f"Initial log probability: {logp_init_total:.2f}")
        print(f"Final log probability: {logp_final_total:.2f}")
        print(f"Log probability improvement: {logp_final_total - logp_init_total:.2f}")
        
        if sequential_opt and (logp_final_total - logp_init_total < logp_threshold):
            print("Optimization improvement below threshold. Attempting sequential optimization.")

            # only linear systematics model
            map_soln = pm.find_MAP(
                start=map_soln, vars=[parameters[name] for name in datasets.keys()]
            )

            # only transit
            pnames = [i for i in 't0 b dur'.split() if i not in fixed]
            if 'ror' not in fixed:
                if chromatic:
                    pnames += [f'ror_{band}' for band in bands]
                else:
                    pnames += ['ror']
            map_soln = pm.find_MAP(
                start=map_soln, vars=[v[p] for p in pnames]
            )

            # only noise
            map_soln = pm.find_MAP(
                start=map_soln, vars=[parameters[f'{name}_noise'] for name in datasets.keys()]
            )

            # # sequential transit parameter optimization
            # pnames = 't0 dur b'.split()
            # for p in [i for i in pnames if i not in fixed]:
            #     map_soln = pm.find_MAP(start=map_soln, vars=[v[p]])

            # final optimization of all parameters
            map_soln = pm.find_MAP(start=map_soln)

    return model, map_soln
