import os
import dill as pickle
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from astropy.time import Time
import logging
import argparse
import shutil

from . import io, util, plot, model


defaults = dict(

    model = dict(
        fixed = [], # transit model parameters (duration basis): m_star r_star t0 period ror b dur u_star
        fit_basis = 'duration',
        chromatic = False,
        include_mean = True, # the mean flux value (should not be True if add_bias=True)
        include_flare = False,
        chromatic_flare = False,
        include_bump = False,
        chromatic_bump = False,
        use_gp = False,
        use_custom_optimizer = True,
    ),

    sampler = dict(
        tune = 2000,
        draws = 2000,
        chains = 2,
        cores = 2,
        clobber = False,
    ),

    data = dict(
        spline = False,
        spline_knots = 5,
        add_bias = False, # the column of 1s in the design matrix
        quadratic = False,
        trend = None,
        trim_beg = None,
        trim_end = None,
        clip = False,
        clip_nsig = 7,
        binsize = 5/1440,
        chunk_offset = False,
        chunk_thresh = 0,
        format = 'generic'
    ),
)

class TransitFit:

    def __init__(self, sys_params, fit_params, wd='.', outdir='out', _force_load_saved=False):
        self.sys_params = sys_params
        self.fit_params = fit_params
        self.wd = os.path.abspath(wd)
        self.outdir = os.path.join(self.wd, outdir)
        self._force_load_saved = _force_load_saved
        self.validate()
        self.setup()
        self.save_input_files()
        self.load_data()
        self.load_saved()
        self.set_priors()
        
    @classmethod
    def from_dir(cls, wd, outdir='out'):
        fp = os.path.join(wd, 'fit.yaml')
        fit_params = yaml.load(open(fp), Loader=yaml.FullLoader)
        fp = os.path.join(wd, 'sys.yaml')
        sys_params = yaml.load(open(fp), Loader=yaml.FullLoader)
        return cls(sys_params, fit_params, wd=wd, outdir=outdir, _force_load_saved=True)

    def validate(self):
        
        # set model defaults
        for k,v in defaults['model'].items():
            if k not in self.fit_params.keys():
                logging.info(f'setting default: {k} = {v}')
                self.fit_params[k] = v
        
        # set sampler defaults
        for k,v in defaults['sampler'].items():
            if k not in self.fit_params.keys():
                logging.info(f'setting default: {k} = {v}')
                self.fit_params[k] = v

        # set data defaults
        for k,v in defaults['data'].items():
            for n in self.fit_params['data'].keys():
                if k not in self.fit_params['data'][n].keys():
                    logging.info(f'setting default for {n}: {k} = {v}')
                    self.fit_params['data'][n][k] = v
        
        # Sanity checks for incompatible configurations
        self._validate_parameter_conflicts()

    def _validate_parameter_conflicts(self):
        """Check for incompatible parameter configurations"""
        fixed_params = set(self.fit_params.get('fixed', []))
        uniform_params = set(self.fit_params.get('uniform', {}).keys())

        # Can't be both fixed and have priors
        conflicts = fixed_params.intersection(uniform_params)
        if conflicts:
            raise ValueError(f"Parameters {list(conflicts)} cannot be both fixed and in uniform priors")

        # Get number of planets for validation
        nplanets = len(self.fit_params['planets'])

        # Validate bounds
        for param, bounds in self.fit_params.get('uniform', {}).items():
            # Check if we have planet-indexed bounds or single bounds
            if param in ['period', 'dur', 'ror', 'b'] and isinstance(bounds[0], (list, tuple)):
                # Planet-indexed bounds: [[low1,high1], [low2,high2], ...]
                if len(bounds) != nplanets:
                    raise ValueError(f"Number of {param} bounds ({len(bounds)}) must match number of planets ({nplanets})")

                for i, planet_bounds in enumerate(bounds):
                    if len(planet_bounds) != 2 or planet_bounds[0] >= planet_bounds[1]:
                        raise ValueError(f"Invalid bounds for '{param}' planet {i}: {planet_bounds}")

                    lower, upper = planet_bounds
                    if param == 'ror' and (lower < 0 or upper > 1):
                        raise ValueError(f"ror must be in [0,1] for planet {i}, got: {planet_bounds}")
                    if param == 'b' and lower < 0:
                        raise ValueError(f"b cannot be negative for planet {i}, got: {planet_bounds}")
            else:
                # Single bounds for all planets or non-planet parameters
                if len(bounds) != 2 or bounds[0] >= bounds[1]:
                    raise ValueError(f"Invalid bounds for '{param}': {bounds}")

                lower, upper = bounds
                if param == 'ror' and (lower < 0 or upper > 1):
                    raise ValueError(f"ror must be in [0,1], got: {bounds}")
                if param == 'b' and lower < 0:
                    raise ValueError(f"b cannot be negative, got: {bounds}")

    def setup(self):
        fit_params = self.fit_params
        # model settings
        self.nplanets = len(fit_params['planets'])
        self.fixed = fit_params['fixed']
        self.fit_basis = fit_params['fit_basis']
        self.planets = fit_params['planets']
        self.chromatic = fit_params['chromatic']
        self.include_mean = fit_params['include_mean']
        self.include_flare = fit_params['include_flare']
        self.chromatic_flare = fit_params['chromatic_flare']
        self.include_bump = fit_params['include_bump']
        self.chromatic_bump = fit_params['chromatic_bump']
        self.use_gp = fit_params['use_gp']
        self.use_custom_optimizer = fit_params['use_custom_optimizer']
        self.uniform = fit_params.get('uniform', {})
        if self.include_flare:
            self.flare = self.fit_params['flare']
        if self.include_bump:
            self.bump = self.fit_params['bump']
        # sampler settings
        self.tune = fit_params['tune']
        self.draws = fit_params['draws']
        self.chains = fit_params['chains']
        self.cores = fit_params['cores']
        self.clobber = fit_params['clobber']
        # initialize
        self.model = None
        self.trace = None
        self.masks = {}
        self.bands = []

    def save_input_files(self):
        """Save input YAML files to output directory for audit purposes."""
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)
        
        # Copy fit.yaml and sys.yaml to output directory
        fit_yaml_src = os.path.join(self.wd, 'fit.yaml')
        sys_yaml_src = os.path.join(self.wd, 'sys.yaml')
        
        if os.path.exists(fit_yaml_src):
            fit_yaml_dst = os.path.join(self.outdir, 'fit.yaml')
            shutil.copy2(fit_yaml_src, fit_yaml_dst)
            logging.info(f'Saved input file: {fit_yaml_dst}')
        
        if os.path.exists(sys_yaml_src):
            sys_yaml_dst = os.path.join(self.outdir, 'sys.yaml')
            shutil.copy2(sys_yaml_src, sys_yaml_dst)
            logging.info(f'Saved input file: {sys_yaml_dst}')

    def load_data(self):
        self.data = {}
        data = self.fit_params['data']
        for n in data.keys():
            fn = data[n]['file']
            b = data[n]['band']
            if b not in self.bands:
                self.bands.append(b)
            fp = os.path.join(self.wd, fn)
            if data[n]['format'] == 'generic':
                read_fn = io.read_generic
            elif data[n]['format'] == 'afphot':
                read_fn = io.read_afphot
            else:
                raise ValueError("format must be 'generic' or 'afphot'")
            x, y, yerr, X, texp, x_hr, ref_time = read_fn(
                fp, 
                binsize=data[n]['binsize'],
                spline=data[n]['spline'],
                spline_knots=data[n]['spline_knots'],
                add_bias=data[n]['add_bias'],
                quad=data[n]['quadratic'],
                trend=data[n]['trend'],
                trim_beg=data[n]['trim_beg'],
                trim_end=data[n]['trim_end'],
                chunk_offset=data[n]['chunk_offset'],
                chunk_thresh=data[n]['chunk_thresh'],
            )
            data_iso = [Time(i+ref_time, format='jd').iso for i in (x.min(), x.max())]
            logging.info(f'loading data: {fn}')
            logging.info(f'data span: {data_iso[0]} - {data_iso[1]}')
            logging.info(f'ref. time: {ref_time}')
            self.data[n] = dict(x=x, y=y, yerr=yerr, X=X, texp=texp, x_hr=x_hr, band=b, ref_time=ref_time)
            self.masks[n] = None
        ref_times = [v['ref_time'] for k,v in self.data.items()]
        self.ref_time = min(ref_times)
        for k,v in self.data.items():
            if v['ref_time'] != self.ref_time:
                delta = v['ref_time'] - self.ref_time
                v['x'] += delta
                v['x_hr'] += delta
                v['ref_time'] = self.ref_time

    def load_saved(self):
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        # Load saved files if clobber is False OR if force_load_saved is True (from from_dir)
        if not self.clobber or self._force_load_saved:
            if os.path.exists(os.path.join(self.outdir, 'mask.pkl')):
                logging.info('loading mask(s) from mask.pkl')
                self.masks = pickle.load(open(os.path.join(self.outdir, 'mask.pkl'), 'rb'))
            if os.path.exists(os.path.join(self.outdir, 'model.pkl')):
                logging.info('loading model from model.pkl')
                self.model = pickle.load(open(os.path.join(self.outdir, 'model.pkl'), 'rb'))
            if os.path.exists(os.path.join(self.outdir, 'map.pkl')):
                logging.info('loading MAP solution from map.pkl')
                self.map_soln = pickle.load(open(os.path.join(self.outdir, 'map.pkl'), 'rb'))
            if os.path.exists(os.path.join(self.outdir, 'trace.pkl')):
                logging.info('loading trace from trace.pkl')
                self.trace = pickle.load(open(os.path.join(self.outdir, 'trace.pkl'), 'rb'))

    def _add_log_sigma_lc_priors(self):
        """Add log_sigma_lc priors for each dataset based on data std"""
        for name, data in self.data.items():
            y = data['y']
            mask = self.masks.get(name, np.ones(len(y), dtype=bool))
            lower = -10
            upper = np.log(10*np.std(y[mask]))
            self.priors[f'{name}_log_sigma_lc'] = (upper+lower)/2
            self.priors[f'{name}_log_sigma_lc_unc'] = upper-lower
            self.priors[f'{name}_log_sigma_lc_prior'] = 'uniform'

    def plot_data(self):
        logging.info("plotting data")
        for name,data in self.data.items():
            x, y, yerr = [data.get(i) for i in 'x y yerr'.split()]
            ref_time = data['ref_time']
            plt.errorbar(x, y, yerr, ls='', label=name)
            plt.xlabel(f"time [BJD$-${ref_time}]")
            plt.ylabel("relative flux [ppt]")
        fn = f'data.png'
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(self.outdir, fn))

    def set_priors(self):
        planets = [self.sys_params['planets'][k] for k in self.planets]
        x_mean = np.mean([v['x'].mean() for k,v in self.data.items()])
        tc_guess, tc_guess_unc = util.get_tc_prior(self.fit_params, x_mean, self.ref_time)
        self.priors = util.get_priors(
            self.fit_basis, self.sys_params['star'],
            planets, self.fixed, self.bands,
            tc_guess, tc_guess_unc, uniform=self.uniform
        )
        # Add log_sigma_lc priors based on data
        self._add_log_sigma_lc_priors()
        if self.include_flare:
            # Determine number of flares from vector length (use tpeak as reference)
            if isinstance(self.flare['tpeak'], (list, tuple)):
                self.nflares = len(self.flare['tpeak'])
            else:
                self.nflares = 1

            for p in 'tpeak fwhm ampl'.split():
                param_val = self.flare[p]
                if isinstance(param_val, (list, tuple)):
                    if len(param_val) != self.nflares:
                        raise ValueError(f"All flare parameters must have same length. "
                                       f"tpeak has {self.nflares} values, {p} has {len(param_val)}")
                    self.priors[f'flare_{p}'] = np.array(param_val)
                else:
                    # Single value - replicate for all flares
                    self.priors[f'flare_{p}'] = np.array([param_val] * self.nflares)

                self.priors[f'flare_{p}_prior'] = self.flare[f'{p}_prior']
                self.priors[f'flare_{p}_unc'] = self.flare[f'{p}_unc']

            # Adjust tpeak for reference time
            p = 'tpeak'
            self.priors[f'flare_{p}'] = self.priors[f'flare_{p}'] - self.ref_time
        if self.include_bump:
            # Determine number of bumps from vector length (use tcenter as reference)
            if isinstance(self.bump['tcenter'], (list, tuple)):
                self.nbumps = len(self.bump['tcenter'])
            else:
                self.nbumps = 1

            for p in 'tcenter width ampl'.split():
                param_val = self.bump[p]
                if isinstance(param_val, (list, tuple)):
                    if len(param_val) != self.nbumps:
                        raise ValueError(f"All bump parameters must have same length. "
                                       f"tcenter has {self.nbumps} values, {p} has {len(param_val)}")
                    self.priors[f'bump_{p}'] = np.array(param_val)
                else:
                    # Single value - replicate for all bumps
                    self.priors[f'bump_{p}'] = np.array([param_val] * self.nbumps)

                self.priors[f'bump_{p}_prior'] = self.bump[f'{p}_prior']
                self.priors[f'bump_{p}_unc'] = self.bump[f'{p}_unc']

            # Adjust tcenter for reference time
            p = 'tcenter'
            self.priors[f'bump_{p}'] = self.priors[f'bump_{p}'] - self.ref_time

    def build_model(self, start=None, force=False, verbose=False, plot=True):
        if force or self.clobber or self.model is None:
            logging.info('building and optimizing model')
            data, priors, masks = self.data, self.priors, self.masks
            nplanets, use_gp, chromatic = self.nplanets, self.use_gp, self.chromatic
            fixed, fit_basis = self.fixed, self.fit_basis
            include_mean, include_flare, chromatic_flare, include_bump, chromatic_bump = self.include_mean, self.include_flare, self.chromatic_flare, self.include_bump, self.chromatic_bump
            use_custom_optimizer = self.use_custom_optimizer
            self.model, self.map_soln = model.build(
                data, priors, nplanets, use_gp=use_gp, fixed=fixed, basis=fit_basis, chromatic=chromatic,
                masks=masks, start=start, include_mean=include_mean, include_flare=include_flare, chromatic_flare=chromatic_flare, include_bump=include_bump, chromatic_bump=chromatic_bump,
                verbose=verbose, use_custom_optimizer=use_custom_optimizer
            )
            logging.info(f"Model: {self.model}")
            pickle.dump(self.model, open(os.path.join(self.outdir, 'model.pkl'), 'wb'))
            pickle.dump(self.map_soln, open(os.path.join(self.outdir, 'map.pkl'), 'wb'))
        # for name in self.data.keys():
        #     fn = f'fit-{name}.png'
        #     self.plot(name, fn=fn)
        if plot:
            self.plot_multi(fn='fit.png')
        
    def plot(self, name, fn=None):
        data, mask, map_soln = self.data[name], self.masks[name], self.map_soln
        nplanets, use_gp, trace = self.nplanets, self.use_gp, self.trace
        include_flare = self.include_flare
        include_bump = self.include_bump
        plot.light_curve(
            data, name, map_soln, nplanets, use_gp=use_gp, trace=trace, mask=mask, 
            include_flare=include_flare, include_bump=include_bump,
            pl_letters=self.fit_params['planets']
        )
        if fn is None:
            fn = f'fit-{name}.png'
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.savefig(os.path.join(self.outdir, fn))

    def get_ic(self, ic='BIC', verbose=False):
        soln, max_logp = util.get_map_soln(self.trace)
        nparams = sum([rv.dsize for rv in self.model.free_RVs])
        ndata = sum([len(v['x']) for v in self.data.values()])
        return util.compute_ic(soln, max_logp, nparams, ndata, method=ic, verbose=verbose)
        
    def plot_systematics(self, name, style=2, fn=None):

        fig = plot.systematics(self, name, style=style)
        if fig is not None and fn is not None:
            plt.savefig(os.path.join(self.outdir, fn), dpi=200, bbox_inches='tight')

    def plot_multi(self, keys=None, figsize=None, despine=True, noticks=True, fn=None):
        if keys is None:
            keys = self.data.keys()
        if figsize is None:
            nds = len(keys)
            figsize = (2*nds,4)

        ncols = len(keys)
        nrows = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex='col', sharey='row')
        if ncols == 1:
            axes = axes[:,None]
        for i,name in enumerate(keys):
            data, mask, map_soln = self.data[name], self.masks[name], self.map_soln
            nplanets, use_gp, trace = self.nplanets, self.use_gp, self.trace
            include_flare = self.include_flare
            include_bump = self.include_bump
            plot.light_curve(
                data, name, map_soln, nplanets, axes=axes[:,i], use_gp=use_gp, trace=trace, mask=mask, include_flare=include_flare, include_bump=include_bump,
                pl_letters=self.fit_params['planets'], 
            )
            if i > 0:
                plt.setp(axes[:,i], ylabel=None)

        if despine:
            [plt.setp(ax.spines.right, visible=False) for ax in axes.flat]
            [plt.setp(ax.spines.top, visible=False) for ax in axes.flat]
        if noticks:
            [ax.tick_params(length=0) for ax in axes.flat]

        fig.subplots_adjust(hspace=0.1, wspace=0.15)
        fig.align_ylabels()
        if fn is not None:
            plt.savefig(os.path.join(self.outdir, fn), dpi=300, bbox_inches='tight')

    def clip_outliers(self, fn=None):
        clipped = False
        include_flare = self.include_flare
        include_bump = self.include_bump
        for name, data in self.data.items():
            if self.fit_params['data'][name].get('clip', False):
                if self.clobber or self.masks[name] is None:
                    x, y = [data.get(i) for i in 'x y'.split()]
                    map_soln, use_gp = self.map_soln, self.use_gp,
                    clip_nsig = self.fit_params['data'][name].get('clip_nsig', 7)
                    if fn is None:
                        current_fn = f'{name}-outliers.png'
                    else:
                        current_fn = fn
                    fp = os.path.join(self.outdir, current_fn)
                    self.masks[name] = util.get_outlier_mask(
                        x, y, name, map_soln, use_gp,
                        nsig=clip_nsig, include_flare=include_flare, include_bump=include_bump, fp=fp
                        )
                    n_outliers = self.masks[name].size - self.masks[name].sum()
                    if n_outliers > 0:
                        logging.info(f'clipped {n_outliers} outlier(s)')
                        clipped = True
        pickle.dump(self.masks, open(os.path.join(self.outdir, 'mask.pkl'), 'wb'))
        if clipped:
            self.build_model(start=self.map_soln, force=True)
            
    def sample(self, fn=None, plot_fit=True, plot_systematics=True):

        if self.clobber or self.trace is None:
            tune = self.tune
            draws = self.draws
            chains = self.chains
            cores = self.cores
            logging.info(f'sampling for {tune} tuning steps and {draws} draws with {chains} chains on {cores} cores')
            self.trace = model.sample(
                self.model, 
                self.map_soln,
                tune=tune,
                draws=draws,
                chains=chains,
                cores=cores
            )
            pickle.dump(self.trace, open(os.path.join(self.outdir, 'trace.pkl'), 'wb'))

        with self.model:
            self.summary = util.get_summary(
                self.trace, self.data, self.bands, self.fit_basis, self.use_gp, self.fixed,
                chromatic=self.chromatic
            )
            logging.info(f'r_hat max: {self.summary["r_hat"].max()}')
            
        self.summary.to_csv(os.path.join(self.outdir, 'summary.csv'))

        soln, logp = util.get_map_soln(self.trace)
        logging.info(f"Max. log probability after sampling: {logp:.2f}")
        self.map_soln = soln
        pickle.dump(self.map_soln, open(os.path.join(self.outdir, 'map.pkl'), 'wb'))
            
        if plot_fit:
            self.plot_multi(fn='fit.png')
            if self.chromatic:
                fig = plot.plot_chromatic_ror(self.trace, self.bands, nplanets=self.nplanets, planets=self.planets)
                fig.savefig(os.path.join(self.outdir, 'chromatic_ror.png'), dpi=300, bbox_inches='tight')
            self.plot_limb_darkening()
        if plot_systematics:
            for name in self.data.keys():
                self.plot_systematics(name, fn=f'sys-{name}.png')

        for name, data in self.data.items():
            y = data['y']
            mask = self.masks[name]
            map_soln = self.map_soln
            use_gp = self.use_gp
            resid = util.get_residuals(name, y, map_soln, mask=mask, use_gp=use_gp)
            logging.info(f"{name} residual scatter: {resid.std()*1e3 :.0f} ppm")
        
    def plot_corner(self, sigma_lc=True, include_flare=True, include_bump=True, fn=None, subset=None):
        """
        Generate a corner plot of model parameters.
        
        Args:
            sigma_lc: Include log-sigma light curve parameters
            include_flare: Include flare parameters (if model has flares)
            include_bump: Include bump parameters (if model has bumps)  
            fn: Output filename (default: 'corner.png')
            subset: List of specific parameter names to plot (e.g., ['flare_tpeak', 'flare_fwhm'])
                   If provided, creates a corner plot for only these parameters
        
        Example usage:
            # Regular corner plot (all parameters)
            fit.plot_corner()
            
            # Subset: only flare parameters
            fit.plot_corner(subset=['flare_tpeak', 'flare_fwhm', 'flare_ampl_g', 'flare_ampl_r'])
            
            # Subset: only transit parameters  
            fit.plot_corner(subset=['t0', 'ror', 'b', 'dur'])
        """
        logging.info('generating corner plot')
        fig = plot.corner(
            self.trace,
            self.map_soln,
            self.priors,
            self.use_gp,
            self.fixed,
            self.nplanets,
            self.bands,
            self.data,
            self.chromatic,
            sigma_lc=sigma_lc,
            include_flare=include_flare&self.include_flare,
            chromatic_flare=self.chromatic_flare,
            include_bump=include_bump&self.include_bump,
            chromatic_bump=self.chromatic_bump,
            subset=subset
        )
        if fn is None:
            fn = 'corner.png'
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.02, wspace=0.02)
        plt.savefig(os.path.join(self.outdir, fn))

    def plot_trace(self, fn=None):
        
        print('generating trace plot')
        var_names = util.get_var_names(
            self.data, self.bands, self.fit_basis, self.use_gp, self.fixed, self.chromatic
        )
        with self.model:
            az.plot_trace(self.trace, var_names=var_names)
        if fn is None:
            fn = 'trace.png'
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, fn))

    def plot_limb_darkening(self, fn=None, corner=False, show_profile=True, show_disk=True):
        if 'u_star' in self.fixed:
            print('Limb darkening parameters are fixed - skipping plot')
            return
        
        if corner:
            print('generating limb darkening corner plot')
            fig = plot.limb_darkening_corner(self.trace, self.map_soln, self.priors, self.bands)
            if fn is None:
                fn = 'corner-limb_darkening.png'
        else:
            print('generating limb darkening plot')
            fig = plot.limb_darkening(self.trace, self.priors, self.bands, 
                                    show_profile=show_profile, show_disk=show_disk, 
                                    map_soln=self.map_soln)
            if fn is None:
                suffix = ''
                if show_profile:
                    suffix += '_with_profile'
                if show_disk:
                    suffix += '_with_disk'
                fn = f'limb_darkening{suffix}.png'
        plt.savefig(os.path.join(self.outdir, fn), dpi=200)
        
    def save_posterior_samples(self, filename='posterior_samples.csv.gz'):
        """
        Save posterior samples to a compressed CSV file.
        
        Parameters:
        -----------
        filename : str
            Name of the output file (default: 'posterior_samples.csv.gz')
        """
        print('saving posterior samples to CSV.gz')
        
        # Extract flattened samples from trace
        flat_samps = self.trace.posterior.stack(sample=("chain", "draw"))
        
        # Get sample stats (log probability, etc.)
        flat_stats = self.trace.sample_stats.stack(sample=("chain", "draw"))
        
        # Get number of samples
        n_samples = len(flat_samps.coords['sample'])
        
        # Initialize data dictionary
        data_dict = {}
        
        # Add chain and draw indices
        data_dict['chain'] = flat_samps.coords['chain'].values
        data_dict['draw'] = flat_samps.coords['draw'].values
        
        # Add log probability
        if 'lp' in flat_stats.data_vars:
            data_dict['log_probability'] = flat_stats['lp'].values
        
        # Add all posterior samples
        for var_name, var_data in flat_samps.data_vars.items():
            values = var_data.values
            
            # Skip large arrays that are not typically needed for analysis
            # (light curves, linear model predictions, etc.)
            if var_name.endswith(('_light_curves', '_light_curves_hr', '_lc_pred', '_lm')):
                continue
            
            # Handle different array shapes
            if values.ndim == 1:
                # 1D array: (n_samples,)
                data_dict[var_name] = values
            elif values.ndim == 2:
                # 2D array: could be (n_params, n_samples) or (n_samples, n_params)
                if values.shape[0] == n_samples:
                    # Shape is (n_samples, n_params)
                    for i in range(values.shape[1]):
                        data_dict[f'{var_name}_{i}'] = values[:, i]
                elif values.shape[1] == n_samples:
                    # Shape is (n_params, n_samples) - transpose needed
                    for i in range(values.shape[0]):
                        data_dict[f'{var_name}_{i}'] = values[i, :]
                else:
                    print(f'Warning: Skipping variable {var_name} with unexpected shape {values.shape}')
                    continue
            elif values.ndim == 3:
                # 3D array: typically (n_data, n_params, n_samples)
                if values.shape[2] == n_samples:
                    # Skip 3D arrays as they're typically large data arrays
                    print(f'Skipping 3D variable {var_name} (shape: {values.shape})')
                    continue
                else:
                    print(f'Warning: Skipping variable {var_name} with unexpected shape {values.shape}')
                    continue
            else:
                print(f'Warning: Skipping variable {var_name} with {values.ndim}D shape {values.shape}')
                continue
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Save to compressed CSV
        output_path = os.path.join(self.outdir, filename)
        df.to_csv(output_path, index=False, compression='gzip')
        
        print(f'Posterior samples saved to: {output_path}')
        print(f'Shape: {df.shape[0]} samples × {df.shape[1]} parameters')
        
        return output_path

    def save_results(self):
        print('saving results')
        flat_samps = self.trace.posterior.stack(sample=("chain", "draw"))
        t0_s = flat_samps['t0'].values
        with open(os.path.join(self.outdir, 'tc.txt'), 'w') as f:
            if self.nplanets > 1:
                for i in range(self.nplanets):
                    f.write(f'{self.planets[i]} {t0_s[i,:].mean() + self.ref_time - 2454833} {t0_s[i,:].std()}\n')
            else:
                f.write(f'{self.planets[0]} {t0_s.mean() + self.ref_time - 2454833} {t0_s.std()}\n')
        with open(os.path.join(self.outdir, 'ic.txt'), 'w') as f:
            soln, max_logp = util.get_map_soln(self.trace)
            nparams = sum([rv.size.eval() for rv in self.model.free_RVs])
            ndata = sum([len(v['x']) for v in self.data.values()])
            ics = 'BIC AIC AICc'.split()
            for ic in ics:
                val = util.compute_ic(soln, max_logp, nparams, ndata, method=ic, verbose=False)
                f.write(f'{ic} {val:.2f}\n')
        if self.clobber:
            pass
        self.save_posterior_samples()
        self.save_corrected()

    def save_corrected(self, subtract_tc=False):
        print('saving corrected light curves')
        soln = self.map_soln
        nplanets = self.nplanets
        for i,(name,data) in enumerate(self.data.items()):
            mask = self.masks[name]
            cor = util.get_corrected(data, name, soln, nplanets, mask=mask, subtract_tc=subtract_tc)
            x = cor['x'] + self.ref_time
            y = cor['y'] * 1e-3
            yerr = cor['yerr'] * 1e-3
            y += 1
            prefix = os.path.basename(self.wd)
            fn = f'{prefix}-{name}-cor.csv'
            fp = os.path.join(self.outdir,fn)
            pd.DataFrame(dict(x=x,y=y,yerr=yerr)).to_csv(fp, index=False)
            print(f'created file: {fp}')


def setup_logging(outdir, verbose=False):
    """Configure logging to file and optionally to console."""
    # Create log file path
    log_file = os.path.join(outdir, 'timer-fit.log')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler - always log everything to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler - only if verbose
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return log_file


def cli():
    """Command-line interface for timer transit fitting."""
    import time
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 150

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Timer transit fitting tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  timer-fit examples/toi2123
  timer-fit examples/250801-muscat3 -v
  timer-fit examples/v1298tau-m2 --verbose
  timer-fit examples/toi2123 -o model1
  timer-fit examples/toi2123 --outdir model2

The working directory must contain both 'fit.yaml' and 'sys.yaml' files.
        """
    )
    
    parser.add_argument(
        'working_directory',
        help='Directory containing fit.yaml, sys.yaml, and data files'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output to console (default: minimal console output)'
    )
    
    parser.add_argument(
        '-o', '--outdir',
        default='out',
        help='Output directory name (default: out)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    wd = args.working_directory
    verbose = args.verbose
    outdir = args.outdir
    
    # Check if working directory exists
    if not os.path.isdir(wd):
        print(f"Error: Working directory '{wd}' does not exist.")
        return 1
    
    # Check for fit.yaml
    fit_yaml_path = os.path.join(wd, 'fit.yaml')
    if not os.path.isfile(fit_yaml_path):
        print(f"Error: fit.yaml not found in '{wd}'")
        return 1
    
    # Check for sys.yaml
    sys_yaml_path = os.path.join(wd, 'sys.yaml')
    if not os.path.isfile(sys_yaml_path):
        print(f"Error: sys.yaml not found in '{wd}'")
        print("Both fit.yaml and sys.yaml are required in the working directory.")
        return 1
    
    # Create output directory early to set up logging
    outdir_path = os.path.join(wd, outdir)
    if not os.path.exists(outdir_path):
        os.makedirs(outdir_path, exist_ok=True)
    
    # Set up logging
    log_file = setup_logging(outdir_path, verbose=verbose)
    
    # Log startup information
    logging.info(f"Timer-fit started")
    logging.info(f"Working directory: {wd}")
    logging.info(f"Output directory: {outdir_path}")
    logging.info(f"Verbose mode: {verbose}")
    logging.info(f"Log file: {log_file}")
    
    # Minimal console output unless verbose
    if not verbose:
        print(f"Timer-fit starting (log: {log_file})")
    
    tick = time.time()
    
    # Load configuration files
    try:
        fit_params = yaml.load(open(fit_yaml_path), Loader=yaml.FullLoader)
        logging.info("Loaded fit.yaml successfully")
    except Exception as e:
        error_msg = f"Error loading fit.yaml: {e}"
        logging.error(error_msg)
        print(error_msg)
        return 1

    try:
        sys_params = yaml.load(open(sys_yaml_path), Loader=yaml.FullLoader)
        logging.info("Loaded sys.yaml successfully")
    except Exception as e:
        error_msg = f"Error loading sys.yaml: {e}"
        logging.error(error_msg)
        print(error_msg)
        return 1

    try:
        logging.info("Initializing TransitFit")
        fit = TransitFit(sys_params, fit_params, wd=wd, outdir=outdir)
    except Exception as e:
        error_msg = f"Error initializing TransitFit: {e}"
        logging.error(error_msg, exc_info=True)
        print(error_msg)
        return 1

    try:
        logging.info("Plotting data")
        fit.plot_data()
    except Exception as e:
        error_msg = f"Error plotting data: {e}"
        logging.error(error_msg, exc_info=True)
        print(error_msg)
        return 1

    try:
        logging.info("Building model")
        fit.build_model(verbose=True)
    except Exception as e:
        error_msg = f"Error building model: {e}"
        logging.error(error_msg, exc_info=True)
        print(error_msg)
        return 1

    try:
        logging.info("Clipping outliers")
        fit.clip_outliers()
    except Exception as e:
        error_msg = f"Error clipping outliers: {e}"
        logging.error(error_msg, exc_info=True)
        print(error_msg)
        return 1

    try:
        logging.info("Sampling")
        fit.sample()
    except Exception as e:
        error_msg = f"Error during sampling: {e}"
        logging.error(error_msg, exc_info=True)
        print(error_msg)
        return 1

    try:
        logging.info("Generating corner plot")
        fit.plot_corner()
    except Exception as e:
        error_msg = f"Error generating corner plot: {e}"
        logging.error(error_msg, exc_info=True)
        print(f"Warning: {error_msg}")
        # Don't return 1 here - continue with other plots

    try:
        logging.info("Generating trace plot")
        fit.plot_trace()
    except Exception as e:
        error_msg = f"Error generating trace plot: {e}"
        logging.error(error_msg, exc_info=True)
        print(f"Warning: {error_msg}")
        # Don't return 1 here - continue with saving results

    try:
        logging.info("Saving results")
        fit.save_results()
    except Exception as e:
        error_msg = f"Error saving results: {e}"
        logging.error(error_msg, exc_info=True)
        print(error_msg)
        return 1

    elapsed = time.time() - tick
    success_msg = f'Timer-fit completed successfully in {elapsed:.0f} seconds'
    logging.info(success_msg)

    if not verbose:
        print(success_msg)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(cli())
