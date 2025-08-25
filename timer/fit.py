import os
import dill as pickle
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from astropy.time import Time

from . import io, util, plot, model


defaults = dict(

    model = dict(
        fixed = [], # transit model parameters (duration basis): m_star r_star t0 period ror b dur u_star
        fit_basis = 'duration',
        chromatic = False,
        include_mean = True, # the mean flux value (should not be True if add_bias=True)
        include_flare = False,
        include_bump = False,
        use_gp = False,
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
                print(f'setting default: {k} = {v}')
                self.fit_params[k] = v
        
        # set sampler defaults
        for k,v in defaults['sampler'].items():
            if k not in self.fit_params.keys():
                print(f'setting default: {k} = {v}')
                self.fit_params[k] = v

        # set data defaults
        for k,v in defaults['data'].items():
            for n in self.fit_params['data'].keys():
                if k not in self.fit_params['data'][n].keys():
                    print(f'setting default for {n}: {k} = {v}')
                    self.fit_params['data'][n][k] = v

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
        self.include_bump = fit_params['include_bump']
        self.use_gp = fit_params['use_gp']
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
            print(f'loading data: {fn}')
            print(f'data span: {data_iso[0]} - {data_iso[1]}')
            print(f'ref. time: {ref_time}')
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
                print('loading mask(s) from mask.pkl')
                self.masks = pickle.load(open(os.path.join(self.outdir, 'mask.pkl'), 'rb'))
            if os.path.exists(os.path.join(self.outdir, 'model.pkl')):
                print('loading model from model.pkl')
                self.model = pickle.load(open(os.path.join(self.outdir, 'model.pkl'), 'rb'))
            if os.path.exists(os.path.join(self.outdir, 'map.pkl')):
                print('loading MAP solution from map.pkl')
                self.map_soln = pickle.load(open(os.path.join(self.outdir, 'map.pkl'), 'rb'))
            if os.path.exists(os.path.join(self.outdir, 'trace.pkl')):
                print('loading trace from trace.pkl')
                self.trace = pickle.load(open(os.path.join(self.outdir, 'trace.pkl'), 'rb'))

    def plot_data(self):
        print("plotting data")
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
        if self.include_flare:
            # lower = min([self.data[k]['x'].min() for k in self.data.keys()])
            # upper = max([self.data[k]['x'].max() for k in self.data.keys()])
            for p in 'tpeak fwhm ampl'.split():
                self.priors[f'flare_{p}'] = self.flare[p]
                self.priors[f'flare_{p}_prior'] = self.flare[f'{p}_prior']
                self.priors[f'flare_{p}_unc'] = self.flare[f'{p}_unc']
            p = 'tpeak'
            self.priors[f'flare_{p}'] = self.flare[p] - self.ref_time
        if self.include_bump:
            for p in 'tcenter width ampl'.split():
                self.priors[f'bump_{p}'] = self.bump[p]
                self.priors[f'bump_{p}_prior'] = self.bump[f'{p}_prior']
                self.priors[f'bump_{p}_unc'] = self.bump[f'{p}_unc']
            p = 'tcenter'
            self.priors[f'bump_{p}'] = self.bump[p] - self.ref_time

    def build_model(self, start=None, force=False, verbose=False, plot=True):
        if force or self.clobber or self.model is None:
            print('building and optimizing model')
            data, priors, masks = self.data, self.priors, self.masks
            nplanets, use_gp, chromatic = self.nplanets, self.use_gp, self.chromatic
            fixed, fit_basis = self.fixed, self.fit_basis
            include_mean, include_flare, include_bump = self.include_mean, self.include_flare, self.include_bump
            self.model, self.map_soln = model.build(
                data, priors, nplanets, use_gp=use_gp, fixed=fixed, basis=fit_basis, chromatic=chromatic,
                masks=masks, start=start, include_mean=include_mean, include_flare=include_flare, include_bump=include_bump,
                verbose=verbose
            )
            print(self.model)
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
        if fn is not None:
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
                        print(f'clipped {n_outliers} outlier(s)')
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
            print(f'sampling for {tune} tuning steps and {draws} draws with {chains} chains on {cores} cores')
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
            print('r_hat max:', self.summary['r_hat'].max())
            
        self.summary.to_csv(os.path.join(self.outdir, 'summary.csv'))

        soln, logp = util.get_map_soln(self.trace)
        self.map_soln = soln
        pickle.dump(self.map_soln, open(os.path.join(self.outdir, 'map.pkl'), 'wb'))
            
        # for name in self.data.keys():
        #     fn = f'fit-{name}.png'
        #     self.plot(name, fn=fn)
        if plot_fit:
            self.plot_multi(fn='fit.png')
            if self.chromatic:
                fig = plot.plot_chromatic_ror(self.trace, self.bands, nplanets=self.nplanets)
                fig.savefig(os.path.join(self.outdir, 'chromatic_ror.png'), dpi=300, bbox_inches='tight')
        if plot_systematics:
            for name in self.data.keys():
                self.plot_systematics(name, fn=f'sys-{name}.png')

        for name, data in self.data.items():
            y = data['y']
            mask = self.masks[name]
            map_soln = self.map_soln
            use_gp = self.use_gp
            resid = util.get_residuals(name, y, map_soln, mask=mask, use_gp=use_gp)
            print(f"{name} residual scatter: {resid.std()*1e3 :.0f} ppm")
        
    def plot_corner(self, sigma_lc=True, include_flare=True, include_bump=True, fn=None):

        print('generating corner plot')
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
            include_bump=include_bump&self.include_bump
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
        
    def save_results(self):
        print('saving results')
        flat_samps = self.trace.posterior.stack(sample=("chain", "draw"))
        t0_s = flat_samps['t0'].values
        with open(os.path.join(self.outdir, 'tc.txt'), 'w') as f:
            if self.nplanets > 1:
                for i in range(self.nplanets):
                    f.write(f'{self.planets[i]} {t0_s[:,i].mean() + self.ref_time - 2454833} {t0_s[:,i].std()}\n')
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
