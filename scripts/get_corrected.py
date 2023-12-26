import os
import sys
import yaml
import numpy as np
import pandas as pd

proj_root = f'{os.environ["HOME"]}/gitlab/timer'
sys.path.append(proj_root)

from timer.fit import TransitFit


def get_corrected(data, name, soln, nplanets, 
                  mask=None, trace=None, use_gp=False, median=True, subtract_tc=True):
    
    if subtract_tc:
        offset = soln['t0']
    else:
        offset = 0
        
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
        tra_mod = np.sum(soln[f"{name}_light_curves"], axis=-1)
        tra_mod_hr = np.sum(soln[f"{name}_light_curves_hr"], axis=-1)
    else:
        if f'{name}_mean' in soln.keys():
            mean = np.median(trace[f"{name}_mean"])
        else:
            mean = 0
        lcjit = np.exp(np.median(trace[f'{name}_log_sigma_lc']))
        lin_mod = np.median(trace[f'{name}_lm'], axis=0)
        tra_mod = np.sum(np.median(trace[f"{name}_light_curves"], axis=0), axis=-1)
        tra_mod_hr = np.sum(np.median(trace[f"{name}_light_curves_hr"], axis=0), axis=-1)
    
    sys_mod = lin_mod + mean
    
    cor = dict(
        x=x[mask]-offset, 
        y=y[mask]-sys_mod,
        yerr=yerr[mask], 
        x_hr=x_hr-offset, 
        tra_mod_hr=tra_mod_hr
    )
    
    return cor

wd = sys.argv[1]
fp = f'{wd}/sys.yaml'
sys_params = yaml.load(open(fp), Loader=yaml.FullLoader)
fp = f'{wd}/fit.yaml'
fit_params = yaml.load(open(fp), Loader=yaml.FullLoader)
fit_params['clobber'] = False
fit = TransitFit(sys_params, fit_params, wd=wd)
soln = fit.map_soln
nplanets = fit.nplanets
for i,(name,data) in enumerate(fit.data.items()):
    mask = fit.masks[name]
    cor = get_corrected(data, name, soln, nplanets, mask=mask, subtract_tc=False)
    x = cor['x'] + fit.ref_time
    y = cor['y'] * 1e-3
    yerr = cor['yerr'] * 1e-3
    y += 1
    prefix = os.path.basename(wd)
    fn = f'{prefix}-{name}-cor.csv'
    fp = os.path.join(wd,'out',fn)
    pd.DataFrame(dict(x=x,y=y,yerr=yerr)).to_csv(fp, index=False)
    print(f'created file: {fp}')
