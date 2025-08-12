import os
import sys
import yaml
import numpy as np
import pandas as pd

proj_root = f'{os.environ["HOME"]}/gitlab/timer'
sys.path.append(proj_root)

from timer.fit import TransitFit
from timer.util import get_corrected

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
