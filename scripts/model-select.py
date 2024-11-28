import os
import sys
import yaml
import time
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150

tick = time.time()

#proj_root = f'{os.environ["HOME"]}/github/timer'
proj_root = f'{os.environ["HOME"]}/gitlab/timer'
sys.path.append(proj_root)

from timer.fit import TransitFit

wd = sys.argv[1]
fp = os.path.join(wd, 'fit.yaml')
fit_params = yaml.load(open(fp), Loader=yaml.FullLoader)

if len(sys.argv) > 2:
    fp = sys.argv[2]
    sys_params = yaml.load(open(fp), Loader=yaml.FullLoader)
elif os.path.isfile('sys.yaml'):
    sys_params = yaml.load(open('sys.yaml'), Loader=yaml.FullLoader)
else:
    fp = os.path.join(wd, 'sys.yaml')
    sys_params = yaml.load(open(fp), Loader=yaml.FullLoader)

models = dict(
    trend1 = dict(
        main = dict(chromatic=False), 
        data = dict(
            all=dict(trend=1, spline=False)
        )
    ),
    spline = dict(
        main = dict(chromatic=False), 
        data = dict(
            all = dict(trend=0, spline=True)
        )
    ),
    chromatic = dict(
        main = dict(chromatic=True), 
        data = dict(
            all = dict(trend=1, spline=False)
        )
    )
)
bics = {}
for name, model in models.items():
    for ds in fit_params['data'].keys():
        if 'all' in model['data'].keys():
            fit_params['data'][ds].update(model['data']['all'])
        elif ds in model['data'].keys():
            fit_params['data'][ds].update(model['data'][ds])
        # print(fit_params['data'][ds])
    fit_params.update(model['main'])
    fit = TransitFit(sys_params, fit_params, wd=wd, outdir=f'model-{name}')
    fit.build_model(verbose=False)
    # fit.clip_outliers()
    fit.plot_multi(fn=f'fit.png')
    bics[name] = fit.get_ic(verbose=False)

print(pd.DataFrame(bics, index=['BIC']).transpose().sort_values(by='BIC'))
print(f'elapsed time: {time.time()-tick :.0f} seconds')
