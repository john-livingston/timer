import os
import sys
import yaml
import time
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

tick = time.time()

proj_root = f'{os.environ["HOME"]}/github/timer'
sys.path.append(proj_root)

from timer.fit import TransitFit

wd = sys.argv[1]
fp = os.path.join(wd, 'fit.yaml')
fit_params = yaml.load(open(fp), Loader=yaml.FullLoader)

if len(sys.argv) > 2:
    fp = sys.argv[2]
    sys_params = yaml.load(open(fp), Loader=yaml.FullLoader)
else:
    fp = os.path.join(wd, 'sys.yaml')
    sys_params = yaml.load(open(fp), Loader=yaml.FullLoader)

fit = TransitFit(sys_params, fit_params, wd=wd)
fit.plot_data()
fit.build_model(verbose=True)
fit.clip_outliers()
fit.sample()
fit.plot_corner()
fit.plot_trace()
fit.save_results()

print(f'elapsed time: {time.time()-tick :.0f} seconds')
