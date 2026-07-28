# Usage

## Command-line interface

```bash
timer-fit <working_directory> [options]
```

| Option | Description |
|--------|-------------|
| `-v`, `--verbose` | Enable verbose console output |
| `-o`, `--outdir` | Output directory name (default: `out`) |

### Examples

```bash
timer-fit examples/hip67522b
timer-fit examples/hip67522b -v
timer-fit examples/hip67522b -o model1
```

## Python API

```python
import yaml
from timer.fit import TransitFit

sys_params = yaml.load(open('sys.yaml'), Loader=yaml.FullLoader)
fit_params = yaml.load(open('fit.yaml'), Loader=yaml.FullLoader)

fit = TransitFit(sys_params, fit_params, wd='.')
fit.build_model()
fit.clip_outliers()
fit.sample()
fit.plot_corner()
fit.save_results()
```

### Loading saved results

```python
from timer.fit import TransitFit

fit = TransitFit.from_dir('examples/hip67522b')
```

## Pipeline

The `timer-fit` CLI runs the following steps in order:

1. **Load data** -- read light curve files, bin, detrend
2. **Build model** -- construct PyMC model with priors, optimize for MAP solution
3. **Clip outliers** -- sigma-clip residuals (if `clip: true` in data config)
4. **Re-fit** -- rebuild model with outlier mask applied
5. **Sample** -- MCMC sampling with PyMC
6. **Plot** -- light curve fits, corner plots, trace plots, limb darkening
7. **Save** -- summary statistics, transit times, posterior samples, corrected light curves

## Outputs

All outputs are saved to the `out/` directory (or custom `--outdir`):

| File | Description |
|------|-------------|
| `fit.png` | Light curve fit with residuals |
| `corner.png` | Corner plot of posterior distributions |
| `trace.png` | MCMC trace plot |
| `summary.csv` | Parameter summary statistics |
| `tc.txt` | Fitted transit center times |
| `ic.txt` | Information criteria (BIC, AIC, AICc), and for GP fits their effective degrees of freedom corrected counterparts |
| `posterior_samples.csv.gz` | Full posterior samples |
| `*-cor.csv` | Corrected (detrended) light curves |
| `timer-fit.log` | Full log file |
| `fit.yaml`, `sys.yaml` | Copies of input configuration |
| `cache.json` | Records which config and data each cached artifact was produced from |

### Reading ic.txt

The criteria are built from the maximum observed-data log likelihood over the
posterior draws, so the penalty is `nparams * log(ndata)` and nothing else. In
particular they do not include the prior terms: a criterion that did would
charge each systematics column the log density of its own weight prior, about
four times the intended penalty, and would move if that prior were widened.

Values written before this change are not comparable with values written after
it. Recompute rather than mixing them.

For GP fits, `edf` is the number of degrees of freedom the GP smoother actually
absorbs, typically far more than its two or three hyperparameters, and the
`*_edf` rows repeat each criterion with that substituted. Treat `nparams_edf`
as an upper bound: it does not subtract the overlap between the GP and the
linear systematics model, so a fit with a wide design matrix is charged for
some flexibility twice.

### Reading *-cor.csv

`yerr` is the uncertainty the likelihood used, the photometric error and the
fitted jitter added in quadrature, not the bare photometric error. Where a
light curve was binned with the default `kind='median'`, the binned error also
carries the `sqrt(pi/2)` factor that the median of a sample costs over its
mean.
