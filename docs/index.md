# timer

A Python package for Bayesian transit fitting analysis of exoplanet light curves.

## Features

- Multi-band simultaneous transit fitting
- Chromatic radius ratio fitting
- Stellar flare and spot-crossing (bump) modeling
- Polynomial and spline detrending
- Limb darkening with theoretical priors
- MCMC sampling via PyMC
- Automated outlier clipping
- Corner plots, trace plots, and publication-quality light curve figures

## Quick start

```bash
git clone git@github.com:john-livingston/timer.git
cd timer
pip install -e .
pip install git+https://github.com/john-livingston/limbdark

timer-fit examples/hip67522b
```

The working directory must contain both `sys.yaml` (system parameters) and `fit.yaml` (fit configuration) files.
