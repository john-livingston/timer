# timer

**T**rans**i**t ti**m**ing **e**stimato**r** -- a Python package for robust, efficient, and flexible Bayesian analysis of individual exoplanet transit events.

The primary goal of `timer` is to precisely measure transit mid-times from one or more light curves of the same event, with full posterior uncertainties via MCMC. It is designed for the common workflow of fitting a single transit epoch covered by one or more simultaneous datasets (e.g., multi-band photometry from MuSCAT), while incorporating prior knowledge of the system parameters.

Because all key transit parameters can be assigned either Gaussian or uniform priors, `timer` is also useful for broader transit analyses beyond timing -- for example, measuring chromatic radius ratios across multiple photometric bands to validate planet candidates or detect atmospheric features (e.g., Na absorption with narrowband spectrophotometry), or characterizing spot-crossing and flare events that overlap with transits.

## Features

- Simultaneous fitting of multiple light curves of the same transit event
- Chromatic (per-band) radius ratio fitting for planet validation and spectrophotometry
- Stellar flare modeling (Davenport et al. 2014 analytic profile) with support for multiple flares
- Spot-crossing (Gaussian bump) modeling, including chromatic amplitudes
- Flexible detrending: polynomial trends, B-splines, and arbitrary covariates (e.g., airmass, centroids)
- Theoretical quadratic limb darkening priors via Claret tables
- MCMC posterior sampling via PyMC with automated MAP initialization
- Sigma-clipping of outliers with automatic re-fitting
- Publication-quality outputs: multi-panel light curve fits, corner plots, trace plots, limb darkening diagnostics

## Quick start

```bash
git clone git@github.com:john-livingston/timer.git
cd timer
pip install -e .
pip install git+https://github.com/john-livingston/limbdark

timer-fit examples/hip67522b
```

The working directory must contain both `sys.yaml` (system parameters) and `fit.yaml` (fit configuration) files.
