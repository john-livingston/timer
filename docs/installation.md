# Installation

## Requirements

- Python >= 3.11 (3.13 recommended)

## Install

```bash
git clone https://github.com/john-livingston/timer.git
cd timer
conda create -n timer python=3.13
conda activate timer
conda install eigen
pip install -e .
```

`eigen` is needed only on Python 3.13: celerite2 is pinned below 0.3.3, and that release
publishes wheels up to 3.12 only, so on 3.13 pip compiles it from source against the Eigen
headers. On 3.11 and 3.12 a wheel is used and the `conda install` line can be skipped.

### limbdark

The `limbdark` package must be installed manually:

```bash
pip install git+https://github.com/john-livingston/limbdark
```

## Dependencies

Installed automatically via pip:

- pymc
- exoplanet, exoplanet-core
- celerite2, pinned below 0.3.3, whose pymc backend imports jax
- astropy
- numpy, pandas, matplotlib
- arviz, corner
- pyyaml, dill, patsy
