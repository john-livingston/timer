# Installation

## Requirements

- Python >= 3.8 (3.13 recommended)

## Install

```bash
git clone git@github.com:john-livingston/timer.git
cd timer
conda create -n timer python=3.13
conda activate timer
pip install -e .
```

### limbdark

The `limbdark` package must be installed manually:

```bash
pip install git+https://github.com/john-livingston/limbdark
```

## Dependencies

Installed automatically via pip:

- pymc
- exoplanet, exoplanet-core
- astropy
- numpy, pandas, matplotlib
- arviz, corner
- pyyaml, dill, patsy
