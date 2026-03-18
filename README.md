# timer

A Python package for transit fitting analysis.

## Installation

### Option 1: Install from source (recommended)

    git clone git@gitlab.com:john-livingston/timer.git
    cd timer
    conda create -n timer python=3.13
    conda activate timer
    pip install .
    pip install git+https://github.com/john-livingston/limbdark

### Option 2: Development installation

    git clone git@gitlab.com:john-livingston/timer.git
    cd timer
    conda create -n timer python=3.13
    conda activate timer
    pip install -e .
    pip install git+https://github.com/john-livingston/limbdark

## Usage

After installation, you can use the command-line interface:

    timer-fit examples/hip67522

The working directory must contain both `fit.yaml` and `sys.yaml` files. 

## Dependencies

The package automatically installs the following dependencies:
- pyyaml
- pymc
- astropy
- patsy
- exoplanet
- exoplanet-core
- dill
- corner
- numpy
- pandas
- matplotlib
- arviz

Note: You still need to manually install `limbdark` from the GitHub repository as shown in the installation instructions above.
