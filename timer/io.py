import os
import numpy as np
import pandas as pd

from .util import bin_df, get_spline_basis


def read_generic(
    fp, 
    binsize=1/1440,
    timeoffset=0,
    spline=False,
    spline_knots=5,
    add_bias=False,
    quad=False,
    trend=None,
    trim_beg=None,
    trim_end=None,
    subtract_reftime=True,
    chunk_offset=False,
    chunk_thresh=0.02,
    timecol='time',
    fluxcol='flux',
    errcol='fluxerr',
    verbose=True
):

    # READ DATA
    if fp.endswith('.txt'):
        ncols = len(open(fp).readline().split())
        names = 'time flux fluxerr'.split() + [f'c{i}' for i in range(ncols-3)]
        df = pd.read_csv(fp, sep=r'\s+', names=names)
    elif fp.endswith('.csv'):
        df = pd.read_csv(fp)
    else:
        raise ValueError("file type not recognized, must be .txt or .csv")

    if verbose:
        print(f'\nreading: {os.path.basename(fp)}')
        print(f'cadence: {np.median(np.diff(df[timecol].values))*86400 :.1f} seconds')
    if binsize is not None:
        df = bin_df(df, timecol, errcol, binsize=binsize)
    x, y, yerr = df[[timecol, fluxcol, errcol]].values.T
    aux_cols = [c for c in df.columns if c not in [timecol, fluxcol, errcol]]
    X = df[aux_cols].values if len(aux_cols) > 0 else None

    # PROCESS TIME AND FLUX
    x += timeoffset
    yerr /= y
    y = (y / np.median(y) - 1)
    y *= 1e3
    yerr *= 1e3

    # SUBTRACT A REFERENCE TIME
    if subtract_reftime:
        ref_time = int(x.min())
#         ref_time = round(x.min())
        x -= ref_time
    else:
        ref_time = 0

    if X is not None:

        # INCLUDE QUADRATIC TERMS
        if quad:
            X = np.c_[X, X**2]

        # STANDARDIZE THE DESIGN MATRIX
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        # X -= X.mean(axis=0)
        # X /= X.std(axis=0)

    # ADD TREND/BIAS COLUMNS TO THE DESIGN MATRIX
    if trend is not None:
        # if trend > 0, i.e. we want a linear or higher order trend
        A = np.vander(x - 0.5*(x.min() + x.max()), trend+1)
        if not add_bias:
            # if we don't want to add a bias column, i.e. a column of ones
            A = A[:,:-1]
        if X is not None:
            X = np.c_[X, A]
        else:
            X = A
    if spline:
        if X is not None:
            X = np.c_[X, get_spline_basis(x, n_knots=spline_knots)]
        else:
            X = get_spline_basis(x, n_knots=spline_knots)
    if add_bias:
        # if trend = None but we want to add a bias column (not needed if include_mean=True in model)
        if X is not None:
            X = np.c_[X, np.ones_like(x)]
        else:
            X = np.ones_like(x)[:,None]

    # ADD CHUNK OFFSET COLUMNS TO THE DESIGN MATRIX TO ACCOUNT FOR DATA GAPS
    if chunk_offset:
        bkpts = list(np.where(np.diff(x) > chunk_thresh)[0]+1) + [x.shape[0]]
        prev_bkpt = 0
        cols = []
        for bkpt in bkpts:
            col = np.zeros_like(x)
            col[prev_bkpt:bkpt] = 1
            prev_bkpt = bkpt
            cols.append(col)
        offsets = np.column_stack(cols)
        X = np.c_[X, offsets]

    # DISCARD BAD DATA (HIGH AIRMASS) AT THE BEGINNING OF THE TIME SERIES
    if trim_beg is not None:
        ix = x > x.min() + trim_beg
        x, y, yerr, X = x[ix], y[ix], yerr[ix], X[ix]

    if trim_end is not None:
        ix = x < x.max() - trim_end
        x, y, yerr, X = x[ix], y[ix], yerr[ix], X[ix]

    # COMPUTE APPROXIMATE EXPOSURE TIME
    texp = np.median(np.diff(x))
    x_hr = np.linspace(x.min(), x.max(), 500)

    if X.shape[1] == 0:
        X = None

    return x, y, yerr, X, texp, x_hr, ref_time

def read_afphpot_csv(
    fp, 
    binsize=1/1440,
    timeoffset=0,
    spline=False,
    spline_knots=5,
    add_bias=False,
    quad=False,
    trend=None,
    trim_beg=None,
    trim_end=None,
    subtract_reftime=True,
    chunk_offset=False,
    chunk_thresh=0.02,
    timecol='time',
    fluxcol='flux',
    errcol='fluxerr',
    verbose=True
):

    return read_generic(
        fp,
        binsize=binsize,
        timeoffset=timeoffset,
        spline=spline,
        spline_knots=spline_knots,
        add_bias=add_bias,
        quad=quad,
        trend=trend,
        trim_beg=trim_beg,
        trim_end=trim_end,
        subtract_reftime=subtract_reftime,
        chunk_offset=chunk_offset,
        chunk_thresh=chunk_thresh,
        timecol='BJD_TDB',
        fluxcol='Flux',
        errcol='Err',
        verbose=verbose
    )