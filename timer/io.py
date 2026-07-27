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

    # a non-positive threshold makes np.diff(x) > chunk_thresh true at every
    # sample, so every point becomes its own chunk and an N x N identity is
    # appended to the design matrix, which fits the data perfectly and
    # silently destroys the transit
    if chunk_offset and (chunk_thresh is None or chunk_thresh <= 0):
        raise ValueError(
            f'chunk_thresh must be > 0 when chunk_offset is enabled, got '
            f'{chunk_thresh!r}: every point would start its own offset column'
        )

    # READ DATA
    if fp.endswith('.txt'):
        with open(fp) as f:
            ncols = len(f.readline().split())
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
        # if trend > 0, i.e. we want a linear or higher order trend.
        # np.vander's last column is constant; always drop it here so the
        # add_bias block below is the single source of the bias column.
        A = np.vander(x - 0.5*(x.min() + x.max()), trend+1)[:,:-1]
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
        X = np.c_[X, offsets] if X is not None else offsets

    # DISCARD BAD DATA (HIGH AIRMASS) AT THE BEGINNING OF THE TIME SERIES
    if trim_beg is not None:
        ix = x > x.min() + trim_beg
        x, y, yerr = x[ix], y[ix], yerr[ix]
        if X is not None:
            X = X[ix]

    if trim_end is not None:
        ix = x < x.max() - trim_end
        x, y, yerr = x[ix], y[ix], yerr[ix]
        if X is not None:
            X = X[ix]

    # trimming past the end of the time series leaves nothing to fit, and the
    # reductions below then fail with an opaque numpy zero-size error far from
    # the misconfiguration that caused it
    if x.size == 0:
        raise ValueError(
            f'trim_beg={trim_beg} and trim_end={trim_end} left {x.size} points '
            f'in {os.path.basename(fp)}'
        )

    # COMPUTE APPROXIMATE EXPOSURE TIME
    texp = np.median(np.diff(x))
    x_hr = np.linspace(x.min(), x.max(), 500)

    if X is not None and X.shape[1] == 0:
        X = None

    return x, y, yerr, X, texp, x_hr, ref_time

def read_afphot(
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