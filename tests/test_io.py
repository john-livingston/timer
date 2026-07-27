import numpy as np
import pytest

from timer import fit, io


def test_read_generic_plain_file_returns_none_design_matrix(synthetic_lc):
    """A time/flux/err file with default settings has no design matrix at all.

    Dereferencing X.shape unconditionally at the end of read_generic makes this
    raise AttributeError, so the plainest possible input cannot be loaded.
    """
    x, y, yerr, X, texp, x_hr, ref_time, _ = io.read_generic(
        synthetic_lc, binsize=None, verbose=False)
    assert X is None
    assert x.shape == y.shape == yerr.shape
    assert x_hr.shape == (500,)


def test_read_generic_trend_with_bias_has_no_duplicate_constant_column(synthetic_lc):
    """np.vander's last column is already constant, so keeping it when add_bias
    also appends a column of ones makes the design matrix exactly collinear."""
    _, _, _, X, _, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, trend=1, add_bias=True, verbose=False)
    assert X is not None
    assert X.shape[1] == 2
    assert np.linalg.matrix_rank(X) == 2


def test_read_generic_trend_without_bias_has_no_constant_column(synthetic_lc):
    _, _, _, X, _, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, trend=1, add_bias=False, verbose=False)
    assert X.shape[1] == 1
    assert not np.any(X.std(axis=0) == 0)


def test_read_generic_trend_order_sets_the_column_count(synthetic_lc):
    """A quadratic trend contributes exactly its two non-constant powers,
    whether or not a bias column is also requested."""
    _, _, _, X_nobias, _, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, trend=2, add_bias=False, verbose=False)
    _, _, _, X_bias, _, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, trend=2, add_bias=True, verbose=False)
    assert X_nobias.shape[1] == 2
    assert X_bias.shape[1] == 3
    assert np.linalg.matrix_rank(X_bias) == 3


def test_read_generic_bias_only_gives_one_constant_column(synthetic_lc):
    _, _, _, X, _, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, add_bias=True, verbose=False)
    assert X.shape[1] == 1
    assert np.allclose(X[:, 0], 1.0)


def test_read_generic_chunk_offset_without_covariates(synthetic_lc):
    """Chunk offsets are appended with np.c_[X, offsets], which cannot be given
    a None X. Contiguous data is a single chunk, so this is one column of ones."""
    _, _, _, X, _, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, chunk_offset=True, chunk_thresh=0.02,
        verbose=False)
    assert X.shape == (120, 1)
    assert np.allclose(X[:, 0], 1.0)


def test_read_generic_standardizes_covariates(unit_spaced_lc):
    """Hand derived: airmass is 1.2 + 0.01*i for i in 0..9, so centering and
    dividing by the population standard deviation gives (i - 4.5)/sqrt(8.25).
    A sample standard deviation would divide by sqrt(82.5/9) instead."""
    _, _, _, X, _, _, _, _ = io.read_generic(
        unit_spaced_lc, binsize=None, verbose=False)
    expected = (np.arange(10) - 4.5) / np.sqrt(8.25)
    assert X[:, 0] == pytest.approx(expected, rel=1e-9)


def test_read_generic_with_covariates(synthetic_lc_aux):
    _, _, _, X, _, _, _, _ = io.read_generic(
        synthetic_lc_aux, binsize=None, verbose=False)
    assert X.shape[1] == 2
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X.std(axis=0), 1)


def test_read_generic_covariates_with_trend_and_bias(synthetic_lc_aux):
    _, _, _, X, _, _, _, _ = io.read_generic(
        synthetic_lc_aux, binsize=None, trend=2, add_bias=True, verbose=False)
    # 2 covariates + 2 trend powers + 1 bias
    assert X.shape[1] == 5
    assert np.linalg.matrix_rank(X) == 5


def test_chunk_offset_splits_at_a_real_gap(gapped_lc):
    """Two blocks separated by a 2 day gap, with the threshold below the gap.

    Pins both the column count and which points belong to which chunk, so an
    off-by-one in the breakpoint index cannot pass.
    """
    _, _, _, X, _, _, _, _ = io.read_generic(
        gapped_lc, binsize=None, chunk_offset=True, chunk_thresh=1.5,
        verbose=False)
    assert X.shape == (8, 2)
    assert list(X[:, 0]) == [1, 1, 1, 1, 0, 0, 0, 0]
    assert list(X[:, 1]) == [0, 0, 0, 0, 1, 1, 1, 1]


def test_chunk_offset_boundary_is_strictly_greater_than(gapped_lc):
    """The gap is exactly 2.0 days and the threshold is exactly 2.0, so the
    strict comparison keeps one chunk. Changing > to >= splits it in two."""
    _, _, _, X, _, _, _, _ = io.read_generic(
        gapped_lc, binsize=None, chunk_offset=True, chunk_thresh=2.0,
        verbose=False)
    assert X.shape == (8, 1)


@pytest.mark.parametrize('thresh', [0, 0.0, -1.0, None])
def test_read_generic_rejects_non_positive_chunk_thresh(synthetic_lc, thresh):
    """np.diff(x) > 0 is true everywhere, so a zero threshold appends an N x N
    identity to the design matrix: a perfect fit that erases the transit."""
    with pytest.raises(ValueError, match='chunk_thresh') as excinfo:
        io.read_generic(synthetic_lc, binsize=None, chunk_offset=True,
                        chunk_thresh=thresh, verbose=False)
    assert repr(thresh) in str(excinfo.value)


def test_non_positive_chunk_thresh_is_ignored_without_chunk_offset(synthetic_lc):
    """The threshold is unused unless chunk_offset asks for the columns, so the
    guard must not reject configurations it cannot affect."""
    _, _, _, X, _, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, chunk_offset=False, chunk_thresh=0,
        verbose=False)
    assert X is None


def test_chunk_offset_with_fit_default_gives_one_column(synthetic_lc):
    """The fit.yaml default has to be a real gap size. At 0 every point starts
    its own chunk, which the guard above now rejects outright."""
    _, _, _, X, _, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, chunk_offset=True,
        chunk_thresh=fit.defaults['data']['chunk_thresh'], verbose=False)
    assert X.shape == (120, 1)


def test_read_generic_reports_the_column_layout(synthetic_lc_aux):
    """plot.systematics slices the design matrix by block. Deriving the block
    sizes from the config alone mislabels the chunk offset columns as
    covariates and shifts every later slice."""
    *_, layout = io.read_generic(
        synthetic_lc_aux, binsize=None, trend=2, spline=True, spline_knots=5,
        add_bias=True, verbose=False)
    assert layout == {'covariates': 2, 'trend': 2, 'spline': 5,
                      'bias': 1, 'chunk': 0}


def test_the_column_layout_accounts_for_every_column(synthetic_lc_aux):
    _, _, _, X, _, _, _, layout = io.read_generic(
        synthetic_lc_aux, binsize=None, trend=2, spline=True, spline_knots=5,
        add_bias=True, verbose=False)
    assert sum(layout.values()) == X.shape[1]


def test_the_column_layout_counts_the_chunk_offset_columns(gapped_lc):
    _, _, _, X, _, _, _, layout = io.read_generic(
        gapped_lc, binsize=None, chunk_offset=True, chunk_thresh=1.5,
        verbose=False)
    assert layout['chunk'] == 2
    assert sum(layout.values()) == X.shape[1]


def test_the_column_layout_counts_quadratic_covariate_terms(synthetic_lc_aux):
    """quad doubles the covariate block, so a layout that reported the file's
    column count would be short by half."""
    *_, layout = io.read_generic(
        synthetic_lc_aux, binsize=None, quad=True, verbose=False)
    assert layout['covariates'] == 4


def test_split_design_slices_blocks_in_the_order_the_columns_were_appended():
    """Hand built: six columns numbered 0..5 so a misplaced boundary is
    immediately visible in the weights each block receives."""
    X = np.arange(24, dtype=float).reshape(4, 6)
    w = np.arange(6, dtype=float)
    layout = {'covariates': 2, 'trend': 1, 'spline': 0, 'bias': 1, 'chunk': 2}

    blocks = io.split_design(X, w, layout)

    assert list(blocks['covariates'][1]) == [0.0, 1.0]
    assert list(blocks['trend'][1]) == [2.0]
    assert blocks['spline'][0].shape == (4, 0)
    assert list(blocks['bias'][1]) == [3.0]
    assert list(blocks['chunk'][1]) == [4.0, 5.0]
    assert blocks['covariates'][0][:, 0] == pytest.approx(X[:, 0])
    assert blocks['trend'][0][:, 0] == pytest.approx(X[:, 2])


def test_trim_beg_drops_points_up_to_and_including_the_cut(unit_spaced_lc):
    """The comparison is strict: x > x.min() + trim_beg, so the point sitting
    exactly on the cut is dropped. A >= would keep it."""
    x, _, _, _, _, _, _, _ = io.read_generic(
        unit_spaced_lc, binsize=None, trim_beg=2.0, verbose=False)
    assert list(x) == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_trim_end_drops_points_from_the_cut_onward(unit_spaced_lc):
    """Mirror of the above at the other end: x < x.max() - trim_end."""
    x, _, _, _, _, _, _, _ = io.read_generic(
        unit_spaced_lc, binsize=None, trim_end=1.0, verbose=False)
    assert list(x) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_trim_keeps_the_design_matrix_aligned(unit_spaced_lc):
    """Dropping the X = X[ix] line leaves the design matrix at its original
    length, which only surfaces later as a shape mismatch in the likelihood."""
    x, _, _, X, _, _, _, _ = io.read_generic(
        unit_spaced_lc, binsize=None, trim_beg=2.0, trim_end=1.0, verbose=False)
    assert list(x) == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert X.shape[0] == 5
    # covariates are standardized over the whole series before trimming, so the
    # surviving rows carry (i - 4.5)/sqrt(8.25) for their original index i.
    # Keeping the first five rows instead would start at -1.567.
    assert X[0, 0] == pytest.approx((3 - 4.5) / np.sqrt(8.25), rel=1e-9)
    assert X[-1, 0] == pytest.approx((7 - 4.5) / np.sqrt(8.25), rel=1e-9)


def test_trim_that_removes_every_point_raises(unit_spaced_lc):
    """A units mistake, days for minutes say, otherwise surfaces as an opaque
    numpy zero-size reduction from np.median(np.diff(x))."""
    with pytest.raises(ValueError) as excinfo:
        io.read_generic(unit_spaced_lc, binsize=None, trim_beg=99.0, verbose=False)
    msg = str(excinfo.value)
    assert 'trim_beg' in msg and '99.0' in msg
    assert 'unit.csv' in msg


def test_trim_from_the_end_that_removes_every_point_raises(unit_spaced_lc):
    with pytest.raises(ValueError) as excinfo:
        io.read_generic(unit_spaced_lc, binsize=None, trim_end=99.0, verbose=False)
    assert 'trim_end' in str(excinfo.value)
