# Fit parameters (fit.yaml)

The `fit.yaml` file configures the data, model, priors, and sampler settings.

## Example

```yaml
data:
  g:
    file: observation_g.txt
    band: g
    trend: 1
    binsize: 0.00139
  i:
    file: observation_i.txt
    band: i
    trend: 1
    binsize: 0.00139
planets: b
tc_pred: 2460844.98
tc_pred_unc: 0.04
chromatic: true
uniform:
  ror: [0.01, 0.15]
  b: [0, 1]
fixed:
  - period
  - u_star
```

## Data

Each dataset is listed under `data:` with a user-chosen key. Multiple datasets can be specified for simultaneous multi-band fitting.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `file`    | Path to data file (relative to working directory) | *required* |
| `band`    | Photometric band name (e.g., `g`, `r`, `i`, `z`) | *required* |
| `format`  | File format: `generic` or `afphot` | `generic` |
| `binsize` | Bin size in days (`null` for no binning) | 5/1440 (~3.5 min) |
| `trend`   | Polynomial detrending order (`null` for none) | `null` |
| `spline`  | Use spline detrending | `false` |
| `spline_knots` | Number of spline knots | `5` |
| `add_bias` | Add a bias (constant) column to the design matrix | `false` |
| `quadratic` | Include quadratic terms for auxiliary regressors | `false` |
| `trim_beg` | Trim data within this many days of the start | `null` |
| `trim_end` | Trim data within this many days of the end | `null` |
| `clip`    | Enable sigma-clipping of outliers | `false` |
| `clip_nsig` | Sigma threshold for outlier clipping | `7` |
| `chunk_offset` | Add offset columns for data gaps | `false` |
| `chunk_thresh` | Gap threshold in days for chunk detection | `0` |

### Data formats

**`generic`** (default): A whitespace-delimited text file (`.txt`) with no header row. The first three columns must be `time`, `flux`, `flux_error`, in that order. Any additional columns are automatically used as covariates in the linear detrending model (e.g., airmass, pixel centroids, sky background). Alternatively, a `.csv` file with a header row and columns named `time`, `flux`, `fluxerr` can be used.

**`afphot`**: A CSV file with columns `BJD_TDB`, `Flux`, `Err`.

## Model

| Parameter | Description | Default |
|-----------|-------------|---------|
| `planets` | Planet letter(s) to fit (must match keys in `sys.yaml`) | *required* |
| `tc_pred` | Predicted transit center time [BJD] | estimated from data |
| `tc_pred_unc` | Uncertainty on predicted transit center [days] | `0.04` |
| `tc_pred_iso` | Alternative: predicted transit time in ISO format | - |
| `fit_basis` | Parameterization: `duration` or `mstar/rstar` | `duration` |
| `chromatic` | Fit radius ratio independently per band | `false` |
| `include_mean` | Fit a mean flux offset per dataset | `true` |
| `include_flare` | Include a stellar flare model | `false` |
| `chromatic_flare` | Fit flare amplitude independently per band | `false` |
| `include_bump` | Include a Gaussian bump model (e.g., spot crossing) | `false` |
| `chromatic_bump` | Fit bump amplitude independently per band | `false` |

## Priors

### Fixed parameters

Parameters listed under `fixed` are held at their `sys.yaml` values and not sampled:

```yaml
fixed:
  - period
  - u_star
```

Supported: `t0`, `period`, `ror`, `b`, `dur`, `m_star`, `r_star`, `u_star`.

### Uniform priors

Parameters listed under `uniform` use uniform priors instead of the default Gaussian:

```yaml
uniform:
  ror: [0.01, 0.15]
  b: [0, 1]
```

For multi-planet systems, you can specify per-planet bounds:

```yaml
uniform:
  ror: [[0.03, 0.06], [0.01, 0.04]]
```

## Flare model

When `include_flare: true`, configure the flare parameters:

```yaml
flare:
  tpeak: 2460925.05        # peak time [BJD]
  tpeak_unc: 0.04
  tpeak_prior: uniform
  fwhm: 0.02               # full width at half max [days]
  fwhm_unc: 0.04
  fwhm_prior: uniform
  ampl: 7.5                # amplitude [ppt]
  ampl_unc: 15
  ampl_prior: uniform
```

Multiple flares can be specified using lists:

```yaml
flare:
  tpeak: [2460925.05, 2460925.12]
  fwhm: 0.02               # scalar values are replicated for all flares
  ampl: 7.5
  ...
```

## Bump model

When `include_bump: true`, configure the bump (e.g., spot crossing) parameters:

```yaml
bump:
  tcenter: 2460942.08
  tcenter_unc: 0.04
  tcenter_prior: uniform
  width: 0.01
  width_unc: 0.02
  width_prior: uniform
  ampl: 1.0
  ampl_unc: 5
  ampl_prior: uniform
```

Multiple bumps work the same as multiple flares.

## Sampler

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tune`    | Number of tuning (burn-in) steps | `2000` |
| `draws`   | Number of posterior draws | `2000` |
| `chains`  | Number of MCMC chains | `2` |
| `cores`   | Number of CPU cores | `2` |
| `clobber` | Re-run even if saved results exist | `false` |
