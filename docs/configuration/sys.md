# System parameters (sys.yaml)

The `sys.yaml` file defines the host star and planet properties used as priors.

## Example

```yaml
star:
  teff: [5675, 75]
  logg: [4.2, 0.2]
  feh: [0.0, 0.5]
  mass: [1.22, 0.05]
  radius: [1.38, 0.06]
planets:
  b:
    t0: [2458604.02371, 0.00037]
    period: [6.959503, 0.000016]
    ror: [0.0667, 0.0012]
    b: [0.134, 0.1]
    dur: [0.201, 0.001]
```

## Star

All stellar parameters are specified as `[value, uncertainty]`.

| Parameter | Description | Required |
|-----------|-------------|----------|
| `teff`    | Effective temperature [K] | Always (for limb darkening) |
| `logg`    | Surface gravity [cgs] | Always (for limb darkening) |
| `feh`     | Metallicity [dex] | Always (for limb darkening) |
| `mass`    | Stellar mass [M_sun] | Only if `fit_basis: mstar/rstar` |
| `radius`  | Stellar radius [R_sun] | Only if `fit_basis: mstar/rstar` |

!!! note
    For the default `fit_basis: duration`, `mass` and `radius` are optional.
    The three spectroscopic parameters (`teff`, `logg`, `feh`) are always required
    because they are used to compute theoretical limb darkening coefficients.

## Planets

Each planet is listed under `planets:` with a letter key (e.g., `b`, `c`).
All parameters are specified as `[value, uncertainty]`.

| Parameter | Description |
|-----------|-------------|
| `t0`      | Reference transit time [BJD] |
| `period`  | Orbital period [days] |
| `ror`     | Planet-to-star radius ratio (Rp/R*) |
| `b`       | Impact parameter |
| `dur`     | Transit duration [days] |

You can define multiple planets in the same file:

```yaml
planets:
  b:
    t0: [2458604.02371, 0.00037]
    period: [6.959503, 0.000016]
    ror: [0.0667, 0.0012]
    b: [0.134, 0.1]
    dur: [0.201, 0.001]
  c:
    t0: [2458602.5025, 0.0022]
    period: [14.334894, 0.00003]
    ror: [0.0517, 0.0042]
    b: [0.15, 0.15]
    dur: [0.2348, 0.0028]
```

Only the planets listed in `fit.yaml`'s `planets` field will be included in the fit.
