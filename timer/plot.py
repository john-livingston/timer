import numpy as np
import matplotlib.pyplot as plt

def annotate(ax, text, color='k', loc=1, bold=False, fontsize=10):
    fontweight = 'bold' if bold else 'normal'
    if loc == 1 or loc == 'upper left':
        xy = 0,1
        ha, va = "left", "top"
        xytext = 5,-5
    elif loc == 2 or loc == 'upper right':
        xy = 1,1
        ha, va = "right", "top"
        xytext = -5,-5
    elif loc == 3 or loc == 'lower right':
        xy = 1,0
        ha, va = "right", "bottom"
        xytext = -5,5
    elif loc == 4 or loc == 'lower left':
        xy = 0,0
        ha, va = "left", "bottom"
        xytext = 5,5
    ax.annotate(text, xy=xy, xycoords="axes fraction", ha=ha, va=va, xytext=xytext, textcoords="offset points",
        fontweight=fontweight, color=color, zorder=10, fontsize=fontsize)

def plot_outliers(x, resid, mask, fp=None):

    plt.figure(figsize=(10, 5))
    plt.plot(x, resid, "k", label="data")
    plt.plot(x[~mask], resid[~mask], "xr", label="outliers")
    plt.axhline(0, color="#aaaaaa", lw=1)
    plt.ylabel("residuals [ppt]")
    plt.xlabel("time [days]")
    plt.legend(fontsize=12, loc=3)
    _ = plt.xlim(x.min(), x.max())

    if fp is not None:
        plt.tight_layout()
        plt.savefig(fp)

def corner(trace, soln, priors, use_gp, fixed, nplanets, bands, data, 
           chromatic=False, sigma_lc=True, include_flare=False, include_bump=False, show_prior=True):

    var_names = [f't0_{i+1}' for i in range(nplanets)] if nplanets > 1 else ['t0']
    trace_ = trace.posterior['t0'].values.reshape(-1, nplanets)
    truths = soln['t0']
    i = nplanets
    for par in 'dur period b'.split():
        if par not in fixed:
            var_names += [f'{par}_{i+1}' for i in range(nplanets)] if nplanets > 1 else [par]
            trace_ = np.c_[trace_, trace.posterior[par].values.reshape(-1, nplanets)]
            truths = np.append(truths, soln[par])
    if 'ror' not in fixed:
        if chromatic:
            for band in bands:
                if nplanets > 1:
                    var_names += [f'ror_{band}_{i+1}' for i in range(nplanets)]
                else:
                    var_names += [f'ror_{band}']
                trace_ = np.c_[trace_, trace.posterior[f'ror_{band}'].values.reshape(-1, nplanets)]
                truths = np.append(truths, soln[f'ror_{band}'])
        else:
            var_names += [f'ror_{i+1}' for i in range(nplanets)] if nplanets > 1 else ['ror']
            trace_ = np.c_[trace_, trace.posterior['ror'].values.reshape(-1, nplanets)]
            truths = np.append(truths, soln['ror'])
    if sigma_lc:
        for name in data.keys():
            par = f'{name}_log_sigma_lc'
            var_names += [par]
            trace_ = np.c_[trace_, trace.posterior[par].values.reshape(-1, 1)]
            truths = np.append(truths, soln[par])
    if include_flare:
        for p in 'tpeak fwhm ampl'.split():
            par = f'flare_{p}'
            var_names += [par]
            trace_ = np.c_[trace_, trace.posterior[par].values.reshape(-1, 1)]
            truths = np.append(truths, soln[par])
    if include_bump:
        for p in 'tcenter width ampl'.split():
            par = f'bump_{p}'
            var_names += [par]
            trace_ = np.c_[trace_, trace.posterior[par].values.reshape(-1, 1)]
            truths = np.append(truths, soln[par])

    import corner

    ndim = len(var_names)
    figsize = (2.2*ndim,2.2*ndim)
    fig, axs = plt.subplots(ndim, ndim, figsize=figsize)

    hist_kwargs = dict(lw=1, alpha=1, density=True)
    title_kwargs = dict(fontdict=dict(fontsize=12))
    data_kwargs = dict(alpha=0.01)

    fig = corner.corner(
        trace_,
        fig=fig,
        labels=var_names,
        truths=truths,
        truth_color='dodgerblue',
        hist_kwargs=hist_kwargs,
        title_kwargs=title_kwargs,
        data_kwargs=data_kwargs,
        smooth=1,
#         smooth1d=1,
        show_titles=True,
        title_fmt='.4f'
    )

    if show_prior:

        import scipy.stats as st
        prior_kwargs = dict(lw=3, color='darkorange', zorder=-10, alpha=0.75)
        axs_diag = np.diag(axs)
        for name,ax in zip(var_names, axs_diag):
            if nplanets > 1:
                par = name.split('_')[0]
                if par not in priors.keys(): continue
                pnum = int(name.split('_')[-1])
                mu = priors[par][pnum-1]
                unc = priors[f'{par}_unc'][pnum-1]
            else:
                if 'ror' in name and chromatic:
                    par = name.split('_')[0]
                else:
                    par = name
                if par not in priors.keys(): continue
                mu = priors[par]
                unc = priors[f'{par}_unc']
            dist = priors[f'{par}_prior']
            xlim = ax.get_xlim()
            if dist == 'uniform':
                a, b = mu-unc/2,mu+unc/2
                ax.axhline(1/(b-a), **prior_kwargs)
            elif dist == 'gaussian':
#                 xi = np.linspace(mu-4*unc,mu+4*unc)
                xi = np.linspace(*ax.get_xlim())
                ax.plot(xi, st.norm.pdf(xi, loc=mu, scale=unc), **prior_kwargs)
            plt.setp(ax, xlim=xlim)

    return fig

def plot_chromatic_ror(trace, bands, nplanets=1, figsize=(6,4)):
    
    from matplotlib.patches import Rectangle

    post_vals = []
    for band in bands:
        if nplanets > 1:
            # TBD
            pass
        else:
            post_vals.append(trace.posterior[f'ror_{band}'].values.flatten())

    # Create figure and axis
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # X positions for the bands
    x_pos = np.arange(len(bands))

    # Add shaded regions for 1-sigma and 2-sigma credible intervals
    for i, vals in enumerate(post_vals):
        p2, p16, p50, p84, p98 = np.percentile(vals, [2.5, 16, 50, 84, 97.5])
        plt.plot([i - 0.25, i + 0.25], [p50, p50], '-', color='black', linewidth=2)
        # 2-sigma region (95% credible interval)
        rect_2sigma = Rectangle((i - 0.25, p2), 0.5, p98-p2, 
                               lw=0, alpha=0.15, color='gray', label='2-sigma' if i == 0 else "")
        
        # 1-sigma region (68% credible interval)
        rect_1sigma = Rectangle((i - 0.25, p16), 0.5, p84-p16, 
                               lw=0, alpha=0.3, color='gray', label='1-sigma' if i == 0 else "")
        
        # Add the rectangles to the plot
        ax.add_patch(rect_2sigma)
        ax.add_patch(rect_1sigma)

    # Add text labels with the values and uncertainties
    for i, vals in enumerate(post_vals):
        mean, sigma = np.mean(vals), np.std(vals)
        plt.text(i, np.percentile(vals, 97.5) + 0.2*sigma, f"{mean:.4f} $\\pm$ {sigma:.4f}", 
                 ha='center', va='bottom', fontsize=9)

    # Set x-axis ticks and labels
    plt.xticks(x_pos, bands, fontsize=12)
    plt.xlim(-0.5, len(bands) - 0.5)

    # Set y-axis limits with some padding
    y_min = min([np.percentile(v, 2.5) for v in post_vals])
    y_max = max([np.percentile(v, 97.5) for v in post_vals])
    y_pad = 0.2 * (y_max - y_min)
    plt.ylim(y_min - y_pad, y_max + y_pad)

    # Add labels and title
    plt.ylabel('$R_p/R_s$', fontsize=12)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt.gcf()

def light_curve(data, name, soln, nplanets, mask=None, trace=None, use_gp=False, 
    include_flare=False, include_bump=False,
    axes=None, figsize=(3,4), pl_letters='bcdefg', inferencedata=False, median=True, annotate_dict={},
    annotate_sigma=True):

    x, y, yerr, x_hr = [data.get(i) for i in 'x y yerr x_hr'.split()]
    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    data_kwargs = dict(ls='', color="gray", zorder=-1)
    colors = ["dodgerblue", "darkorange", "indianred"]
    if trace is None or not median:
        if f'{name}_mean' in soln.keys():
            mean = soln[f"{name}_mean"]
        else:
            mean = 0
        lcjit = np.exp(soln[f'{name}_log_sigma_lc'])
        lin_mod = soln[f'{name}_lm'] if f'{name}_lm' in soln.keys() else np.zeros(mask.sum())
        flare_mod = soln[f'{name}_flare'] if include_flare else 0
        bump_mod = soln[f'{name}_bump'] if include_bump else 0
        tra_mod = np.sum(soln[f"{name}_light_curves"], axis=-1)
        tra_mod_hr = np.sum(soln[f"{name}_light_curves_hr"], axis=-1)
    else:
        if f'{name}_mean' in soln.keys():
            mean = np.median(trace.posterior[f"{name}_mean"].values)
        else:
            mean = 0
        lcjit = np.exp(np.median(trace.posterior[f'{name}_log_sigma_lc'].values))
        lin_mod = np.median(trace.posterior[f'{name}_lm'].values, axis=(0, 1)) if f'{name}_lm' in soln.keys() else np.zeros(mask.sum())
        flare_mod = np.median(trace.posterior[f'{name}_flare'].values, axis=(0, 1)) if include_flare else 0
        bump_mod = np.median(trace.posterior[f'{name}_bump'].values, axis=(0, 1)) if include_bump else 0
        tra_mod = np.sum(np.median(trace.posterior[f"{name}_light_curves"].values, axis=(0, 1)), axis=-1)
        tra_mod_hr = np.sum(np.median(trace.posterior[f"{name}_light_curves_hr"].values, axis=(0, 1)), axis=-1)
    sys_mod = lin_mod + flare_mod + bump_mod + mean

    if use_gp:
        if trace is None or not median:
            gp_mod = soln[f"{name}_gp_pred"]
        else:
            gp_mod = np.median(trace.posterior[f"{name}_gp_pred"].values, axis=0)
        sys_mod += gp_mod

    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    else:
        fig = axes.flat[0].get_figure()

    ax = axes[0]
    ax.errorbar(x[mask], y[mask], yerr[mask], **data_kwargs)
    ax.errorbar(x[mask], y[mask], np.sqrt(yerr**2 + lcjit**2)[mask], alpha=0.5, **data_kwargs)
    ax.plot(x[mask], sys_mod, color=colors[1], label="systematics")
    ax.plot(x[mask], tra_mod+sys_mod, color=colors[2], label="systematics+transit")
#    ax.legend(fontsize=10)
    ax.set_ylabel("relative flux\n[ppt]")
    label = annotate_dict[name] if name in annotate_dict else name
    annotate(ax, label, bold=True)

    ax = axes[1]
    ax.errorbar(x[mask], y[mask]-sys_mod, yerr[mask], **data_kwargs)
    ax.errorbar(x[mask], y[mask]-sys_mod, np.sqrt(yerr**2 + lcjit**2)[mask], alpha=0.5, **data_kwargs)
    if trace is not None:
        if inferencedata:
            flat_samps = trace.posterior.stack(sample=("chain", "draw"))
            pred = np.percentile(flat_samps[f"{name}_lc_pred_hr"], [16, 50, 84], axis=-1)
        else:
            pred = np.percentile(trace.posterior[f"{name}_light_curves_hr"].values, [16, 50, 84], axis=(0, 1))
        ax.plot(x_hr, pred[1].sum(axis=-1), color=colors[0], label='transit')
        art = ax.fill_between(
            x_hr, pred[0].sum(axis=-1), pred[2].sum(axis=-1), color=colors[0], alpha=0.5, zorder=1
        )
        art.set_edgecolor("none")
    else:
        ax.plot(x_hr, tra_mod_hr, color=colors[0], label='transit')
    ax.set_ylabel("de-trended\n[ppt]")
#    ax.legend(fontsize=10)

    ax = axes[2]
    ax.errorbar(x[mask], y[mask]-tra_mod-sys_mod, yerr[mask], **data_kwargs)
    ax.errorbar(x[mask], y[mask]-tra_mod-sys_mod, np.sqrt(yerr**2 + lcjit**2)[mask], alpha=0.5, **data_kwargs)
    ax.axhline(0, color="#aaaaaa", lw=1)
    ax.set_ylabel("residuals\n[ppt]")
    # ax.set_xlim(x[mask].min(), x[mask].max())
    # ax.set_xlim(x[mask].min()-3/1440, x[mask].max()+3/1440)
    ax.set_xlabel(f"BJD$-${data['ref_time']}")
    resid = y[mask] - tra_mod - sys_mod
    if annotate_sigma:
        # cadence = np.median(np.diff(x)) * 86400
        # annotate(ax, f"$\sigma$ = {resid.std() :.1f} ppt / {cadence :.0f} sec")
        annotate(ax, f"$\\sigma$ = {resid.std() :.1f} ppt")

#     fig.suptitle(name)
#     axes[0].set_title(name)
    fig.subplots_adjust(hspace=0)
    return fig

def systematics(fit, name, style=1):

    trend = fit.fit_params['data'][name]['trend']
    ntrend = trend if trend else 0
    spline = fit.fit_params['data'][name]['spline']
    nspline = fit.fit_params['data'][name]['spline_knots'] if spline else 0
    bias = fit.fit_params['data'][name]['add_bias']
    nbias = 1 if bias else 0

    x = fit.data[name]['x']
    X = fit.data[name]['X']
    mask = fit.masks[name]
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    w = fit.map_soln[f'{name}_weights']
    covariates = not X.shape[1] == nspline + ntrend + nbias
    ncovariates = X.shape[1] - ntrend - nspline - nbias

    x_ = x[mask]
    X_ = X[mask]
    X_cov = X_[:,:ncovariates]
    w_cov = w[:ncovariates]
    X_tre = X_[:,ncovariates:(ncovariates+ntrend)]
    w_tre = w[ncovariates:(ncovariates+ntrend)]
    X_spl = X_[:,(ncovariates+ntrend):(ncovariates+ntrend+nspline)]
    w_spl = w[(ncovariates+ntrend):(ncovariates+ntrend+nspline)]

    if style == 1:

        ncols = sum([covariates, (trend is not None), spline])
        figsize = (3*ncols,6)
        fig, axs = plt.subplots(2, ncols, figsize=figsize, sharex=True)

        def plot(axs, x, X, w, name):
            # axs[0].plot(x, X)
            for i,y in enumerate(X.T):
                axs[0].plot(x, y, label=f'w = {w[i].item() :.3f}')
            axs[0].legend()
            axs[1].plot(x, np.dot(X,w), color='k')
            plt.setp(axs[0], title=f'basis vectors: {name}')
            plt.setp(axs[1], title=f'linear combination: {name}')
            
        if covariates and not spline and not trend:
            plot(axs, x_, X_cov, w_cov, 'covariates')

        elif spline and not covariates and not trend:
            plot(axs, x_, X_spl, w_spl, 'spline')

        elif trend and not covariates and not spline:
            plot(axs, x_, X_tre, w_tre, 'trend')

        elif covariates and spline and not trend:
            plot(axs[:,0], x_, X_cov, w_cov, 'covariates')
            plot(axs[:,1], x_, X_spl, w_spl, 'spline')

        elif covariates and trend and not spline:
            plot(axs[:,0], x_, X_cov, w_cov, 'covariates')
            plot(axs[:,1], x_, X_tre, w_tre, 'trend')

        elif trend and spline and not covariates:
            plot(axs[:,0], x_, X_tre, w_tre, 'trend')
            plot(axs[:,1], x_, X_spl, w_spl, 'spline')

        elif covariates and trend and spline:
            plot(axs[:,0], x_, X_cov, w_cov, 'covariates')
            plot(axs[:,1], x_, X_tre, w_tre, 'trend')
            plot(axs[:,2], x_, X_spl, w_spl, 'spline')

        plt.setp(axs, xlabel='time', ylabel='flux')
        fig.tight_layout()
    
    elif style == 2:

        ncols = sum([covariates, (trend is not None), spline])
        if ncols > 1:
            ncols += 1 # add col for sum
        figsize = (3*ncols,3)
        fig, axs = plt.subplots(1, ncols, figsize=figsize, sharex=True)

        def plot(ax, x, X, w, name):
            for i,y in enumerate(X.T):
                ax.plot(x, y, label=f'w = {w[i].item() :.1f}')
            plt.setp(ax, title=f'{name}')

        if covariates and not spline and not trend:
            plot(axs, x_, X_cov, w_cov, 'covariates')

        elif spline and not covariates and not trend:
            plot(axs, x_, X_spl, w_spl, 'spline')
            axs[0].plot(x_, np.dot(X_spl, w_spl), color='k', label=f'sum')

        elif trend and not covariates and not spline:
            plot(axs, x_, X_tre, w_tre, 'trend')

        elif covariates and spline and not trend:
            plot(axs[0], x_, X_cov, w_cov, 'covariates')
            plot(axs[1], x_, X_spl, w_spl, 'spline')
            axs[0].plot(x_, np.dot(X_cov, w_cov), color='k', label=f'sum')
            axs[1].plot(x_, np.dot(X_spl, w_spl), color='k', label=f'sum')

        elif covariates and trend and not spline:
            plot(axs[0], x_, X_cov, w_cov, 'covariates')
            plot(axs[1], x_, X_tre, w_tre, 'trend')
            axs[0].plot(x_, np.dot(X_cov, w_cov), color='k', label=f'sum')

        elif trend and spline and not covariates:
            plot(axs[0], x_, X_tre, w_tre, 'trend')
            plot(axs[1], x_, X_spl, w_spl, 'spline')
            axs[1].plot(x_, np.dot(X_spl, w_spl), color='k', label=f'sum')

        elif covariates and trend and spline:
            plot(axs[0], x_, X_cov, w_cov, 'covariates')
            plot(axs[1], x_, X_tre, w_tre, 'trend')
            plot(axs[2], x_, X_spl, w_spl, 'spline')
            axs[0].plot(x_, np.dot(X_cov, w_cov), color='k', label=f'sum')
            axs[2].plot(x_, np.dot(X_spl, w_spl), color='k', label=f'sum')

        for ax in axs[:-1]:
            ax.legend(fontsize=8)

        # if ncols > 1:
        #     axs[-1].plot(x_, np.dot(X,w), color='k')
        #     plt.setp(axs[-1], title='sum')
        if ncols > 1:
            axs[-1].plot(x_, np.dot(X_, w), color='k')
            plt.setp(axs[-1], title='sum')

        plt.setp(axs, xlabel='time', ylabel='flux')
        fig.tight_layout()

    return fig

def limb_darkening(trace, priors, bands, show_profile=False, show_disk=False, map_soln=None):
    import arviz as az
    import scipy.stats as st
    
    samples = az.extract(trace.posterior)
    
    n_bands = len(bands)
    n_rows = 2
    if show_profile:
        n_rows += 1
    if show_disk:
        n_rows += 1
    
    figsize = (3 * n_bands, 2.5 * n_rows)
    fig, axs = plt.subplots(n_rows, n_bands, figsize=figsize)
    
    # Handle case with single band
    if n_bands == 1:
        axs = axs.reshape(n_rows, 1)
    
    for i, k in enumerate(bands):
        u_s = samples[f'u_star_{k}'].values.T
        for j in range(2):
            ax = axs[j, i]
            ax.hist(u_s[:, j], histtype='step', density=True, bins=30)
            
            # Add MAP value as vertical line if available
            if map_soln is not None:
                u_map = map_soln[f'u_star_{k}'].flatten()
                ax.axvline(u_map[j], color='dodgerblue', lw=2, alpha=0.8, label='MAP' if j == 0 and i == 0 else '')
            
            # Handle single vs multi-band priors
            if isinstance(priors['u_star'][k], (tuple, list, np.ndarray)):
                # Multi-band case
                mu, unc = priors['u_star'][k][j], priors['u_star_unc'][k][j]
            else:
                # Single-band case
                mu, unc = priors['u_star'][k], priors['u_star_unc'][k]
            
            # Plot prior based on type
            dist = priors['u_star_prior']
            if dist == 'uniform':
                a, b = mu - unc/2, mu + unc/2
                ax.axhline(1/(b-a), color='darkorange', lw=3, alpha=0.75, label='prior' if j == 0 and i == 0 else '')
            elif dist == 'gaussian':
                xi = np.linspace(mu - 4 * unc, mu + 4 * unc)
                ax.plot(xi, st.norm.pdf(xi, mu, unc), color='darkorange', lw=3, alpha=0.75, label='prior' if j == 0 and i == 0 else '')
                
            ax.set_xlabel(['u1', 'u2'][j])
            ax.yaxis.set_visible(False)
            if j == 0:
                ax.set_title(k)
                # Add legend only to the first subplot
                if i == 0 and (map_soln is not None or dist in ['uniform', 'gaussian']):
                    ax.legend(fontsize=8, loc='upper left', frameon=False)
        
        # Get best-fit parameters for profile and disk plots
        if map_soln is not None:
            u_map = map_soln[f'u_star_{k}'].flatten()
            u1_best, u2_best = u_map[0], u_map[1]
            label = 'MAP'
        else:
            u1_best, u2_best = np.median(u_s, axis=0)
            label = 'median'
        
        current_row = 2
        
        # Add limb darkening profile plot if requested
        if show_profile:
            ax = axs[current_row, i]
            
            # Create mu grid (cosine of angle from disk center)
            mu = np.linspace(0, 1, 100)
            
            # Plot sample of intensity profiles
            n_samples = min(100, u_s.shape[0])
            indices = np.random.choice(u_s.shape[0], n_samples, replace=False)
            
            for idx in indices:
                u1, u2 = u_s[idx, 0], u_s[idx, 1]
                intensity = 1 - u1 * (1 - mu) - u2 * (1 - mu)**2
                ax.plot(mu, intensity, 'k-', alpha=0.05, lw=0.5)
                
            intensity_best = 1 - u1_best * (1 - mu) - u2_best * (1 - mu)**2
            ax.plot(mu, intensity_best, 'r-', lw=2, label=label)
            
            ax.set_xlabel('μ = cos(θ)')
            ax.set_ylabel('I(μ)/I(0)')
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
            current_row += 1
        
        # Add 2D stellar disk visualization if requested
        if show_disk:
            ax = axs[current_row, i]
            
            # Create 2D grid for stellar disk
            size = 500
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            X, Y = np.meshgrid(x, y)
            
            # Distance from center
            R = np.sqrt(X**2 + Y**2)
            
            # Create circular mask (stellar disk)
            disk_mask = R <= 1.0
            
            # Calculate mu = cos(theta) = sqrt(1 - R^2) for points inside disk
            mu = np.zeros_like(R)
            mu[disk_mask] = np.sqrt(1 - R[disk_mask]**2)
            
            # Calculate limb darkening intensity
            intensity = np.zeros_like(R)
            intensity[disk_mask] = 1 - u1_best * (1 - mu[disk_mask]) - u2_best * (1 - mu[disk_mask])**2
            
            # Set outside disk to NaN for white background
            intensity[~disk_mask] = np.nan
            
            # Plot the stellar disk
            im = ax.imshow(intensity, extent=[-1, 1, -1, 1], origin='lower', 
                          cmap='inferno', vmin=0, vmax=1)
            
            # Add colorbar only for first band
            if i == 0:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('I(μ)/I(0)', fontsize=8)
                cbar.ax.tick_params(labelsize=6)
            
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
            ax.set_xlabel('x/R*')
            ax.set_ylabel('y/R*')
            ax.set_title(f'{k} disk ({label})', fontsize=8)
    
    fig.tight_layout()
    return fig

def limb_darkening_corner(trace, soln, priors, bands):
    import corner
    import scipy.stats as st
    
    # Build parameter names and trace data
    var_names = []
    trace_data = []
    truths = []
    
    for band in bands:
        # Get samples for this band
        u_samples = trace.posterior[f'u_star_{band}'].values.reshape(-1, 2)
        
        # Add parameter names
        var_names.extend([f'u1_{band}', f'u2_{band}'])
        
        # Add to trace data
        if len(trace_data) == 0:
            trace_data = u_samples
        else:
            trace_data = np.c_[trace_data, u_samples]
            
        # Add truth values from MAP solution
        u_map = soln[f'u_star_{band}'].flatten()
        truths.extend([u_map[0], u_map[1]])
    
    # Create corner plot
    ndim = len(var_names)
    figsize = (2.2*ndim, 2.2*ndim)
    fig, axs = plt.subplots(ndim, ndim, figsize=figsize)
    
    hist_kwargs = dict(lw=1, alpha=1, density=True)
    title_kwargs = dict(fontdict=dict(fontsize=12))
    data_kwargs = dict(alpha=0.01)
    
    fig = corner.corner(
        trace_data,
        fig=fig,
        labels=var_names,
        truths=truths,
        truth_color='dodgerblue',
        hist_kwargs=hist_kwargs,
        title_kwargs=title_kwargs,
        data_kwargs=data_kwargs,
        smooth=1,
        show_titles=True,
        title_fmt='.4f'
    )
    
    # Add priors
    prior_kwargs = dict(lw=3, color='darkorange', zorder=-10, alpha=0.75)
    axs_diag = np.diag(axs)
    
    for i, (name, ax) in enumerate(zip(var_names, axs_diag)):
        band = name.split('_')[1]
        coeff_idx = 0 if name.startswith('u1') else 1
        
        # Handle single vs multi-band priors
        if isinstance(priors['u_star'][band], (tuple, list, np.ndarray)):
            mu, unc = priors['u_star'][band][coeff_idx], priors['u_star_unc'][band][coeff_idx]
        else:
            mu, unc = priors['u_star'][band], priors['u_star_unc'][band]
        
        dist = priors['u_star_prior']
        xlim = ax.get_xlim()
        
        if dist == 'uniform':
            a, b = mu - unc/2, mu + unc/2
            ax.axhline(1/(b-a), **prior_kwargs)
        elif dist == 'gaussian':
            xi = np.linspace(*ax.get_xlim())
            ax.plot(xi, st.norm.pdf(xi, loc=mu, scale=unc), **prior_kwargs)
            
        plt.setp(ax, xlim=xlim)
    
    return fig
