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
           chromatic=False, sigma_lc=True, include_flare=False, show_prior=True):

    var_names = [f't0_{i+1}' for i in range(nplanets)] if nplanets > 1 else ['t0']
    trace_ = trace['t0'].copy()
    truths = soln['t0']
    i = nplanets
    for par in 'dur period b'.split():
        if par not in fixed:
            var_names += [f'{par}_{i+1}' for i in range(nplanets)] if nplanets > 1 else [par]
            trace_ = np.c_[trace_, trace[par].copy()]
            truths = np.append(truths, soln[par])
    if 'ror' not in fixed:
        if chromatic:
            for band in bands:
                if nplanets > 1:
                    var_names += [f'ror_{band}_{i+1}' for i in range(nplanets)]
                else:
                    var_names += [f'ror_{band}']
                trace_ = np.c_[trace_, trace[f'ror_{band}'].copy()]
                truths = np.append(truths, soln[f'ror_{band}'])
        else:
            var_names += [f'ror_{i+1}' for i in range(nplanets)] if nplanets > 1 else ['ror']
            trace_ = np.c_[trace_, trace['ror'].copy()]
            truths = np.append(truths, soln['ror'])
    if sigma_lc:
        for name in data.keys():
            par = f'{name}_log_sigma_lc'
            var_names += [par]
            trace_ = np.c_[trace_, trace[par].copy()]
            truths = np.append(truths, soln[par])
    if include_flare:
        for p in 'tpeak fwhm ampl'.split():
            par = f'flare_{p}'
            var_names += [par]
            trace_ = np.c_[trace_, trace[par].copy()]
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

def light_curve(data, name, soln, nplanets, mask=None, trace=None, use_gp=False, include_flare=False,
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
        tra_mod = np.sum(soln[f"{name}_light_curves"], axis=-1)
        tra_mod_hr = np.sum(soln[f"{name}_light_curves_hr"], axis=-1)
    else:
        if f'{name}_mean' in soln.keys():
            mean = np.median(trace[f"{name}_mean"])
        else:
            mean = 0
        lcjit = np.exp(np.median(trace[f'{name}_log_sigma_lc']))
        lin_mod = np.median(trace[f'{name}_lm'], axis=0) if f'{name}_lm' in soln.keys() else np.zeros(mask.sum())
        flare_mod = np.median(trace[f'{name}_flare'], axis=0) if include_flare else 0
        tra_mod = np.sum(np.median(trace[f"{name}_light_curves"], axis=0), axis=-1)
        tra_mod_hr = np.sum(np.median(trace[f"{name}_light_curves_hr"], axis=0), axis=-1)
    sys_mod = lin_mod + flare_mod + mean

    if use_gp:
        if trace is None or not median:
            gp_mod = soln[f"{name}_gp_pred"]
        else:
            gp_mod = np.median(trace[f"{name}_gp_pred"], axis=0)
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
            pred = np.percentile(trace[f"{name}_light_curves_hr"], [16, 50, 84], axis=0)
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
        annotate(ax, f"$\sigma$ = {resid.std() :.1f} ppt")

#     fig.suptitle(name)
#     axes[0].set_title(name)
    fig.subplots_adjust(hspace=0)
    return fig

def spline(fit, name, style=1):

    spline = fit.fit_params['data'][name]['spline']
    nspline = 5 if spline else 0    
    trend = fit.fit_params['data'][name]['trend']
    ntrend = trend if trend else 0
    x = fit.data[name]['x']
    X = fit.data[name]['X']
    mask = fit.masks[name]
    w = fit.map_soln[f'{name}_weights']
    covariates = not nspline == X.shape[1]
    ncovariates = X.shape[1] - nspline - ntrend

    x_ = x[mask]
    X_cov = X[mask,:ncovariates]
    X_spl = X[mask,ncovariates:(ncovariates+nspline)]
    w_cov = w[:ncovariates]
    w_spl = w[ncovariates:(ncovariates+nspline)]

    if style == 1:

        if covariates and spline:
            fig, axs = plt.subplots(2, 2, figsize=(6,6), sharex=True)
        elif covariates or spline:
            fig, axs = plt.subplots(2, 1, figsize=(3,6), sharex=True)

        def plot(axs, x, X, w, name):
            # axs[0].plot(x, X)
            for i,y in enumerate(X.T):
                axs[0].plot(x, y, label=f'w = {w[i] :.3f}')
            axs[0].legend()
            axs[1].plot(x, np.dot(X,w), color='k')
            plt.setp(axs[0], title=f'basis vectors: {name}')
            plt.setp(axs[1], title=f'linear combination: {name}')
            
        if covariates and not spline:
            plot(axs, x_, X_cov, w_cov, 'covariates')

        elif spline and not covariates:
            plot(axs, x_, X_spl, w_spl, 'spline')

        elif spline and covariates:
            plot(axs[:,0], x_, X_cov, w_cov, 'covariates')
            plot(axs[:,1], x_, X_spl, w_spl, 'spline')

        plt.setp(axs, xlabel='time', ylabel='flux')
        fig.tight_layout()
    
    elif style == 2:

        if covariates and spline:
            nax = 3
            fig, axs = plt.subplots(1, 3, figsize=(9,3), sharex=True)
        elif covariates or spline:
            nax = 1
            fig, ax = plt.subplots(1, 1, figsize=(3,3), sharex=True)

        def plot(ax, x, X, w, name):
            for i,y in enumerate(X.T):
                ax.plot(x, y, label=f'w = {w[i] :.3f}')
            ax.plot(x, np.dot(X,w), color='k', label=f'sum')
            ax.legend()
            plt.setp(ax, title=f'{name}')

        if covariates and not spline:
            plot(ax, x_, X_cov, w_cov, 'covariates')
            
        elif spline and not covariates:
            plot(ax, x_, X_spl, w_spl, 'spline')

        elif spline and covariates:
            plot(axs[0], x_, X_cov, w_cov, 'covariates')
            plot(axs[1], x_, X_spl, w_spl, 'spline')
            axs[2].plot(x_, np.dot(X_cov,w_cov)+np.dot(X_spl,w_spl), color='k')
            plt.setp(axs[2], title='sum')

        if nax == 1:
            plt.setp(ax, xlabel='time', ylabel='flux')
        else:
            plt.setp(axs, xlabel='time', ylabel='flux')
        fig.tight_layout()

    return fig