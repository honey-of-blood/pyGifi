# mypy: ignore-errors
"""
pygifi.plot — Visualization utilities for Gifi method results.

Python port of Gifi/R/plot.homals.R, Gifi/R/plot.princals.R, Gifi/R/plot.morals.R
(Mair, De Leeuw, Groenen. GPL-3.0).

Functions
---------
plot_homals    : Plot object scores and category quantifications (homals result)
plot_princals  : Biplot of object scores and component loadings (princals result)
plot_morals    : Transformation and fitted-vs-residual plot (morals result)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_homals(result, dim=None, which='objectscores', ax=None, **kwargs):
    """
    Plot homals results: object scores and/or category quantifications.

    Python port of R's plot.homals().

    Parameters
    ----------
    result : dict or object with attributes
        Output from Homals.fit().
    dim : list of 2 ints, default=[0, 1]
        Indices of dimensions to plot on x and y axes.
    type : str, optional
        Type of plot. One of 'jointplot', 'objplot', 'biplot', 'screeplot', 'transplot'.
        Defaults to 'jointplot'.
    which : str or list, optional
        Deprecated mapping parameter. Kept for backwards compatibility.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If None, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if dim is None:
        dim = [0, 1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    # Handle both dict and object-style results
    def get(r, k):
        return r[k] if isinstance(r, dict) else getattr(r, k)

    plot_type = kwargs.pop('type', 'jointplot')
    # Fallback to older `which` parameter 
    if which != 'objectscores':
        plot_type = which

    if plot_type == 'screeplot':
        evals = np.asarray(get(result, 'evals'))
        ax.bar(np.arange(len(evals)) + 1, evals, color='steelblue', alpha=0.7)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Homals — Scree Plot')
        ax.set_xticks(np.arange(len(evals)) + 1)
        return ax
        
    if plot_type == 'transplot':
        # Transformation plot groups by variable
        transforms = get(result, 'transform')
        data = get(result, 'data')
        var_colors = plt.cm.Set1(np.linspace(0, 1, len(transforms)))
        for j, tr in enumerate(transforms):
            # plot first copy of transform against original numeric values
            col_data = np.asarray(get(result, 'datanum')[:, j])
            sorted_idx = np.argsort(col_data)
            ax.step(col_data[sorted_idx], np.asarray(tr)[sorted_idx, 0], 
                     where='mid', color=var_colors[j], label=data.columns[j] if hasattr(data, 'columns') else f"Var{j}", alpha=0.8)
        
        ax.set_xlabel('Original Values (quantified)')
        ax.set_ylabel('Transformed Scale')
        ax.set_title('Homals — Transformation Plot')
        ax.legend()
        return ax

    which_list = ['objectscores', 'quantifications'] if plot_type == 'jointplot' else ['objectscores'] if plot_type == 'objplot' else ['objectscores', 'loadings']

    if 'objectscores' in which_list:
        scores = np.asarray(get(result, 'objectscores'))
        ax.scatter(scores[:, dim[0]], scores[:, dim[1]],
                   color='steelblue', alpha=0.4, s=8, zorder=2, **kwargs)

    if 'quantifications' in which_list:
        quants = get(result, 'quantifications')   # list of (n_cats, ndim) arrays
        data = get(result, 'data')                # original DataFrame
        var_colors = plt.cm.Set1(np.linspace(0, 1, len(quants)))

        for j, q in enumerate(quants):
            q = np.asarray(q)
            # Get category labels from the original data column
            col = data.iloc[:, j]
            if hasattr(col, 'cat'):
                cat_labels = col.cat.categories.tolist()
            else:
                cat_labels = sorted(col.unique())

            # Handle 1D quantifications (variables with <= ndim categories)
            if q.ndim == 1:
                q = q[:, None]
            if q.shape[1] <= dim[1]:
                # Pad with zeros for missing dimensions
                padded = np.zeros((q.shape[0], dim[1] + 1))
                padded[:, :q.shape[1]] = q
                q = padded

            n_cats = q.shape[0]
            for i in range(min(n_cats, len(cat_labels))):
                ax.scatter(q[i, dim[0]], q[i, dim[1]],
                           color=var_colors[j], marker='^', s=60,
                           edgecolors='black', linewidths=0.5, zorder=5)
                ax.annotate(str(cat_labels[i]),
                            xy=(q[i, dim[0]], q[i, dim[1]]),
                            xytext=(4, 4), textcoords='offset points',
                            fontsize=7, color=var_colors[j], fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7,
                                      edgecolor='none', pad=0.3))

    if 'loadings' in which_list:
        try:
            loadings = np.asarray(get(result, 'loadings'))
            scores = np.asarray(get(result, 'objectscores'))
            scalef = np.percentile(np.abs(scores), 80) / (np.max(np.abs(loadings)) + 1e-12)
            for i in range(loadings.shape[0]):
                ax.annotate('', xy=(loadings[i, dim[0]] * scalef,
                                    loadings[i, dim[1]] * scalef),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color='coral', lw=1.5))
        except Exception:
            pass

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(f'Dimension {dim[0]+1}')
    ax.set_ylabel(f'Dimension {dim[1]+1}')
    ax.set_title('Homals — Object Scores & Quantifications')

    return ax


def plot_princals(result, dim=None, type='biplot', ax=None, **kwargs):
    """
    Plot princals results: biplot of object scores and loadings.

    Python port of R's plot.princals().

    Parameters
    ----------
    result : dict or object
        Output from Princals.fit().
    dim : list of 2 ints, default=[0, 1]
        Dimensions to plot.
    type : str
        'biplot' — overlay object scores and loadings.
        'scores' — only object scores.
        'loadings' — only component loadings (bar chart).
        'screeplot' — eigenvalue bar chart.
        'transplot' — original vs transformed data.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if dim is None:
        dim = [0, 1]

    def get(r, k):
        return r[k] if isinstance(r, dict) else getattr(r, k)

    if type == 'screeplot':
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        evals = np.asarray(get(result, 'evals'))
        ax.bar(np.arange(len(evals)) + 1, evals, color='steelblue', alpha=0.7)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Princals — Scree Plot')
        ax.set_xticks(np.arange(len(evals)) + 1)
        return ax
        
    if type == 'transplot':
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        transforms = get(result, 'transform')
        data = get(result, 'data')
        
        # If transform is a list of matrices, handle it. If it's a single matrix, split by columns.
        if isinstance(transforms, np.ndarray):
            transforms = [transforms[:, i:i+1] for i in range(transforms.shape[1])]
            
        var_colors = plt.cm.Set1(np.linspace(0, 1, len(transforms)))
        for j, tr in enumerate(transforms):
            col_data = np.asarray(get(result, 'datanum')[:, j])
            sorted_idx = np.argsort(col_data)
            ax.step(col_data[sorted_idx], np.asarray(tr)[sorted_idx, 0], 
                     where='mid', color=var_colors[j], label=data.columns[j] if hasattr(data, 'columns') else f"Var{j}", alpha=0.8)
        
        ax.set_xlabel('Original Values (quantified)')
        ax.set_ylabel('Transformed Scale')
        ax.set_title('Princals — Transformation Plot')
        ax.legend()
        return ax

    if type == 'loadings':
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        loadings = np.asarray(get(result, 'loadings'))
        n_vars = loadings.shape[0] if loadings.ndim > 1 else 1
        x_pos = np.arange(n_vars)
        ax.bar(x_pos, loadings[:, dim[0]] if loadings.ndim > 1 else loadings,
               color='steelblue', alpha=0.7, label=f'Dim {dim[0]+1}')
        if loadings.ndim > 1 and loadings.shape[1] > 1:
            ax.bar(x_pos, loadings[:, dim[1]], color='coral', alpha=0.7,
                   label=f'Dim {dim[1]+1}')
        ax.set_title('Princals — Component Loadings')
        ax.legend()
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        scores = np.asarray(get(result, 'objectscores'))
        ax.scatter(scores[:, dim[0]], scores[:, dim[1]],
                   color='steelblue', alpha=0.6, zorder=3, label='Objects', **kwargs)

        if type == 'biplot':
            try:
                loadings = np.asarray(get(result, 'loadings'))
                scalef = np.percentile(np.abs(scores), 80) / (np.max(np.abs(loadings)) + 1e-12)
                for i in range(loadings.shape[0]):
                    ax.annotate('', xy=(loadings[i, dim[0]] * scalef,
                                        loadings[i, dim[1]] * scalef),
                                xytext=(0, 0),
                                arrowprops=dict(arrowstyle='->', color='coral', lw=1.5))
            except Exception:
                pass  # Silently skip loadings overlay if not available

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel(f'Dimension {dim[0]+1}')
        ax.set_ylabel(f'Dimension {dim[1]+1}')
        ax.set_title('Princals — Biplot')

    return ax


def plot_morals(result, ncols=2):
    """Mirrors R's plot.morals: observed vs transformed per variable."""
    def get(r, k):
        return r[k] if isinstance(r, dict) else getattr(r, k)
    
    # support both Morals object and dict containing results
    data_X = get(result, 'X_') if hasattr(result, 'X_') else get(result, 'data')
    if data_X is None or not hasattr(data_X, 'columns'):
        # fallback if attributes not found
        cols = [f"X{i}" for i in range(get(result, 'n_pred_') or get(result, 'xhat').shape[1])]
        X_obs = np.asarray(data_X) if data_X is not None else np.zeros((get(result, 'xhat').shape[0], len(cols)))
    else:
        cols = list(data_X.columns)
        X_obs = data_X.values

    y_obs = get(result, 'y_')
    if y_obs is None:
        y_obs = np.zeros(len(X_obs))
    
    xhat = get(result, 'result_')['xhat'] if hasattr(result, 'result_') else get(result, 'xhat')
    yhat = get(result, 'result_')['yhat'] if hasattr(result, 'result_') else get(result, 'yhat')
    
    nplots = len(cols) + 1   # predictors + response
    nrows = int(np.ceil(nplots / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.atleast_1d(axes).flatten()
    
    # predictor transformations
    for i, col in enumerate(cols):
        x_obs_i  = X_obs[:, i]
        x_hat_i  = xhat[:, i]   # transformed predictor i
        order  = np.argsort(x_obs_i)
        axes[i].plot(x_obs_i[order], x_hat_i[order], 'k-')
        axes[i].set_xlabel('Observed'); axes[i].set_ylabel('Transformed')
        axes[i].set_title(col)
    
    # response transformation
    y_obs_arr = np.asarray(y_obs)
    order = np.argsort(y_obs_arr)
    axes[len(cols)].plot(y_obs_arr[order], yhat[order], 'k-')
    axes[len(cols)].set_xlabel('Observed'); axes[len(cols)].set_ylabel('Transformed')
    
    y_name = 'Response'
    if hasattr(y_obs, 'name') and y_obs.name:
        y_name = str(y_obs.name)
    axes[len(cols)].set_title(y_name)
    
    # hide unused panels
    for j in range(nplots, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('MORALS', fontweight='bold')
    plt.tight_layout()
    return fig
