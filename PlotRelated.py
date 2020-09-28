'''
Available methods are the followings:
[1] matplotlib_cmap
[2] create_cmap
[3] plot_pie
[4] plot_radar
[5] plot_pcs

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 26-08-2020
'''
import inspect
from inspect import signature

import pandas as pd, numpy as np
from warnings import warn

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

from Utils import _get_argument, _is_array_like, _to_DataFrame

__all__ = ['matplotlib_cmap', 'create_cmap', 
           'plot_pie', 'plot_radar', 'plot_pcs', 'plot_barh']
    
def matplotlib_cmap(name='tab20', n=10):

    '''
    Create list of color from `matplotlib.cmap`.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    name : `matplotlib.colors.Colormap` or str, 
    optional,  default:'tab20'
    \t The name of a colormap known to `matplotlib`. 
    
    n : `int`, optional, defualt:10
    \t Number of shades for defined color map.
    
    Returns
    -------
    List of color-hex codes from defined 
    `matplotlib.colors.Colormap`. Such list contains
    `n` shades.
    
    Examples
    --------
    >>> from matplotlib import cm
    >>> matplotlib_cmap(n=3)
    ['#440154', '#20908c', '#fde724']
    '''
    c_hex = '#%02x%02x%02x'
    c = cm.get_cmap(name)(np.linspace(0,1,n))
    c = (c*255).astype(int)[:,:3]
    return [c_hex % (c[i,0],c[i,1],c[i,2]) for i in range(n)]
  
def create_cmap(c1=None, c2=None):
    
    '''
    Creating `matplotlib.colors.Colormap` (Colormaps)
    with two colors.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    c1 : `hex code` or (r,g,b), optional, default:None
    \t The beginning color code. If `None`, `c1` is
    \t default to (23,10,8).
    
    c2 : hex code or (r,g,b), optional, default:None
    \t The ending color code. If `None`, `c2` is 
    \t default to (255,255,255).
    
    Returns
    -------
    `matplotlib.colors.ListedColormap`
    
    Examples
    --------
    >>> import numpy as np
    >>> create_cmap()
    <matplotlib.colors.ListedColormap at 0x12aa5aa58>
    '''
    to_rgb = lambda c : tuple(int(c.lstrip('#')[i:i+2],16) 
                           for i in (0,2,4))
    # Default values for `c1`, and `c2`.
    if c1 is None: c1 = (23,10,8)
    if c2 is None: c2 = (255,255,255)  
    # Convert to RGB.
    if isinstance(c1,str): c1 = to_rgb(c1)
    if isinstance(c2,str): c2 = to_rgb(c2)
    colors = np.ones((256,4))
    for i in range(3):
        colors[:,i] = np.linspace(c1[i]/256,c2[i]/256,256)
    colors = colors[np.arange(255,-1,-1),:]
    return ListedColormap(colors)

def plot_pie(ax, y, **params):
    
    '''
    Plot pie chart.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    ax : `axes.Axes` object
    \t `Axes` object from `matplotlib.pyplot.subplots`.
    
    y : 1D-array or `pd.Series`
    \t Array of labels.
    
    **params : dictionary of properties, optional
    \t params are used to specify or override properties of 
    
        colors : `list` of color-hex codes or (r,g,b)
        \t List must contain at the very least, 'n' of 
        \t color-hex codes or (r,g,b) that matches number
        \t of labels in 'y'. If `None`, the matplotlib  
        \t color maps, is default to `matplotlib_cmap`.

        labels : `list` of `str`
        \t `list` of labels whose items must be arranged
        \t in ascending order. If `None`, algorithm will
        \t automatically assign labels.
        
        piedict : dictionary
        \t A dictionary to override the default `ax.pie` 
        \t properties. If `piedict` is `None`, the 
        \t defaults are determined.
    
    Examples
    --------
    >>> import matplolib.pyplot as plt
    >>> import pandas as pd

    # Import data and create dataframe.
    >>> from sklearn.datasets import load_breast_cancer as data
    >>> X, y = data(return_X_y=True)

    # plot pie chart.
    >>> fig = plt.figure(figsize=(5,6))
    >>> ax = plt.subplot(polar=True)
    >>> params = dict(piedict={'shadow':False, 'startangle':90})
    >>> plot_pie(ax, y, **params)
    '''
    # Get `colors` from `**params`.
    if params.get('colors') is None:
        n_labels = np.unique(y).shape[0]
        colors = matplotlib_cmap(n=max(10,n_labels))
    else: colors = params.get('colors')
    
    # Get `labels` from `**params`.
    if params.get('labels') is None is None: 
        labels = ['Group {0}, y={1}'.format(n,s) 
                  for n,s in enumerate(np.unique(y),1)]
    else: labels = params.get('labels')
        
    a, sizes = np.unique(y, return_counts=True)    
    explode = (sizes==max(sizes)).astype(int)*0.1
    labels = ['%s \n (%s)' % (m,'{:,d}'.format(n)) 
              for m,n in zip(labels,sizes)]
    
    # Get `piedict` from `**params`.
    piedict = dict(explode=explode, labels=labels, autopct='%1.1f%%', 
                   shadow=True, startangle=90, colors=colors, 
                   wedgeprops = dict(edgecolor='black'))
    if params.get('piedict') is not None:
        piedict = {**piedict, **params.get('piedict')}
        
    ax.pie(sizes, **piedict)
    ax.axis('equal')

def plot_radar(ax, X, y, q=50, **params):

    '''
    Plot radar chart from normalized `X` by labels in `y`.
    
    .. versionadded:: 02-09-2020
    
    Parameters
    ----------
    ax : `axes.Axes` object
    \t `Axes` object from `matplotlib.pyplot.subplots`.

    X : array-like object
    \t All elements of X must be finite, i.e. no 
    \t `np.nan` or `np.inf`.

    y : 1D-array or `pd.Series`
    \t Array of labels.

    q : `float`, optional, default:50
    \t Percentile to compute, which must be between 0 
    \t and 100 e.g. If `q1` is 70, that means values 
    \t (normalize) from both classes will be 
    \t determined at 70th-percentile. Then difference 
    \t is computed from obtained values.
    
    **params : dictionary of properties, optional
    \t params are used to specify or override properties of 
    
        colors : `list` of color-hex codes or (r,g,b)
        \t List must contain at the very least, 'n' of 
        \t color-hex codes or (r,g,b) that matches number
        \t of labels in 'y'. If `None`, the matplotlib  
        \t color maps, is default to `matplotlib_cmap`.

        labels : `list` of `str`
        \t `list` of labels whose items must be arranged
        \t in ascending order. If `None`, algorithm will
        \t automatically assign labels.
        
        plotdict : dictionary
        \t A dictionary to override the default `ax.plot` 
        \t properties. If `plotdict` is `None`, the 
        \t defaults are determined.
        
        filldict : dictionary
        \t A dictionary to override the default `ax.fill` 
        \t properties. If `filldict` is `None`, the 
        \t defaults are determined.
        
        float_format : `format()` method
        \t String formatting method e.g. 
        \t {'float_format':'{:.3g}'.format}. 

    Examples
    --------
    >>> import matplolib.pyplot as plt
    >>> import pandas as pd

    # Import data and create dataframe.
    >>> from sklearn.datasets import load_breast_cancer as data
    >>> X, y = data(return_X_y=True)
    >>> X = pd.DataFrame(X, columns=[s.replace(' ','_') 
    ...                  for s in data().feature_names])

    # plot attributes in radar format.
    >>> fig = plt.figure(figsize=(5,6))
    >>> ax = plt.subplot(polar=True)
    >>> params = dict(float_format='{:.3g}'.format,
    ...               plotdict={'ls':'--','lw':2}, 
    ...               filldict={'alpha':0.2,'hatch':'///'})
    >>> plot_radar(ax, X, y, **params)
    '''
    # Count number of labels.
    cnt = np.bincount(y)
    
    # Convert array-like object to `pd.DataFrame`. 
    # The conversion is necessary as fields are used in
    # creating `ticklabels` in radar plot.
    X = _to_DataFrame(X)
    
    # To display all variables, particularly ones that 
    # have difference in scale, we normalize `X`. This 
    # makes `X` stays within 0 and 1, which allows 
    # comparison possible across variables and classes.
    n_min, n_max = np.nanpercentile(X.values, q=[0,100], axis=0)
    denom = np.where((n_max-n_min)==0,1,n_max-n_min)
    x = np.array((X.values-n_min)/denom)
    
    # Get `colors` from `**params`.
    if params.get('colors') is None:
        n_labels = np.unique(y).shape[0]
        colors = matplotlib_cmap(n=max(10,n_labels))
    else: colors = params.get('colors')
    
    # Get `labels` from `**params`.
    if params.get('labels') is None is None: 
        labels = ['Group {0}, y={1}'.format(n,s) 
                  for n,s in enumerate(np.unique(y),1)]
    else: labels = params.get('labels') 
        
    # Get `plotdict` from `**params`.
    plotdict = {'ls':'-', 'lw':1, 'ms':3, 'marker':'o', 
                'fillstyle':'full'}
    if params.get('plotdict') is not None:
        plotdict = {**plotdict, **params.get('plotdict')}
        
    # Get `filldict` from `**params`.
    filldict = {'alpha':0.5}
    if params.get('filldict') is not None:
        filldict = {**filldict, **params.get('filldict')}
        
    # Angle of plots.
    angles = [n/float(x.shape[1])*2*np.pi for n in range(x.shape[1])]
    angles += angles[:1]

    # If you want the first axis to be on top.
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    
    # Get `ticklabel_format`, if any.
    if params.get('float_format') is not None: 
        float_format = params.pop('float_format')
    else: float_format = '{:.4g}'.format
    
    def _ticklabels(X, y, q, float_format):
        # Find nth percentile (`q`) for each label.
        # Each element along the length of array,
        # represents value in each variable.
        v = [np.nanpercentile(X[y==c], q, axis=0).ravel() 
             for c in np.unique(y)]
        # Reshape array = row: variables, columns: labels.
        v = np.hstack([n.reshape(-1,1) for n in v])
        # Convert all values in array to str (`float_format`).
        v = [', '.join(tuple([float_format(v1) for v1 in v0])) for v0 in v]  
        # Parenthesize `str`.
        v = ['( {} )'.format(v0) for v0 in v]              
        return ['\n'.join((f,v0)) for f,v0 in zip(list(X),v)]

    ticklabels = _ticklabels(X, y, q, float_format)
            
    # Draw one axis per variable and add ticklabels. 
    kwargs = dict(color='#3d3d3d', fontsize=10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ticklabels, **kwargs)

    # Set alignment of ticks.
    for n,t in enumerate(ax.get_xticklabels()):
        if (0<angles[n]<np.pi): t._horizontalalignment = 'left'
        elif (angles[n]>np.pi): t._horizontalalignment = 'right'
        else: t._horizontalalignment = 'center'

    for n,k in enumerate(np.unique(y)):
        values = np.nanpercentile(x[y==k], q, axis=0).tolist()
        values += [values[0]]
        ax.plot(angles, values,**{**plotdict, **{'c':colors[n]}})
        kwargs = dict(c=colors[n], label=labels[n] + ' ({:,d})'.format(cnt[n]))
        ax.fill(angles, values, **{**filldict, **kwargs})

    ax.set_yticklabels([])
    kwargs = dict(facecolor='none', edgecolor='none',
                  fontsize=10, bbox_to_anchor=(0,0))
    ax.legend(**kwargs)
    ax.set_facecolor('White')
    ax.grid(True, color='grey', lw=0.5, ls='--')

def plot_pcs(ax, a, PC, loadings, pc_pair=[1,2], cutoff=0.5, factor=None, **params):
    
    '''
    Plot principal components with binary class.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    ax : `axes.Axes` object
    \t `Axes` object from `matplotlib.pyplot.subplots`.
    
    a : array of `bool` or `int`
    \t Binary array, where `0` and `1` represent `Normal`,
    \t and `Anomaly`, respectively.
    
    PC : `pd.DataFrame` object
    \t Principal Components.
    
    loadings : `pd.DataFrame` object
    \t `eigenvector` dataframe with a shape of 
    \t n_features by n_features (see `pca`).
    
    pc_pair : `list` of `int`, optional, default: [1,2]
    \t Pair of principal components to be plotted.
    \t The 0th index is plotted in x-axis.

    cutoff : `float`, optional, default: 0.5
    \t Minimum value of `eigenvectors` that is used to 
    \t determine variables that explain principal 
    \t components. 

    factor : `float`, optional, default: None
    \t `factor` is applied to `eigenvector` to scale 
    \t the length of vector towards visibility in plot. 
    \t If `None`, `factor` is determined automatically.
        
    **params : dictionary of properties, optional
    \t params are used to specify or override properties of 
    
        colors : `list` of color-hex codes or (r,g,b)
        \t List must contain at the very least, 'n' of 
        \t color-hex codes or (r,g,b) that matches number
        \t of labels in 'y'. If `None`, the matplotlib  
        \t color maps, is default to `matplotlib_cmap`.
        
        labels : `list` of `str`
        \t `list` of labels whose items must be arranged
        \t in ascending order. If `None`, algorithm will
        \t automatically assign labels.
        
        n : `int`, optional, default: 5000
        \t Number of samples from axis to be plotted.

        random_state : `int`, optional, default: 0
        \t Seed for the random number generator.
        
        scatterdict : dictionary
        \t A dictionary to override the default `ax.scatter` 
        \t properties. If `scatterdict` is `None`, the 
        \t defaults are determined.
    
    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X, eigen_vectors, eigen_values, var = pca(X)
    >>> fig, ax = plt.subplots()
    >>> plot_pcs(ax, y, X, loadings, cutoff=0.3)
    >>> plt.show()
    '''
    # Get `colors` from `**params`.
    if params.get('colors') is None:
        colors = matplotlib_cmap(n=max(10,len(np.unique(a))))
    else: colors = params.get('colors')
        
    # Get `labels` from `**params`.
    if params.get('labels') is None is None: 
        labels = ['Group {0}, y={1}'.format(n,s) 
                  for n,s in enumerate(np.unique(a),1)]
    else: labels = params.get('labels')
        
    # Add amount and percentage.
    def _amt_pct(y,n):
        amt = (y==n).sum(); pct = amt/len(y)*100
        return ': {:,.4g} ({:,.3g}%)'.format(amt, pct)
    labels = [s + _amt_pct(a,n) for s,n in zip(labels,np.unique(a))]
    
    # Get `n` from `**params`.
    if params.get('n') is None: n = 5000
    else: n = params.get('n')
    
    # Get `random_state` from `**params`.
    if params.get('random_state') is None: random_state = 0
    else: random_state = params.get('random_state')
        
    # Get `scatterdict` from `**params`.
    scatterdict = dict(s=15, alpha=0.4, marker='s', ec='#4b4b4b')
    if params.get('scatterdict') is not None:
        scatterdict = {**scatterdict,**params.get('scatterdict')}
        
    # If number of samples from `PC` exceeds `n`, `X` will
    # be resampled to match with `n`, otherwise everything 
    # remains unchanged.
    X = _to_DataFrame(PC)
    if X.shape[0] > n:
        kw = dict(n=n, random_state=random_state)
        X['flag'] = a.copy()
        X = X.sample(**kw).reset_index(drop=True)
        y = X.pop('flag')
    else: y, X = a.copy(), PC.copy() 
        
    # Select `eigenvector`, whose magnitude is above `cutoff`.
    pair = loadings.columns[np.array(pc_pair)-1]
    index = (abs(loadings[pair]).max(axis=1)>cutoff).values
    
    # Set horizontal and vertical reference lines.
    kw = dict(lw=1, ls='--', color='#2f3542')
    ax.axvline(0, **kw); ax.axhline(0, **kw)

    # Plot scatter by label.
    for n,c in enumerate(np.unique(a)):
        kwargs = dict(color=colors[n],label=labels[n])
        args = (X.loc[y==c, pair[0]], X.loc[y==c, pair[1]])
        ax.scatter(*args, **{**kwargs,**scatterdict})
    
    # Determine `factor` to scale `eigenvector`.
    if (factor is None) & (np.array(index).sum()>0):
        # Minimum distance from center (0,0).
        min_ax = min(abs(np.array(ax.get_ylim() + ax.get_xlim())))
        # Maximum value of eigenvectors.
        max_pc = max(abs(loadings.loc[index, pair].values.ravel()))
        # Factor reduced by 10%.
        factor = 0.9*min_ax/max_pc
    else: factor = 0
    
    # Determine eigenvector based on `cutoff`.
    if (np.array(index).sum()>0):
        vec = loadings.loc[index, pair].values*factor
        var = loadings.loc[index,:].index
        
        # `bbox` arguments and eigenvectors with value above `cutoff`.
        bbox_kw = dict(boxstyle='round', facecolor='white', 
                       alpha=0.7, edgecolor='#2f3542', pad=0.4)

        # Influencing Variables for Principal Components.
        for n,(dx,dy) in enumerate(vec):
            ax.plot([0,dx], [0,dy], color='#3742fa', lw=2)
            ax.scatter(dx, dy, s=20, color='#3742fa')
            kw = dict(xytext=(0, 15*np.sign(dy)), 
                      textcoords='offset points', 
                      ha='center', color='#2f3640', bbox=bbox_kw)
            ax.annotate(var[n], (dx,dy), **kw) 
    else: vec = []
    
    # Set label for both axes as well as title.
    kw = dict(fontsize=12, color='#2f3542', fontweight='bold')
    ax.set_xlabel(pair[0],**kw); ax.set_ylabel(pair[1],**kw)
    title = 'Factor: {:,.3g}, Cutoff: {:,.3g}, # of features: {:,.2g}'
    ax.set_title(title.format(factor, cutoff, len(vec)),**kw) 
    ax.legend(loc='best', fontsize=9)
    
def plot_barh(ax, y, width, **params):
    
    '''
    Plot horizontal bar.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    ax : `axes.Axes` object
    \t `Axes` object from `matplotlib.pyplot.subplots`.

    y : array-like or `list` of `str`
    \t The y coordinates of the bars as well as `ticklabels`.

    width : array-like or `list` of `float`
    \t The width(s) of the bars.
    
    **params : dictionary of properties, optional
    \t params are used to specify or override properties of 
        
        groups : `list` of `str` or `int`
        \t `list` of multiclass that categorizes `y` into
        \t subgroups. If `groups` is `None`, it will treat
        \t `y` as a one single group.
        
        colors : `list` of color-hex codes or (r,g,b)
        \t List must contain at the very least, 'n' of 
        \t color-hex codes or (r,g,b) that matches number
        \t of labels in 'groups'. If `None`, the matplotlib  
        \t color maps, is default to `matplotlib_cmap`.

        labels : `list` of `str`
        \t `list` of labels whose items must be arranged
        \t in correspond to `groups`. If `None`, algorithm
        \t will automatically assign labels.
        
        barhdict : dictionary
        \t A dictionary to override the default `ax.barh` 
        \t properties. If `barhdict` is `None`, the 
        \t defaults are determined.
        
        annodict : dictionary
        \t A dictionary to override the default `ax.annotate` 
        \t properties. If `annodict` is `None`, the 
        \t defaults are determined.
        
        float_format : `format()` method
        \t String formatting method for annotation e.g. 
        \t {'float_format':'{:.3g}'.format}. If `float_format`
        \t is `None`, it defaults to '{:,.3g}'.format.
    
    Examples
    --------
    >>> import matplolib.pyplot as plt
    >>> import numpy as np

    # Create data
    >>> y = ['a','b','c','d']
    >>> width = [-2.1, 0.4, 1.2, 2.4]
    
    >>> fig = plt.figure(figsize=(5,6))
    >>> ax = plt.subplot()
    >>> plot_barh(ax, y, width)
    '''
    # Get `groups` from `**params`.
    if params.get('groups') is None:
        groups = np.zeros(len(y))
    else: groups = params.get('groups')
    
    # Get `colors` from `**params`.
    if params.get('colors') is None:
        n_labels = np.unique(groups).shape[0]
        colors = matplotlib_cmap(n=max(10,n_labels))
    else: colors = params.get('colors')
    
    # Get `labels` from `**params`.
    if params.get('labels') is None is None: 
        labels = ['Group {0}, y={1}'.format(n,s) 
                  for n,s in enumerate(np.unique(groups),1)]
    else: labels = params.get('labels')
        
    # Get `barhdict` from `**params`.
    barhdict = dict(alpha=0.7, ec='#2f3542', height=0.7)
    if params.get('barhdict') is not None:
        barhdict = {**barhdict, **params.get('barhdict')}
    
    # Get `annodict` from `**params`.
    annodict = dict(textcoords='offset points', 
                    va='center', color='#2f3640')
    if params.get('annodict') is not None:
        annodict = {**annodict, **params.get('annodict')}
    
    # Get `float_format` from `**params`.
    if params.get('float_format') is None is None: 
        float_format = '{:,.3g}'.format
    else: float_format = params.get('float_format')
    
    # Horizontal bar.
    for n,c in enumerate(np.unique(groups)):
        kwargs = {**barhdict,**dict(color=colors[n],label=labels[n])}
        ax.barh(range(len(y)), np.where(groups==c,width,np.nan), **kwargs)   
    ax.set_yticks(range(len(y)))
    ax.set_yticklabels(y, color='#2f3640', fontsize=10)
    
    # Annotate horizontal bars.
    for (dx,dy) in enumerate(width):
        if dy>=0: annodict = {**annodict,**dict(xytext=(2,0),ha='left')}
        else: annodict = {**annodict,**dict(xytext=(-2,0),ha='right')}
        ax.annotate(float_format(dy), (dy,dx), **annodict)

    # Set `ylimit` and `xlimit`.
    ax.set_ylim(-0.5, len(y)-0.5)
    incr = (max(width) - min(width))*0.2
    ax.set_xlim(min(width)-incr, max(width)+incr)
    ax.legend(loc='best', framealpha=0)
