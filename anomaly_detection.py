'''
Available methods are the followings:
[1] PlotScore_base

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 12-02-2022

'''
import pandas as pd, numpy as np, sys
from inspect import signature
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import matplotlib.tri as tri
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.ticker import(FixedLocator, 
                              FixedFormatter, 
                              StrMethodFormatter,
                              FuncFormatter)
import collections
from itertools import product

# Adding fonts
from matplotlib import font_manager
paths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
for font_path in paths:
    if font_path.find("Hiragino Sans GB W3")>-1: 
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = prop.get_name()
        plt.rcParams.update({'font.family':'sans-serif'})
        plt.rcParams.update({'font.sans-serif':prop.get_name()})
        plt.rc('axes', unicode_minus=False)
        break

__all__ = ["PlotScore_base"]

def PlotScore_base(y_true, scores, ax=None, bins="fd", n_ticks=6, 
                   labels=None, colors=None, bar_kwds=None, 
                   tight_layout=True):

    '''
    Histogram plot for anomaly socres.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target array (binary).

    scores : array-like of shape (n_samples,)
        Score array.
        
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
        
    bins : int or str, default="fd"
        Number of bins (np.histogram_bin_edges).

    n_ticks : int, default=6
        Number of ticks to be displayed on x-axis, and y-axis.   
    
    labels : list, default: None
        A sequence of strings providing the labels for each class. 
        If None, 'Class {n+1}' is assigned, where n is the class in y.
    
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to number 
        of classes. If None, it uses default colors from Matplotlib.

    bar_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.bar". If None, it uses 
        default settings.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
 
    References
    ----------
    .. [1] https://numpy.org/doc/stable/reference/generated/numpy.
           histogram_bin_edges.html#numpy.histogram_bin_edges
    
    Returns
    -------
    ax : Matplotlib axis object
            
    '''  
    
    # =============================================================
    if ax is None: ax = plt.subplots(figsize=(6.5, 4.2))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(2)] 
              if colors is None else colors)
    # -------------------------------------------------------------
    if labels is None: labels = ['Class 0', 'Class 1']
    default =  {'alpha': 0.2, "width": 0.8}   
    if bar_kwds is None: bar_kwds = default
    else: bar_kwds = {**default, **bar_kwds}
    # =============================================================    
        
    # Create `bins` and `dist`.
    # =============================================================
    cutoff = ks_cutoff(y_true, scores)
    bins = np.unique(np.histogram_bin_edges(scores, bins=bins))
    width = np.diff(bins)[0]
    length = (cutoff - bins[0]) / width
    step = (length - np.floor(length)) * width
    if step>0: bins = np.r_[bins[0] + step - width, bins + step] 
    eps  = np.finfo(float).eps
    bins[-1] = bins[-1] + eps
    x = bins[:-1] + np.diff(bins)/2
    # -------------------------------------------------------------
    # Distribution of samples.
    n_samples = len(y_true)
    data = [scores[y_true==n] for n in [0,1]]
    kwds = dict(bins=bins, range=(0, 1+eps))
    dist = [np.histogram(data[n], **kwds)[0]/
            n_samples for n in [0,1]]
    # =============================================================

    # Plot horizontal bar.
    # =============================================================
    width = np.diff(bins)[0] * bar_kwds.get("width", 0.8)
    bar_kwds.update({"width": width})
    patches, labels_, text = [], [], " ({:.0%})".format
    for n in [0,1]:
        bar_kwds.update({"facecolor" : colors[n]}) 
        ax.bar(x, dist[n], **bar_kwds)
        kwds = bar_kwds.copy()
        kwds.update({"facecolor" : "none", 
                     "edgecolor" : colors[n], 
                     "linewidth" : bar_kwds.get("linewidth",1.2),
                     "linestyle" : bar_kwds.get("linestyle","-"),
                     "alpha"     : 1}) 
        ax.bar(x, dist[n], **kwds)
        patches += [(mpl.patches.Patch(**update_kwargs(bar_kwds)), 
                     mpl.patches.Patch(**update_kwargs(kwds)))]
        labels_ += [labels[n] + text(sum(y_true==n)/n_samples)]
    # ============================================================= 
    
    # Draw all vertical lines and their components
    # =============================================================
    kwds = dict(color="#bbbbbb", linestyle="--", lw=0.5, zorder=-1)
    ax.axvline(cutoff, **kwds)    
    args = (ax.transData, ax.transAxes)
    trans = transforms.blended_transform_factory(*args)
    ax.text(cutoff, 1, "Cutoff = {:+,.2f}".format(cutoff), 
            transform=trans, fontsize=13, va='bottom', ha="center")
    x_locator = FixedLocator(np.sort([cutoff]))
    ax.xaxis.set_minor_locator(x_locator)
    ax.tick_params(axis="x", which="minor", length=3, color="k")
    # =============================================================

    # Set other attributes.
    # =============================================================
    ax.tick_params(axis='both', labelsize=11)
    format_y = ticker.PercentFormatter(xmax=1, decimals=0)
    format_x = ticker.FormatStrFormatter('%0.2f')
    ax.yaxis.set_major_formatter(format_y)
    ax.xaxis.set_major_formatter(format_x)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(n_ticks))
    # -------------------------------------------------------------
    ax.set_xlim(*ax.get_xlim())
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(0, y_max/0.85)
    # -------------------------------------------------------------
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax.set_xlabel('Scores', fontsize=13)
    ax.set_ylabel('% of Samples', fontsize=13)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    ax.text(0, 1.01, "f(x)", fontsize=13, va='bottom', ha="center", 
            transform=transform)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    # -------------------------------------------------------------
    ax.legend(patches, labels_, loc="best", borderaxespad=0.3, 
              columnspacing=0.3, handletextpad=0.5, 
              prop=dict(size=12))
    if tight_layout: plt.tight_layout()
    # =============================================================

    return ax

def ks_cutoff(y_true, y_score, decimal=4):
    
    '''Private function: Determine KS cutoff'''
    scores = np.round(y_score, decimal)
    bins = np.unique(scores)
    bins[-1] = bins[-1] + np.finfo("float32").eps
    dist = []
    for n in range(2):
        hist = np.histogram(scores[y_true==n], bins)[0]
        dist.append(np.cumsum(hist)/sum(hist))
    n = np.argmax(abs(dist[0]-dist[1]))
    return np.round(bins[n:n+2].mean(), decimal)

def update_kwargs(kwargs):
    
    '''Private function: Update params for patches'''
    params = {}
    for key,value in kwargs.items():
        try: 
            mpl.patches.Patch(**{key:value})
            params.update({key:value})
        except: pass
    return params 