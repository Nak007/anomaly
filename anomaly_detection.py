'''
Method
------
 1) pca
 2) plot_pcs
 3) find_cutoff
 4) find_condition
 5) optimize_condition
 6) matplotlib_cmap
 7) create_cmap
 8) plot_pie
 9) plot_radar
10) diff_normalize
11) plot_barh
12) describe_y
13) outliers
'''
from sklearn.preprocessing import StandardScaler
import pandas as pd, numpy as np, time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
import ipywidgets as widgets
from IPython.display import display
from itertools import combinations

def pca(X, val_cutoff=1.0, vec_cutoff=0.5, stdev=(-3,3)):
    
    '''
    Perform `Principal Component Analysis` (PCA) to `X`.
    
    Parameters
    ----------
    X : `pd.DataFrame` object, of shape (n_samples, n_features)
    \t Input array.
    
    val_cutoff : `float`, optional, (default:1.0)
    \t Any eigen pair (`eigenvalue`,`eigenvector`), whose
    \t `eiginvalue` is greater than or eqaul to `cutoff` is 
    \t selected as one of the principal components. 
    \t The minimum number of `PC` is set to be 2.
    
    vec_cutoff : `float`, optional, (default:0.5)
    \t Minimum value of eigenvectors that is used to determine 
    \t importance of variables to principal components.
    
    stdev : tuple of `float`, optional, (default:(-3,3))
    \t Given an interval (`min`,`max`), standardized `X` outside 
    \t the interval (standard deviation) are clipped to the 
    \t interval edges. For example, if an interval of (0, 1) is 
    \t specified, values smaller than 0 become 0, and values 
    \t larger than 1 become 1.
    
    Returns
    -------
    PC : `pd.DataFrame` object, of shape (n_samples, n_PCs)
    \t Principal Components of `X`, where number of `PC` is 
    \t determined by corresponding `eigenvalue` that is greater
    \t than or equal to `val_cutoff`.
    
    eigenvector : `pd.DataFrame` object, of shape 
    (n_features, n_features)
    \t `eigenvector` is a nonzero vector that applies linear
    \t transformation to vectors.
    
    eigenvalue : list of 'float' 
    \t The corresponding `eigenvalue` is the factor by which 
    \t the eigenvector is scaled either stretched or shrinked.
    
    var : list of `str`
    \t List of variables that important to principal components.
    '''
    # Standardized X.
    X_std = StandardScaler().fit_transform(X)
    X_std = np.clip(X_std, stdev[0], stdev[1])
    # Compute correlation.
    corr = pd.DataFrame(X_std,columns=list(X)).corr()
    # Compute eigenvalue and eigenvector (loadings).
    eigen_value, eigen_vector = np.linalg.eig(corr)
    
    # Sort eigenvectors by eigenvalues.
    eigen = [v for v in zip(eigen_value, eigen_vector)]
    eigen.sort(reverse=True, key=lambda x: x[0])
    
    # Sorted eigenvalue and factor loadings.
    rng = lambda x : np.arange(1,x.shape[1]+1)
    col = lambda x : ['PC%s'%str(n).zfill(2) for n in rng(x)]
    to_dict = lambda x : dict((n,k[1]) for n,k in enumerate(x))
    eigen_val = np.array([k[0] for k in eigen])
    eigen_vec = pd.DataFrame(to_dict(eigen))
    eigen_vec.index, eigen_vec.columns = list(X), col(eigen_vec)
    
    # Determine principal components given `val_cutoff`.
    n_pc = max(2,(eigen_val>=val_cutoff).sum())
    pc = X_std.dot(eigen_vec.values[:,:n_pc])
    pc = pd.DataFrame(pc, columns=col(pc))
    
    # Select vector, whose magnitude is above `vec_cutoff`.
    loadings = eigen_vec.iloc[:,:(eigen_val>=val_cutoff).sum()]
    var = list(X.columns[(abs(loadings).max(axis=1)>vec_cutoff).values])
   
    return pc, eigen_vec, eigen_val, var

def plot_pcs(ax, a, b, loadings, pc=[1,2], cutoff=0.5, factor=None, n=5000, random_state=0):
    
    '''
    Plot principal components.
    
    Parameters
    ----------
    ax : `axes.Axes` object
    \t `Axes` object from `matplotlib.pyplot.subplots`.
    
    a : array of `bool` or `int`
    \t Binary array, where `0` and `1` represent `Normal`,
    \t and `Anomalous` samples, respectively.
    
    b : `pd.DataFrame` object, of shape (n_samples, n_PCs)
    \t Principal components.
    
    loadings : `pd.DataFrame` object, of shape 
    (n_features, n_features)
    \t `eigenvector` dataframe (see `pca`).
    
    pc : list of `int`, optional, (default:[1,2])
    \t Pair of principal components to be plotted.
    
    cutoff : `float`, optional, (default:0.5)
    \t Minimum value of `eigenvectors` that is used to determine 
    \t importance of variables to principal components. 
    
    factor : `float`, optional, (default:None)
    \t `factor` is applied to `eigenvector` to magnify the 
    \t length of vector towards visibility in plot. If `None`,
    \t `factor` is determined automatically.
    
    n : `int`, optional, (default:5000)
    \t Number of samples from axis to be plotted.
    
    random_state : `int`, optional, (default:0)
    \t Seed for the random number generator.
    '''
    if len(b) > n:
        kw = dict(n=n, random_state=random_state)
        X = b.copy(); X['flag'] = a.copy()
        X = X.sample(**kw).reset_index(drop=True)
        y = X.pop('flag')
    else: y, X = a.copy(), b.copy() 

    # Select vector, whose magnitude is above `cutoff`.
    pc = loadings.columns[np.array(pc)-1]
    index = (abs(loadings[pc]).max(axis=1)>cutoff).values
    
    # Plot `Inlier` and `Outlier`.
    c = lambda y,n : '{:,.4g}'.format((y==n).sum())
    p = lambda y,n : ' ({:,.3g}%)'.format((y==n).sum()/len(y)*100)
    kw = dict(s=15, color='#009432', alpha=0.4, marker='s',
              label='Normal: ' + c(y,0) + p(y,0),ec='#4b4b4b')
    ax.scatter(X.loc[y==0, pc[0]], X.loc[y==0, pc[1]], **kw)
    kw = dict(s=25, color='#EA2027', alpha=0.7, marker='s',
              label='Anomalous: '+ c(y,1) + p(y,1),ec='#4b4b4b')
    ax.scatter(X.loc[y==1, pc[0]], X.loc[y==1, pc[1]], **kw)
    
    # Set horizontal and vertical reference lines.
    kw = dict(lw=1, ls='--', color='#2f3542')
    ax.axvline(0, **kw); ax.axhline(0, **kw)
    
    if (factor is None) & (np.array(index).sum()>0):
        # Minimum distance from center (0,0).
        min_ax = min(abs(np.array(ax.get_ylim() + ax.get_xlim())))
        # Maximum value of eigenvectors.
        max_pc = max(abs(loadings.loc[index, pc].values.ravel()))
        # Factor reduced by 10%.
        factor = 0.9*min_ax/max_pc
    else: factor = 0
    
    # Determine eigenvector based on `cutoff`.
    if (np.array(index).sum()>0):
        vec = loadings.loc[index, pc].values*factor
        var = loadings.loc[index,:].index
        
        # `bbox` arguments and eigenvectors with value above `cutoff`.
        bbox_kw = dict(boxstyle='round', facecolor='white', 
                       alpha=0.7, edgecolor='#2f3542', pad=0.4)

        # Influencing Variables for Principal Components.
        for n,(dx,dy) in enumerate(vec):
            ax.plot([0,dx], [0,dy], color='#3742fa', lw=2)
            ax.scatter(dx, dy, s=20, color='#3742fa')
            kw = dict(xytext=(0, 15*np.sign(dy)), textcoords='offset points', 
                      ha='center', color='#2f3640', bbox=bbox_kw)
            ax.annotate(var[n], (dx,dy), **kw) 
    else: vec = []
    
    # Set label for both axes as well as title.
    kw = dict(fontsize=12, color='#2f3542', fontweight='bold')
    ax.set_xlabel(pc[0],**kw); ax.set_ylabel(pc[1],**kw)
    title = 'Anomaly Detection\nFactor: {:,.3g}, Cutoff: {:,.3g}, # of features: {:,.2g}'
    ax.set_title(title.format(factor, cutoff, len(vec)),**kw) 
    ax.legend(loc='best', fontsize=9)

def find_cutoff(x, y, decimal=4, n_max=10):
    
    '''
    Find `cutoff` by optimizing `F1-Score` from center of 
    distribution (median) towards both tails.
    
    Parameters
    ----------
    x : `pd.Series` or array
    \t Array of `float` without `np.nan`.
    
    y : array of `bool` or `int`
    \t Binary array, where `0` and `1` represent `normal`,
    \t and `anomalous` samples, respectively.
    
    decimal : `int`, optional, (default:4)
    \t Number of decimal places only for `f1_score`.
    
    Returns
    -------
    dictionary of :
    
    cutoff : list of [`str`,`float`]
    \t `str` is either `≤` or `≥` to identify which side
    \t of distribution that anomaly is on beyond `cutoff`.
    
    confusion_matrix : array of `int`, 
    \t It elements are arranged in the following order i.e. 
    \t True-Negative, False-Negative, False-Positive, and 
    \t True-Positive.
    
    f1_score : `float`
    \t `f1_score` is a measure of a test's accuracy. It 
    \t considers both the precision and the recall of the 
    \t test to compute the score. The value is in 0-1 range.
    
    n_max : `int`, optional, (default:10)
    \t Maximum number of `cutoff` per iteration.
    '''
    def cfm_(x, y, cutoff, right_side=True):
        if right_side: 
            tn = (y[x< cutoff]==0).sum()
            fn = (y[x< cutoff]==1).sum()
            fp = (y[x>=cutoff]==0).sum()
            tp = (y[x>=cutoff]==1).sum()
        elif right_side==False:
            tn = (y[x> cutoff]==0).sum()
            fn = (y[x> cutoff]==1).sum()
            fp = (y[x<=cutoff]==0).sum()
            tp = (y[x<=cutoff]==1).sum()
        return (tn,fn,fp,tp)
    
    def cal_f1_score(x):
        r = x[:,3]/x[:,[1,3]].sum(axis=1).ravel()
        p = x[:,3]/x[:,[2,3]].sum(axis=1).ravel()
        return (2*r*p)/np.where(r+p==0,1,r+p)
    
    def get_cutoffs(x, right_side=True):
        if right_side: return np.unique(x[x>=np.median(x)])[1:-1]
        else: return np.unique(x[x<=np.median(x)])[1:-1]
    
    retval, scores, sign = [], [], ['≤','≥']
    for k,right_side in enumerate([False, True]):
        # Get list of cutoffs given side of distribution.
        cutoffs = get_cutoffs(x, right_side)
        if len(cutoffs)>n_max:
            step = max(int(len(cutoffs)/n_max),1)
            cutoffs = np.array([cutoffs[n] for n in 
                                range(0,len(cutoffs),step)])

        # Attain Confusion Matrix's elements from all cutoffs.
        if len(cutoffs)>0:
            CFM = np.vstack([cfm_(x, y, c,right_side) for c in cutoffs])
            # Calculate F1-Score for all cutoffs.
            f1 = np.round(cal_f1_score(CFM),decimal)
            # Store cutoff that returns the best F1-Score.
            retval.append(dict(cutoff=[sign[k],cutoffs[np.argmax(f1)]],
                               confusion_matrix=CFM[np.argmax(f1),:].tolist(),
                               f1_score=max(f1)))
            scores.append(max(f1))
    # len(scores)=0, when all values are the same (constant).
    return retval[np.argmax(scores)] if len(scores)>0 else None

def find_condition(X, y, columns=None, decimal=4, n_max=10):
    
    '''
    Basing on `y`, it determines the best `cutoff` that defines 
    anomalies through use of `f1_score`. Each `cutoff` is set 
    dependently to its predecessor one after another. 
    
    Parameters
    ----------
    X : `pd.DataFrame` object, of shape (n_samples, n_features)
    \t `X` must be numerical and not contain `np.nan`. 

    y : array of `bool` or `int`
    \t Binary array, where `0` and `1` represent `Normal`,
    \t and `Anomalous` samples, respectively.
    
    columns : list of `str`, optional, (default:None)
    \t List of field names, where condition finding runs 
    \t according to the list ordering (left-to-right).
    \t If `None`, `X.columns` will be used instead.
    
    decimal : `int`, optional, (default:4)
    \t Number of decimal places only for `f1_score`.
    
    n_max : `int`, optional, (default:10)
    \t Maximum number of `cutoff` per iteration.
    
    Returns
    -------
    dictionary of :
    
    `criteria` : list of [`str`,`str`,`float`]
    \t Corresponding to order of variables (`columns`), each 
    \t `list` contains information of `cutoff` that defines 
    \t anomalies e.g. ['variable','≥',99]. This example 
    \t implies that given `variable`, any value that is more 
    \t than or equal to 99 is defined as `anomaly`.
    
    `confusion_matrix` :  list of `int`
    \t Elements in `confusion_matrix` are arranged in
    \t the following order i.e. True-Negative, 
    \t False-Negative, False-Positive, and True-Positive.
    
    `f1_score` : `float`
    \t `f1_score` is a measure of a test's accuracy. It 
    \t considers both the precision and the recall of the 
    \t test to compute the score. The value is in 0-1 range.
    '''
    if columns is None: columns = list(X)
    x, t = np.array(X[columns]), np.array(y)
    index, criteria, cfm = np.full(len(y), True), [], []
    for (n,v) in enumerate(columns):
        if t.sum()>0:
            x, t = x[index,:], t[index]
            m = find_cutoff(x[:,n], t, n_max=n_max)
            if m is not None:
                criteria.append([v] + m['cutoff'])
                cfm.append(m['confusion_matrix'])
                if m['cutoff'][0] == '≤': index = ~(x[:,n]<=m['cutoff'][1])
                else: index = ~(x[:,n]>=m['cutoff'][1])
        else: break
    
    if cfm is not None:
        # Optimize `f1-score`.
        cfm = np.array(cfm); fn = cfm[:,1]
        tp, fp = np.cumsum(cfm[:,3]), np.cumsum(cfm[:,2])
        r, p = tp/(tp+fn), tp/(tp+fp)
        f1_score = (2*r*p)/(r+p); n = np.argmax(f1_score)
        criteria = criteria[:n+1]; f1_score = np.round(f1_score[n],decimal)

        # Determine `Y` given `cutoff`.
        x = np.array(X[columns])
        r0 = lambda x,n,v : np.array(x[:,n]<=v).reshape(-1,1)
        r1 = lambda x,n,v : np.array(x[:,n]>=v).reshape(-1,1)
        Y = [r0(x,n,m[2]) if m[1]=='≤' else r1(x,n,m[2]) 
             for (n,m) in enumerate(criteria)]
        if len(Y)>1: Y = np.hstack(Y).sum(axis=1).astype(bool).astype(int)
        # Confustion Matrix (tn,fn,fp,tp).
        cfm = [((y==r) & (Y==n)).sum() for n in range(2) for r in range(2)]
        return dict(criteria=criteria, confusion_matrix=cfm, f1_score=f1_score)
    else: return None

def optimize_condition(X, y, n_max=10):

    '''
    This function finds variable that provides the highest 
    `f1_score` (see `find_condition`) in each iteration. 
    The optimization works dependently to previous iteration,
    meaning that once `variable` along with its corresponding 
    `cutoff` are determined, number of samples in `X` will be 
    reduced accordingly, and the process continues until 
    `f1_score` stops improving. This filering is called 
    `cascading filter`.

    Parameters
    ----------
    X : `pd.DataFrame` object, of shape (n_samples, n_features)
    \t `X` must be numerical and not contain `np.nan`. 

    y : array of `bool` or `int`
    \t Binary array, where `0` and `1` represent `Normal`,
    \t and `Anomalous` samples, respectively.
    
    n_max : `int`, optional, (default:10)
    \t Maximum number of `cutoff` per iteration.

    Returns
    -------
    dictionary of :
    
    `criteria` : list of [`str`,`str`,`float`]
    \t Corresponding to `f1_score`, each `list` contains 
    \t information of `cutoff` that defines anomalies e.g. 
    \t ['variable','≥',99]. This example implies that given 
    \t `variable`, any value that lies beyond 99 is defined 
    \t as `anomaly`. This `cascading filters` allow user to 
    \t filter based on multiple values in a hierarchy.
    
    `confusion_matrix` :  list of `int`
    \t Elements in `confusion_matrix` are arranged in
    \t the following order i.e. True-Negative, 
    \t False-Negative, False-Positive, and True-Positive.
    
    `f1_score` : `float`
    \t `f1_score` is a measure of a test's accuracy. It 
    \t considers both the precision and the recall of the 
    \t test to compute the score. The value is in 0-1 range.
    
    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from sklearn.preprocessing import StandardScaler
    
    # Import data and create dataframe.
    >>> from sklearn.datasets import load_breast_cancer as data
    >>> X, y = data(return_X_y=True)
    >>> X = pd.DataFrame(X, columns=[s.replace(' ','_') 
    ...                  for s in data().feature_names])
    
    # Standardized X.
    X_std = StandardScaler().fit_transform(X)
    
    # Anomaly detection.
    >>> from sklearn.ensemble import IsolationForest
    >>> IF = IsolationForest(n_estimators=200, random_state=0, 
    ...                      behaviour='new', contamination='auto')
    >>> y_ = np.where(IF.fit_predict(X_std)==-1,1,0)
    
    # Determine `criteria`.
    >>> optimize_condition(X, y_)
    '''
    t = widgets.HTMLMath(value='Initializing . . .')
    display(widgets.HBox([t])); time.sleep(1)
    var = lambda u,v : u+[v] if len(u)>0 else [v]
    
    # Initialize variables.
    remain, u, f1, n_var, n = list(X), list(), 0, -1, 0
    while len(u)!=n_var:
        # Determine score from all combinations.
        n_var, score = len(u), []
        combi = [var(u,v) for v in remain]
        for c in combi:
            n += 1; t.value = 'Calculating   ' + '  ■' * (n%20)
            a = find_condition(X, y, c, n_max=n_max)
            if a is not None: score.append(a['f1_score'])
        # If new combination's f1-score is greater than the
        # previous one, then this combination will replace its
        # predecessor.
        if max(score) > f1:
            u, f1 = combi[np.argmax(score)], max(score)
            remain = set(remain).difference(u)
    t.value = 'Complete'
    return find_condition(X, y, u, n_max=n_max)

def matplotlib_cmap(name='viridis', n=10):

    '''
    Create list of color from `matplotlib.cmap`.
    
    Parameters
    ----------
    name : `matplotlib.colors.Colormap` or str, 
    optional, (default:'viridis')
    \t The name of a colormap known to `matplotlib`. 
    
    n : `int`, optional, (defualt: 10)
    \t Number of shades for defined color map.
    
    Returns
    -------
    List of color-hex codes from defined 
    `matplotlib.colors.Colormap`. Such list contains
    `n` shades.
    '''
    c_hex = '#%02x%02x%02x'
    c = cm.get_cmap(name)(np.linspace(0,1,n))
    c = (c*255).astype(int)[:,:3]
    return [c_hex % (c[i,0],c[i,1],c[i,2]) for i in range(n)]
  
def create_cmap(c1=(23,10,8), c2=(255,255,255)):
    
    '''
    Creating `matplotlib.colors.Colormap` (Colormaps)
    with two colors.
    
    Parameters
    ----------
    c1 : hex code or (r,g,b), optional, (default:(23,10,8))
    \t The beginning color code.
    
    c2 : hex code or (r,g,b), optional, (default:(255,255,255))
    \t The ending color code.
    
    Returns
    -------
    `matplotlib.colors.ListedColormap`
    '''
    RGB = lambda c : tuple(int(c.lstrip('#')[i:i+2],16) 
                           for i in (0,2,4))
    # Convert to RGB.
    if isinstance(c1,str): c1 = RGB(c1)
    if isinstance(c2,str): c2 = RGB(c2)
    colors = np.ones((256,4))
    for i in range(3):
        colors[:,i] = np.linspace(c1[i]/256,c2[i]/256,256)
    colors = colors[np.arange(255,-1,-1),:]
    return ListedColormap(colors)

def plot_pie(ax, y, colors=None, labels=None):
    
    '''
    Plot pie chart given `y`.
    
    Parameters
    ----------
    ax : `axes.Axes` object
    \t `Axes` object from `matplotlib.pyplot.subplots`.
    
    y : 1D-array or `pd.Series`
    \t Array of labels.
    
    colors : list of color-hex codes or (r,g,b), 
    optional, (default:None)
    \t List must contain at the very least, 'n' of 
    \t color-hex codes or (r,g,b) that matches number of 
    \t clusters in 'y'. If None, the matplotlib color maps, 
    \t namely 'gist_rainbow' is used.
    
    labels : List of `str`, optional, (defualt:None)
    \t List of labels (`int`) whose items must be arranged
    \t in ascending order. If None, (n+1) cluster is 
    \t assigned, where `n` in the cluster label.
    '''
    a, sizes = np.unique(y, return_counts=True)
    if colors is None: colors = matplotlib_cmap('gist_rainbow',len(sizes))
    explode = (sizes==max(sizes)).astype(int)*0.1
    if labels is None: 
        labels = ['Group %d \n (%s)' % (m+1,'{:,d}'.format(n)) 
                  for m,n in zip(a,sizes)]
    else: labels = ['%s \n (%s)' % (m,'{:,d}'.format(n)) 
                    for m,n in zip(labels,sizes)]
    kwargs = dict(explode=explode, labels=labels, autopct='%1.1f%%', 
                  shadow=True, startangle=90, colors=colors, 
                  wedgeprops = dict(edgecolor='black'))
    ax.pie(sizes, **kwargs)
    ax.axis('equal')

class plot_radar:
    
    def __init__(self, **params):
        
        '''
        Parameters
        ----------
        **params : dictionary of properties, optional
        \t params are used to specify or override properties of 
        \t the following functions, which are;
        \t - axis.plot = {'ls':'-', 'lw':1, 'ms':3, 'marker':'o', 
        \t               'fillstyle'='full'}
        \t - axis.fill = {'alpha'=0.5}
        '''
        init = dict(d0 = dict(ls='-', lw=1, ms=3, marker='o', fillstyle='full'),
                    d1 = dict(alpha=0.5))
        for k in init.keys():
            keys = set(init[k].keys()).intersection(params)
            init[k] = {**init[k], **dict((n,params[n]) for n in keys)}
        self.params = init
        
    def __normalize(self, X, y, q=50):

        # Normalize `X`.
        features = np.array(list(X)); x = np.array(X)
        n_min, n_max = np.nanpercentile(x, q=[0,100], axis=0)
        norm = np.where((n_max-n_min)==0,1,n_max-n_min)
        norm_X = np.array((x-n_min)/norm)

        # Percentile.
        nan_pct = lambda x,q : np.nanpercentile(x,q,axis=0).ravel()
        get_pct = lambda x,y,q : [nan_pct(x[y==c],q) for c in np.unique(y)]
        
        # Add values from classes to features.
        v0, v1 = get_pct(x, y, q); s = '( {:,.4g}, {:.4g} )'
        features = ['\n'.join((f,s.format(m,n))) 
                    for f,m,n in zip(features,v0,v1)]
        
        return np.vstack(get_pct(norm_X,y,q)), np.unique(y), features
    
    def fit(self, ax, X, y, q=50, colors=None, labels=None):
        
        '''
        Parameters
        ----------
        ax : `axes.Axes` object
        \t `Axes` object from `matplotlib.pyplot.subplots`.
    
        X : `pd.dataframe` object, of shape 
        (n_samples, n_features)
        \t All elements of X must be finite, i.e. no 
        \t `np.nan` or `np.inf`.
        
        y : 1D-array or `pd.Series`
        \t Array of labels.
        
        q : `float`, optional, (default:50)
        \t Percentile to compute, which must be between 0 
        \t and 100 e.g. If `q1` is 70, that means values 
        \t (normalieze) from both classes will be 
        \t determined at 70th-percentile. Then difference 
        \t is computed from obtained values.

        colors : list of color-hex codes or (r,g,b), 
        optional, (default:None)
        \t List must contain at the very least, `n` of 
        \t color-hex codes or (r,g,b) that matches number of 
        \t labels in `y`. If `None`, the matplotlib color 
        \t maps, namely 'gist_rainbow' is used.
        
        labels : List of `str`, optional, (defualt:None)
        \t List of labels (integer) whose items must be 
        \t arranged in ascending order. If `None`, (`n`+1) 
        \t cluster is assigned, where `n` in the cluster 
        \t label.
        
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
        >>> plot_radar().fit(ax, X, y)
        '''
        # Resample and normalize X
        cnt = np.bincount(y)
        X, y, c = self.__normalize(X, y, q)
        if colors is None: colors = matplotlib_cmap('coolwarm_r',len(y))
        
        # Angle of plots
        angles = [n/float(X.shape[1])*2*np.pi for n in range(X.shape[1])]
        angles += angles[:1]

        # If you want the first axis to be on top
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)

        # draw one axe per variable + add labels 
        kwargs = dict(color='#3d3d3d', fontsize=10)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(c, **kwargs)
        
        # set alignment of ticks
        for n,t in enumerate(ax.get_xticklabels()):
            if (0<angles[n]<np.pi): t._horizontalalignment = 'left'
            elif (angles[n]>np.pi): t._horizontalalignment = 'right'
            else: t._horizontalalignment = 'center'
        
        if labels is None: labels = ['Cluster {0}'.format(n+1) for n in y]
        for n,k in enumerate(y):
            values = X[k,:].tolist() + [X[k,0]]
            kwargs = dict(c=colors[n])
            ax.plot(angles, values, **{**self.params['d0'],**kwargs})
            kwargs = dict(c=colors[n], label=labels[n] + ' ({:,d})'.format(cnt[n]))
            ax.fill(angles, values, **{**self.params['d1'],**kwargs})
        
        ax.set_yticklabels([])
        kwargs = dict(facecolor='none', edgecolor='none',
                      fontsize=10, bbox_to_anchor=(0,0))
        ax.legend(**kwargs)
        ax.set_facecolor('White')
        ax.grid(True, color='grey', lw=0.5, ls='--')

def diff_normalize(X, y, q1=50, q2=50):
      
    '''
    Find the difference between normalized `normal` 
    and `anomaly` by features.
    
    Parameters
    ----------
    X : `pd.dataframe` object, of shape 
    (n_samples, n_features)
    \t All elements of X must be finite, i.e. no 
    \t `np.nan` or `np.inf`.

    y : 1D-array or `pd.Series`
    \t Array of labels.

    q1 : `float`, optional, (default:50)
    \t Percentile to compute, which must be between 0 
    \t and 100 e.g. If `q1` is 70, that means values 
    \t (normalieze) from both classes will be 
    \t determined at 70th-percentile. Then difference 
    \t is computed from obtained values.

    q2 : `float`, optional, (default:50)
    \t Percentile of difference cutoff, which must be 
    \t between 0 and 100.
    
    Returns
    -------
    `diff` : Array of `float`, of shape (n_features,4) 
    \t 4 columns are  `difference`, `q1`-percentile of 
    \t class `0`, `q1`-percentile of class `1`, index
    \t of features according to `X`.
    
    `cutoff` : `float`
    \t `cutoff` that is computed from `q2`.
    '''
    # Normalize `X`.
    features = np.array(list(X)); x = np.array(X)
    n_min, n_max = np.nanpercentile(x, q=[0,100], axis=0)
    norm = np.where((n_max-n_min)==0,1,n_max-n_min)
    norm_X = np.array((x-n_min)/norm)
  
    # Percentile (`q1`).
    nan_pct = lambda x,q : np.nanpercentile(x,q,axis=0).ravel()
    get_pct = lambda x,y,q : [nan_pct(x[y==c],q) for c in np.unique(y)]
    v0, v1 = get_pct(norm_X, y, q1)
  
    # Determine absolute differences and sort them accordingly.
    diff = [n for n in zip(abs(v1-v0), v0, v1, range(len(v0)))]
    diff.sort(reverse=True, key=lambda x: x[0])
    cutoff = np.nanpercentile(np.array(diff)[:,0], q=q2)

    return np.array(diff), cutoff

def plot_barh(ax, x, y, c):
    
    '''
    Plot horizontal bar of `difference` between 
    `normal` and `anomaly`. This can be directly from
    `diff_normalize`.
    
    Parameters
    ----------
    ax : `axes.Axes` object
    \t `Axes` object from `matplotlib.pyplot.subplots`.
    
    x : list or array of `str`, of shape (n_features,)
    \t List of `feature`.
        
    y : 1D-array or `pd.Series`, of shape (n_features,)
    \t Array of `difference` (see `diff_normalize`).
    
    c : `float`
    \t `cutoff` does not necessarily have to be from 
    \t `diff_normalize`. It can be any number deemed
    \t appropriate.
    '''
    # Horizontal bar.
    kw = dict(alpha=0.7, ec='#2f3542', color='#009432', height=0.7)
    ax.barh(range(len(y)), np.where(y<c,np.nan,y), **kw)
    kw = dict(alpha=0.7, ec='#2f3542', color='red', height=0.7)
    ax.barh(range(len(y)), np.where(y>=c,np.nan,y), **kw)
    ax.set_yticks(range(len(y)))
    ax.set_yticklabels(x, color='#2f3640', fontsize=10)
    
    # Annotate horizontal bars.
    for (dx,dy) in enumerate(y):
        kw = dict(xytext=(2,0), textcoords='offset points', 
                  va='center' , ha='left', color='#2f3640')
        ax.annotate('{:,.3g}'.format(dy), (dy,dx), **kw)
    
    # Set `title`, `xlabel`, and `axvline`.
    kw = dict(fontsize=12, color='#2f3542', fontweight='bold')
    ax.set_xlabel('Difference of medain of normalized Xs', **kw)
    ax.axvline(c, lw=1, ls='--', color='grey')
    ax.set_title('Distinctive Characteristics ({:,.3g})'.format((y>=c).sum()), **kw)
    
    # Annotation of `cutoff`.
    kw = dict(xytext=(2,0), textcoords='offset points', va='center' , 
              ha='left', color='#2f3640', fontsize=12)
    ax.annotate('← cutoff = {:,.3g}'.format(c), (c,len(x)), **kw)
    
    # Set `ylimit` and `xlimit`.
    ax.set_ylim(-0.5, len(x)+0.5)
    ax.set_xlim(0, min(ax.get_xlim()[1]*1.1,1.05))
    ax.invert_yaxis()

def describe_y(X, y, func):
    
    '''
    Parameters
    ----------
    X : `pd.dataframe` object, of shape 
    (n_samples, n_features)
    \t All elements of X must be finite, i.e. no 
    \t `np.nan` or `np.inf`.

    y : 1D-array or `pd.Series`
    \t Array of labels.
    
    func : list of `function`
    \t List of functions to use for aggregating the 
    \t data. The applicable function can be either 
    \t `numpy` or other functions that implement the 
    \t same interface e.g. [np.sum, np.mean].
    '''
    # Define functions.
    cols = lambda x : ['feature', 'flag'] + np.array(list(x))[:,1].tolist()
    data = lambda x,s : np.hstack((np.full((2,1),s),x.reset_index().values))
    to_df = lambda x,s : pd.DataFrame(data(x,s),columns=cols(x))
    concat = lambda a,b : pd.concat((a.copy(),pd.Series(b, name='flag')),axis=1)
    agg_x = lambda x,s,f : x[[s,'flag']].groupby('flag').agg(f)
    
    x , k = concat(X,y), None
    for d in [to_df(agg_x(x,s,func),s) for s in list(X)]:
        if k is None: k = d
        else: k = k.append(d,ignore_index=False)
    return k.set_index(['feature', 'flag'])

class outliers:
  
    '''
    ** Capping Outliers **
    Each of approaches fundamentally predetermines lower and 
    upper bounds where any point that lies either below or above
    those points is identified as outlier. Once identified, such 
    outlier is then capped at a certain value above the upper 
    bound or floored below the lower bound.

    1) Percentile : (α, 100-α)
    2) Sigma : (average - β * σ , average + β * σ )  
    3) Interquartile Range (IQR) : (Q1 - β * IQR, Q3 + β * IQR)
    4) Gamma : (min(IQR, γ), max(IQR, γ))
    \t This approach determines rate of change (r) and locates 
    \t cut-off (different α at both ends) where "r" is found to 
    \t be the highest within given range. Nevertheless, this may 
    \t produce an unusally big "α" causing more to be outliers, 
    \t especially when range is considerably large. Therefore, 
    \t outputs are then compared against results from 
    \t "Interquartile Range" as follow.
    '''
    
    def __init__(self, method='gamma', pct_alpha=1.0, beta_sigma=3.0, beta_iqr=1.5, 
                 pct_limit=10, n_interval=100):
        '''
        Parameters
        ----------
        method : `str`, optional, (default: 'gamma')
        \t Method of cappling outliers i.e. 'percentile', 
        \t 'interquartile', 'sigma', and 'gamma'

        pct_alpha : `float`, optional, (default: 1.0)
        \t α (`pct_alpha`) refers to the likelihood that 
        \t the population lies outside the confidence interval. 
        \t α is usually expressed as a proportion or in 
        \t this case percentile e.g. 1 (1%).

        beta_sigma : `float`, optional, (default: 3.0)
        \t β (`beta_sigma`) is a constant that refers to 
        \t amount of standard deviations away from its mean 
        \t of the population.

        beta_iqr : `float`, optional, (default: 1.5)
        \t β (`beta_iqr`) is a muliplier of IQR

        pct_limit : `float`, optional, (default: 10)
        \t This only applies when "gamma" is selected. It 
        \t limits the boundary of both tails for gammas to be 
        \t calculated.

        n_interval : `int`, optional, (default: 100)
        \t This only applies when "gamma" is selected. It 
        \t refers to number of intervals where gammas are 
        \t calculated.
        '''
        self.method = method
        self.pct_alpha = pct_alpha
        self.beta_sigma = beta_sigma
        self.beta_iqr = beta_iqr
        
        self.n_interval = n_interval
        # index that devides left and right tails (middle)
        self.mid_index = int(n_interval*0.5)
        self.low_pct = int(n_interval*pct_limit/100)
        
    def __to_array(self, X):
    
        if isinstance(X, pd.Series):
            return [X.name], np.array(a).reshape(-1,1)
        elif isinstance(X, pd.DataFrame):
            return list(X), X.values
        elif isinstance(X, np.ndarray) & (X.size==len(X)):
            return ['Unnamed'], np.array(X).reshape(-1,1)
        elif isinstance(X, np.ndarray) & (X.size!=len(X)):
            digit = 10**math.ceil(np.log(X.shape[1])/np.log(10))
            columns = ['Unnamed: '+str(digit+n)[1:] for n in range(X.shape[1])]
            return columns, X
    
    def __iqr_cap(self, a):
        
        b = self.beta_iqr
        nanpct = lambda x : np.nanpercentile(x,[25, 75])
        iqr = lambda x,b : b*float(np.diff(nanpct(x)))
        q1,q3 = np.nanpercentile(a,[25, 75])
        return q1-iqr(a,b), q3+iqr(a,b)
  
    def __sigma_cap(self, a):
        
        b = self.beta_sigma
        sigma = lambda x,s,c : np.nanmean(x) + s*np.nanstd(x)*c
        return sigma(a,-1,c), sigma(a,1,c)
  
    def __pct_cap(self, a):
        b = self.pct_alpha
        return np.nanpercentile(a,[b,100-b])
  
    def __delta_gamma(self, X, delta_asec=True, gamma_asec=True):

        '''
        Determine change (delta), and rate of change (gamma)
        
        Parameters
        ----------
        X : 1D-array
        \t An array, whose members are arrange in 
        \t monotonically increasing manner. 
        
        delta_asec : `bool`, optional, (default: True)
        \t If True, deltas are ascendingly ordered. 
        
        gamma_asec : `bool`, optional, (default: True)
        \t If `True`, gammas are ascendingly ordered.
        '''
        # Slope (1st derivative, delta)
        diff_X = np.diff(X)
        divisor = abs(X.copy()); divisor[divisor==0] = 1
        if delta_asec: delta = diff_X/divisor[:len(diff_X)]
        else: delta = diff_X/-divisor[1:]

        # Change in slope (2nd derivative, gamma)
        diff_del = np.diff(delta)
        divisor = abs(delta); divisor[divisor==0] = 1
        if gamma_asec: gamma = diff_del/divisor[:len(diff_del)]
        else: gamma = diff_del/-divisor[1:]
       
        return delta, gamma
  
    def __gamma_cap(self, X):

        # Create range of percentiles
        p_range = np.arange(self.n_interval+1)/self.n_interval*100
        a = np.array([np.nanpercentile(X,p) for p in p_range])
        r = self.__iqr_cap(X)

        # Low side delta and gamma. Gamma is arranged in reversed order 
        # as change is determined towards the lower number (right to left)
        gamma = self.__delta_gamma(a, gamma_asec=False)[1]
        chg_rate = gamma[:(self.mid_index-1)]
      
        # Low cut-off and index of maximum change (one before)
        min_index = np.argmax(chg_rate[:self.low_pct]) + 1
        low_cut = min_index/self.n_interval*100 # convert to percentile
        low = min(np.percentile(a, low_cut), r[0])

        # Recalculate for high-side delta and gamma (ascending order)
        gamma = self.__delta_gamma(a)[1] 
        chg_rate = gamma[self.mid_index:]
        
        # High cut-off and index of maximum change (one before)
        max_index = np.argmax(chg_rate[-self.low_pct:])-1
        max_index = self.mid_index + max_index - self.low_pct
        high_cut = (max_index/self.n_interval*100)+50 # convert to percentile
        high = max(np.percentile(a, high_cut), r[1])
        
        return low, high
  
    def fit(self, X):
        
        '''
        Parameters
        ----------
        X : array-like, of shape (n_samples, n_features)
        \t Sample data.
        
        Returns
        -------
        self.limit_ : `dict` object, shape of (n_features, 3)
        \t `dict` of lower and upper limits for all variables.
        
        self.capped_X : `dict` object, shape of (n_samples, n_features)
        \t `dict` object of capped variables. 
        '''
        columns, a = self.__to_array(X); c = [None]*len(columns)
        self.limit_ = dict(variable=columns, lower=c.copy(), upper=c.copy())
        get_x = lambda x,n : x[:,n][~np.isnan(x[:,n])]
        compare = lambda r,x : (max(r[0], min(x)),min(r[1], max(x)))
        
        for n in range(a.shape[1]):
            k = get_x(a,n)
            if self.method == 'gamma': r = self.__gamma_cap(k)   
            elif self.method == 'interquartile': r = self.__iqr_cap(k)
            elif self.method == 'sigma': r = self.__sigma_cap(k)
            elif self.method == 'percentile': r = self.__pct_cap(k)
                
            # Compare against actual data.
            low, high = compare(r,k)
            # Replace data in array.
            k[(k>high)], k[(k<low)] = high, low
            a[:,n][~np.isnan(a[:,n])] = k
            self.limit_['lower'][n] = low
            self.limit_['upper'][n] = high
         
        self.capped_X = pd.DataFrame(a,columns=columns).to_dict(orient='list')
