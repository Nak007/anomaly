'''
Available methods are the followings:
[1] find_cutoff
[2] find_condition
[3] optimize_condition

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 02-09-2020
'''
import inspect, pandas as pd, numpy as np, time
import ipywidgets as widgets
from IPython.display import display
from itertools import combinations

from Utils import _get_argument, _to_DataFrame

__all__ = ['find_cutoff', 'find_condition', 'optimize_condition']

def find_cutoff(x, y, decimal=4, n_max=10):
    
    '''
    Find `cutoff` by optimizing `F1-Score` from center of 
    distribution (median) towards both tails.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    x : `pd.Series` or array.
    \t Array of `float` without `np.nan`.
    
    y : array of `bool` or `int`
    \t Binary array, where `0` and `1` represent `normal`,
    \t and `anomalous` samples, respectively.
    
    decimal : `int`, optional, default:4
    \t Number of decimal places only for `f1_score`.
    
    n_max : `int`, optional, default:10
    \t Maximum number of `cutoff` per iteration.
    
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
    
    Examples
    --------
    # Import data and create dataframe.
    >>> from sklearn.datasets import load_breast_cancer as data
    >>> X, y = data(return_X_y=True)
    >>> X = pd.DataFrame(X, columns=[s.replace(' ','_') for 
    ...                              s in data().feature_names])
    
    # Anomaly detection.
    >>> from sklearn.ensemble import IsolationForest
    IF = IsolationForest(n_estimators=200, random_state=0, 
    ...                  behaviour='new', contamination='auto')
    >>> anomaly = np.where(IF.fit_predict(X)==-1,1,0)
    
    # Find anomaly cutoff.
    >>> find_cutoff(X['mean_radius'], anomaly)
    {'cutoff': ['≥', 20.57],
     'confusion_matrix': [503, 36, 12, 18],
     'f1_score': 0.4286}
    '''
    def cfm_(x, y, cutoffs, right_side=True):
        '''
        Returns confusion matrix elements for each
        cutoff.
        '''
        a = []
        for c in cutoffs:
            if right_side: 
                tn = (y[x< c]==0).sum()
                fn = (y[x< c]==1).sum()
                fp = (y[x>=c]==0).sum()
                tp = (y[x>=c]==1).sum()
            else:
                tn = (y[x> c]==0).sum()
                fn = (y[x> c]==1).sum()
                fp = (y[x<=c]==0).sum()
                tp = (y[x<=c]==1).sum()
            a.append([tn,fn,fp,tp])
        return np.vstack(a)
    
    def cal_f1_score(x):
        '''
        Returns f1_score ginve confusion matrix.
        '''
        r = x[:,3]/x[:,[1,3]].sum(axis=1).ravel()
        p = x[:,3]/x[:,[2,3]].sum(axis=1).ravel()
        return (2*r*p)/np.where(r+p==0,1,r+p)
    
    def get_cutoffs(x, right_side=True):
        if right_side: return np.unique(x[x>=np.median(x)])[1:-1]
        else: return np.unique(x[x<=np.median(x)])[1:-1]
    
    def is_array_like(x):
        return (hasattr(x, 'shape'),hasattr(x, '__array__'))

    if max(is_array_like(x))==False:
        raise TypeError('Data must be array-like.')
        return None
    
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
            # Confustion matrix.
            CFM = cfm_(x, y, cutoffs, right_side)
            # Calculate F1-Score for all cutoffs.
            f1 = np.round(cal_f1_score(CFM),decimal)
            # Store cutoff that returns the best F1-Score.
            retval.append(dict(cutoff=[sign[k],cutoffs[np.argmax(f1)]],
                               confusion_matrix=CFM[np.argmax(f1),:].tolist(),
                               f1_score=max(f1)))
            scores.append(max(f1))
    
    # len(scores)=0, when all values are the same (constant).
    return retval[np.argmax(scores)] if len(scores)>0 else None

def find_condition(X, y, columns=None, **params):
    
    '''
    Basing on `y`, it determines the best `cutoff` that defines 
    anomalies through use of `f1_score`. Each `cutoff` is set 
    dependently to its predecessor one after another. 
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    X : `pd.DataFrame` object
    \t `X` must be numerical and not contain `np.nan`. 

    y : array of `bool` or `int`
    \t Binary array, where `0` and `1` represent `Normal`,
    \t and `Anomalous` samples, respectively.
    
    columns : list of `str`, optional, default:None
    \t List of field names, where condition finding runs 
    \t according to the list ordering (left-to-right).
    \t If `None`, `X.columns` will be used instead.
    
    **params : dictionary of properties, optional
    \t Keyword arguments of `find_cutoff` i.e. `decimal`,
    \t and `n_max`.
    
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
    
    Examples
    --------
    # Import data and create dataframe.
    >>> from sklearn.datasets import load_breast_cancer as data
    >>> X, y = data(return_X_y=True)
    >>> X = pd.DataFrame(X, columns=[s.replace(' ','_') for 
    ...                              s in data().feature_names])
    
    # Anomaly detection.
    >>> from sklearn.ensemble import IsolationForest
    IF = IsolationForest(n_estimators=200, random_state=0, 
    ...                  behaviour='new', contamination='auto')
    >>> anomaly = np.where(IF.fit_predict(X)==-1,1,0)
    >>> columns = ['smoothness_error', 'compactness_error',
    ...            'mean_compactness']
    >>> find_condition(X, anomaly, columns)
    {'criteria': [['smoothness_error', '≥', 0.006383],
                  ['compactness_error', '≥', 0.04844],
                  ['mean_compactness', '≥', 0.2363]],
     'confusion_matrix': array([266,   8, 249,  46]),
     'f1_score': 0.2636}
    '''
    X = _to_DataFrame(X)
    
    if columns is None: columns = list(X)
    x, t = np.array(X[columns]), np.array(y)
    
    # Find keyword argument of `find_cutoff`.
    kwargs = _get_argument(find_cutoff)[1]
    params = {**kwargs,**params}
    bad_keys = set(params.keys()).difference(kwargs)
    params = dict([n for n in params.items() if n[0] not in bad_keys])  
    
    # Initialize default values.
    index, criteria, cfm = np.full(len(y), True), [], []
    
    # Find set of `cutoffs` iteratively. The algorithm
    # stops when all anomalies are separated or all
    # columns (features) are used.
    for (n,v) in enumerate(columns):
        
        # Number of anomalies must be greater than 0.
        if t[index].sum()>0:
            x, t = x[index,:], t[index]
            # Using `find_cutoff` to determine `cutoff`.
            m = find_cutoff(x[:,n], t, **params)
            
            if m is not None:
                # Store criteria and cfm.
                criteria.append([v] + m['cutoff'])
                cfm.append(m['confusion_matrix'])
                
                # Create index for successive trial.
                if m['cutoff'][0] == '≤': 
                    index = ~(x[:,n]<=m['cutoff'][1])
                else: index = ~(x[:,n]>=m['cutoff'][1])
        else: break
    
    if cfm is not None:
        # Optimize `f1-score`.
        cfm = np.array(cfm); fn = cfm[:,1]
        tp, fp = np.cumsum(cfm[:,3]), np.cumsum(cfm[:,2])
        
        # Recall, Precision, and F1-Score.
        recall, precision = tp/(tp+fn), tp/(tp+fp)
        f1_score = np.round((2*recall*precision)/
                            (recall+precision),params['decimal'])
        
        # Index of where `f1_score` is optimal.
        n = np.argmax(f1_score)
        cfm = np.hstack((cfm[n,[0,1]], fp[n], tp[n]))
        return dict(criteria=criteria[:n+1], confusion_matrix=cfm, 
                    f1_score=f1_score[n])
    else: return None

def optimize_condition(X, y, **params):

    '''
    This function finds variable that provides the highest 
    `f1_score` (see `find_condition`) in each iteration. 
    The optimization works dependently to previous iteration,
    meaning that once `variable` along with its corresponding 
    `cutoff` are determined, number of samples in `X` will be 
    reduced according to `cutoff`, and the process continues  
    until `f1_score` stops improving. This filtering is called 
    `cascading filter`.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    X : `pd.DataFrame` object
    \t `X` must be numerical and not contain `np.nan`. 

    y : array of `bool` or `int`
    \t Binary array, where `0` and `1` represent `Normal`,
    \t and `Anomalous` samples, respectively.
    
    **params : dictionary of properties, optional
    \t Keyword arguments of `find_cutoff` i.e. `decimal`,
    \t and `n_max`.

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
    >>> anomaly = np.where(IF.fit_predict(X_std)==-1,1,0)
    
    # Determine `criteria`.
    >>> optimize_condition(X, anomaly)
    '''
    X = _to_DataFrame(X)

    # Initialize widget.
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
            a = find_condition(X, y, c, **params)
            if a is not None: score.append(a['f1_score'])
                
        # If new combination's f1-score is greater than the
        # previous one, then this combination will replace its
        # predecessor.
        if max(score) > f1:
            u, f1 = combi[np.argmax(score)], max(score)
            remain = set(remain).difference(u)
    t.value = 'Complete'
    return find_condition(X, y, u, **params)
