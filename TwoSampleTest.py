'''
Available methods are the followings:
[1] t_test
[2] chi_square_test
[3] ks_test
[4] qq_plot
[5] chi_contingency_test
Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
'''
import numpy as np
from scipy.stats import t, chi2, ks_2samp

def t_test(x1, x2):

    '''
    Assuming unequal variances, Two-sample t-test, whose
    null hypothesis or `H0` is μ1 = μ2 and alternative 
    hypothesis `HA` is μ1 ≠ μ2.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    x1, x2 : array-like (1-dimensional) of `float`
    \t Input data that assumed to be drawn from a continuous 
    \t distribution. Sample sizes can be different.
    
    Returns
    -------
    t_stat : `float` 
    \t `t-statistic` is used to determine whether to 
    \t accept or reject to accept the null hypothesis.
    
    p : `float`
    \t `p` is a one-tailed p-value that corresponds to 
    \t `t_stat`. We accept the null hypothesis (μ1 = μ2) 
    \t when `p` ≤ α/2 (rejection region), otherwise we 
    \t reject to accept the null hypothesis (μ1 ≠ μ2).

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    
    # Compare two different distributions (μ1, μ2).
    >>> d1 = np.random.normal(loc=0.2, size=500)
    >>> d2 = np.random.normal(loc=0.3, size=200)
    >>> ttest(d1,d2)
    (2.23103..., 0.0134..)
    
    # If α/2 = 0.005 (0.5%), then we accept the null
    # hypothesis that μ1 = μ2.
    
    # Histogram of two distributions.
    >>> plt.hist(d1, bins=20, alpha=0.5, label='X1')
    >>> plt.hist(d2, bins=20, alpha=0.5, label='X2')
    >>> plt.legend(loc='best')
    >>> plt.show()
    '''
    # (1) Standard errors.
    se = lambda x : np.nanstd(x,ddof=1)/np.sqrt(len(x))
    # (2) Standard errors of two distributions.
    sed = lambda x1,x2 : np.sqrt(se(x1)**2 + se(x2)**2)
    # (3) When `x1` and `x2` are constant, `sed` is 0.
    #     This also results t-statistic to be 0.
    mu = lambda x : np.nanmean(x)
    tstat = lambda x1,x2,s : (mu(x1)-mu(x2))/s if s>0 else 0
    
    # Calculate t-statistic.
    t_stat = tstat(x1,x2,sed(x1,x2)) 
    # Calculate degree of freedom.
    a = np.array([se(x1)**2/len(x1), se(x2)**2/len(x2)])
    b = np.array([1/(len(x1)-1), 1/(len(x2)-1)])
    if a.sum()>0: df = np.floor(a.sum()/(a*b).sum())
    else: df = len(x1) + len(x2) - 2
    return abs(t_stat), 1-t.cdf(abs(t_stat), df)

def chi_square_test(observed, expected, step=10):

    '''
    Chi-Square (χ2) is used to test whether sample data 
    fits a distribution from a certain population or 
    not. Its null hypothesis or `H0` says that the 
    observed population fits the expected population.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    observed : array-like (1-dimensional) of `float`
    \t Input data of observed samples.
    
    expected : array-like (1-dimensional) of `float`
    \t Input data of expected samples.

    step : `int`, optional, (default:10)
    \t Spacing of percentile values. This is the distance 
    \t between two adjacent values. `step` is applied only
    \t to `expected`, while the beginning and ending bin
    \t edges are replaced with `-np.inf` and `np.inf`,
    \t respectively.
    
    Returns
    -------
    cv : `float`
    \t `cv` is a critival value. If the critical value
    \t from the table given `degrees of freedom` and `α` 
    \t (rejection region) is less than the computed 
    \t critical value (`cv`), then the observed data does
    \t not fit the expected population or in another word, 
    \t we reject to accept the null hypothesis.
    
    p : `float`
    \t `p` is a p-value that corresponds to `cv`.
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    
    # Compare observed and expected distributions.
    >>> expected = np.random.normal(loc=0.2, size=500)
    >>> observed = np.random.normal(loc=0.3, size=200)
    >>> chi_square(observed, expected)
    (15.88, 0.06943018381071808)
    
    # If α = 0.05 (5%), then we accept the null hypothesis that
    # observed sample fits the expected population.
    
    # Histogram of two distributions.
    >>> plt.hist(expected, bins=20, alpha=0.5, label='Expected')
    >>> plt.hist(observed, bins=20, alpha=0.5, label='Observed')
    >>> plt.legend(loc='best')
    >>> plt.show()
    '''
    # Determine BIN of `expect`.
    hist = lambda x,b : np.histogram(x,b)[0].ravel()
    pct = [min(n,100) for n in np.arange(0,100+step,step)]
    bins = np.unique(np.nanpercentile(expected, pct))
    bins[0] = -np.inf; bins[-1] = np.inf
    
    # Count of observed and expected distributions.
    obs = hist(observed, bins)
    factor = len(observed)/len(expected)
    exp = hist(expected, bins)*factor

    # Degrees of freedom.
    df = max(len(exp)-1,1)
    # Critical value
    cv = ((obs-exp)**2/np.where(exp==0,1,exp)).sum()
    return cv, 1-chi2.cdf(cv, df=df)

def ks_test(a, b, return_dist=False):
    
    '''
    The two-sample Kolmogorov-Smirnov test is a general 
    nonparametric method for comparing two distributions by 
    determining the maximum distance from the cumulative 
    distributions, whose function (`s`) can be expressed as: 
    
                      s(x,m) = f(m,x)/n(m)
    
    where `f(m,x)` is a cumulative frequency of distribution 
    `m` given `x` and `n(m)` is a number of samples of `m`.
    The Kolmogorov–Smirnov statistic for two given cumulative 
    distribution function, `a` and `b` is:
    
                   D(a,b) = max|s(x,a) - s(x,b)|
                
    where a ∪ b = {x: x ∈ a or x ∈ b}. The null hypothesis or 
    `H0` says that both independent samples have the same  
    distribution.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    a, b : array-like (1-dimensional) of `float`
    \t Input data that assumed to be drawn from a continuous 
    \t distribution. Sample sizes can be different.
    
    return_dist : `bool`, optional, default: False
    \t If `True`, also return list of cumulative distribution
    \t of `a` and `b`, respectively.
    
    Returns
    -------
    statistic : `float`
    \t Kolmogorov-Smirnov statistic (critical value).
    
    p : `float`
    \t Two-tailed p-value  that corresponds to `statistic`.
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    
    >>> d1 = np.random.normal(loc=0.2, size=501)
    >>> d2 = np.random.normal(loc=0.5, size=223)
    >>> ks_test(d1, d2)
    (0.1656..., 0.00036...)
    
    # Kolmogorov-Smirnov plot.
    >>> x1, x2 = ks_test(d1,d2, return_dist=True)[-1]
    >>> plt.plot(np.arange(len(x1)),x1, label='X1')
    >>> plt.plot(np.arange(len(x2)),x2, label='X2')
    >>> plt.axvline(np.argmax(abs(x1-x2)), color='k', 
    ...             ls='--', label='max distance')
    >>> plt.ylabel('Cumulative Probability')
    >>> plt.legend(loc='best')
    >>> plt.show()
    '''
    redim = lambda x : np.array(x).ravel()
    x = np.hstack(tuple(redim(x) for x in [a,b])).ravel()
    bins = np.unique(x); bins[-1] = np.inf
    hist = lambda x,b : np.cumsum(np.histogram(x,b)[0]/len(x)).reshape(-1,1)
    dist = [hist(data,bins) for data in [a,b]]
    statistic = float(max(abs(np.diff(np.hstack(dist), axis=1))))
    p = ks_2samp(a, b)[1]
    if return_dist: return statistic, p, [n.ravel() for n in dist]
    else: return statistic, p

def qq_plot(x, y, q=None, return_xy=False):
    
    '''
    Q–Q (quantile-quantile) plot is a probability plot, which 
    is a graphical method for comparing two distributions by 
    plotting their quantiles against each other. 
    
    If two distributions are linearly related with reasonable 
    level of `r_square` from best-fitting line, we can assume 
    that their distributions are similar. However, visualizing
    a Q-Q plot is highly recommended as indicators can 
    sometimes be inadequate to conclude "goodness of fit" of 
    both distributions.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    x : array-like (1-dimensional) of `float`
    \t Input data of independent variable.
    
    y : array-like (1-dimensional) of `float`
    \t Input data of dependent variable.
    
    q : list of `float`, optional, (defualt:None)
    \t Percentile or sequence of percentiles to compute, 
    \t which must be between 0 and 100. If `None`, a range of
    \t percentile from 0th to 100th with 1-percentile increment.
    
    return_xy : `bool`, optional, (default:False)
    \t If `True`, also return list of `x` and `y` at specified
    \t quantiles.
    
    Returns
    -------
    intercpet : `float`
    \t The intercept of best-fitting line, where `x` = 0.
    
    slope : `float`
    \t The constant, `β` that represents the linear relationship 
    \t between `y` and `x`.
    
    r_square : `float`
    \t It is a statistical measure that represents the proportion 
    \t of the variance for a dependent variable that's explained by 
    \t an independent variable in a regression model.
    
    quantile_xy : list of `x`, and `y`
    \t List of `x` and `y` at specified quantiles. Only provided if 
    \t `return_xy` is True.
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    
    >>> d1 = np.random.normal(loc=2, size=500)
    >>> d2 = np.random.normal(loc=1, size=200)
    >>> kwargs = dict(marker='o', s=50, color='k',
    ...               ec='k', lw=1, alpha=0.3)
    
    >>> a = qq_plot(d1, d2, return_xy=True)
    >>> print(a[ :-1])
    (-0.9065..., 0.9855..., 0.9956...)
    
    # Q-Q plot.
    >>> x, y = a[-1]
    >>> plt.scatter(x, y, **kwargs)
    >>> plt.grid(True, ls='--')
    >>> plt.show()
    
    # Histogram of two distributions.
    >>> plt.hist(d1, bins=20, alpha=0.5, label='X1')
    >>> plt.hist(d2, bins=20, alpha=0.5, label='X2')
    >>> plt.legend(loc='best')
    >>> plt.show()
    '''
    pct = lambda x,q : np.nanpercentile(x,q)
    def var(a, b=None):
        if b is not None: return ((a-b)**2/(len(a)-1)).sum()
        else: return ((a-np.nanmean(a))**2/(len(a)-1)).sum()
        
    # Create `a` and `b` based on defined quantiles.
    if q is None: q = np.arange(0,101,1)
    a, b = [pct(v,q).reshape(-1,1) for v in [x,y]]
    a = np.hstack((np.ones((len(a),1)),a))
    
    # Linear Regression.
    reg = np.linalg.inv(a.T.dot(a)).dot(a.T).dot(b)
    intercept, slope = float(reg[0]), float(reg[1])
    
    # `y_pred` and r2.
    xy = [a[:,1].ravel(), b.ravel()] 
    y_pred = xy[0]*slope + intercept
    r_square = 1 - var(y_pred,xy[1])/var(b)
    
    if return_xy: return intercept, slope, r_square, xy
    else: return intercept, slope, r_square
    
def chi_contingency_test(cont, n_min=5, step=10):
    
    '''
    ** Chi-Square and Tests of Contingency tables **
    
    Hypothesis test that is performed on contingency table is
    to determine whether there is an relationship between the 
    row and column variables or not. The expected frequency 
    for a cell in the `i` row and the `j` column is equal to:
    
                    E(i,j) = f(i) * f(j) / f
        
    where `f(i)` and `f(j)` are total number of observations 
    for the `i` row, and `j` column, respectively. `f` is the 
    total number of observations. The critical value for 
    chi-square can be computed as follows:
    
        χ2 = Σ(O(i,j)-E(i,j))^2 / E(i,j), {i ∈ R or j ∈ C}
    
    where `R` and `C` are all rows and columns.
    
    .. versionadded:: 26-08-2020

    Parameters
    ----------
    cont : `np.array` of `int` or `list` of `np.array`
    \t If `np.array` of `int` (`contingency table`) is a 
    \t frequency distribution table, whose observation in each 
    \t cell should be greater than 0. If `list`, it must 
    \t contain more than one 1D-array, which can either be
    \t same or different in length. In this case, row will be 
    \t `nth` arrays found in `cont` while columns value 
    \t intervals.
    
    n_min : `int`, optional, (default:5)
    \t Minimum number of observations in `contingency table`.
    \t Such number should be greater than 0.
    
    step : `int`, optional, (default:10)
    \t Spacing of percentile values. This is the distance 
    \t between two adjacent values. The beginning and ending 
    \t bin edges are replaced with `-np.inf` and `np.inf`, 
    \t respectively.
    
    Returns
    -------
    cv : `float`
    \t `cv` is a critival value. If the critical value from
    \t the table given `degrees of freedom` and `α` (rejection 
    \t region) is less than the computed critical value (`cv`),  
    \t it indicates that there might be association between 
    \t variables or in another word, the null hypothesis of no 
    \t relationship (independent) between variables can be 
    \t rejected.
    
    p : `float`
    \t `p` is a p-value that corresponds to `cv`.
    
    df : `int`
    \t Degrees of freedom.
    
    Examples
    --------
    >>> import nunpy as np
    # Type of Diet
    # http://onlinestatbook.com/2/chi_square/contingency.html
    >>> a = {'Cancers':[15,7], 
    ...      'Fatal Heart Disease':[24,14], 
    ...      'Non-Fatal Heart Disease':[25,8], 
    ...      'Healthy':[239,273]}
    >>> a = pd.DataFrame(a)
    >>> a.index = ['AHA','Mediterranean']
    >>> chi_contingency_test(a)
    (16.5545..., 0.000873..., 3)
    '''
    def create_bins(X, n_min=5, step=2):
        
        def merge(X):
            redim = lambda x : np.array(x).ravel()
            return np.hstack(tuple(redim(x) for x in X)).ravel()

        def pct_bins(x, step=1):
            pct = [min(n,100) for n in np.arange(0,100+step,step)]
            bins = np.unique(np.nanpercentile(x, pct))
            bins[0], bins[-1]  = np.NINF, np.inf
            return bins

        def n_pos(x, a):
            if (x==a).sum()==0: return (x<=a).sum()+1
            else: return (x<=a).sum()

        # Set initial values.
        default = [-np.inf, np.median(merge(X)), np.inf]
        bins, new_bins = pct_bins(merge(X)), [-np.inf]
        while len(bins)>2:
            C = [np.cumsum(np.histogram(x, bins)[0]) for x in X]
            n = max([n_pos(c, n_min) for c in C])
            if n < len(bins): new_bins.append(bins[n])
            else: new_bins += [np.inf]
            bins = bins[n+1:]

        # Collaspe last `bin` when "N" is less than `n_min`.
        if new_bins[-1] != np.inf: new_bins += [np.inf]
        n = sum([np.histogram(x, new_bins)[0][-1]<n_min for x in X])
        if n>0: new_bins = new_bins[:-2] + [np.inf]
        return new_bins if len(new_bins)>3 else default    

    # Create `contingency table`.
    if isinstance(cont, list):
        n_samples = sum([len(x) for x in X])
        bins = create_bins(X, n_min, step)
        cont = np.array([np.histogram(x, bins)[0] for x in X])
    elif isinstance(cont, (np.ndarray, pd.DataFrame)):
        cont = np.array(cont)
        n_samples = cont.sum(axis=0).sum()
    
    # Expected frequency.
    row = cont.sum(axis=1).reshape(-1,1)
    col = cont.sum(axis=0).reshape(1,-1)
    exp = row.dot(col)/n_samples
    cv = ((cont-exp)**2/exp).ravel().sum()
    # Degrees of freedom.
    df = max((cont.shape[0]-1)*(cont.shape[1]-1),1)
    return cv, 1-chi2.cdf(cv, df=df), df 
