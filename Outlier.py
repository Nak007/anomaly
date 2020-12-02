'''
Available methods are the followings:
[1] outliers
[2] ...

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 02-09-2020
'''
import pandas as pd, numpy as np
from Utils import _to_DataFrame
from warnings import warn
import numbers

class outliers:
      
    '''
    ** Capping Outliers **
    
    Each of approaches fundamentally predetermines lower and upper bounds 
    where any point that lies either below or above those points is identified 
    as outlier. Once identified, such outlier is then capped at a certain 
    value above the upper bound or floored below the lower bound.

    1) Percentile : (α, 100-α)
    2) Sigma : (average - β * σ , average + β * σ )
    3) Interquartile Range (IQR) : (Q1 - β * IQR, Q3 + β * IQR)
    4) Gamma : (min(IQR, γ), max(IQR, γ))
    \t This approach determines rate of change (γ) and locates cut-off 
    \t (different α at both ends) where "γ" is found to be the highest within 
    \t given range. Nevertheless, this may produce an unusally big "α" causing 
    \t more to be outliers, especially when range is considerably large. 
    \t Therefore, outputs are then compared against results from "Interquartile 
    \t Range".
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    method : `str`, optional, (default: 'gamma')
    \t Method of capping outliers i.e. 'percentile', 'interquartile', 'sigma', 
    \t and 'gamma'.

    pct_alpha : `float`, optional, (default: 1.0)
    \t α (`pct_alpha`) refers to the likelihood that the population lies outside 
    \t the confidence interval. α is usually expressed as a proportion or in this 
    \t case percentile e.g. 1 (1%).

    beta_sigma : `float`, optional, (default: 3.0)
    \t β (`beta_sigma`) is a constant that refers to amount of standard deviations 
    \t away from its mean of the population.

    beta_iqr : `float`, optional, (default: 1.5)
    \t β (`beta_iqr`) is a muliplier of IQR (InterQuartile Range).

    pct_limit : `float`, optional, (default: 10)
    \t This only applies when "gamma" is selected. It limits the boundary of both 
    \t tails for gammas to  be calculated.

    n_interval : `int`, optional, (default: 100)
    \t This only applies when "gamma" is selected. It refers to number of intervals 
    \t where gammas are calculated.

    Examples
    --------
    >>> from CappingOutlier import outliers
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> model = outliers(method='gamma')
    >>> model.fit(X)
    
    # DataFrame of capped X.
    >>> pd.DataFrame(model.capped_X)
    
    # Dictionary of lower and upper limits.
    >>> model.limit_
    '''
    def __init__(self, method='gamma', pct_alpha=1.0, beta_sigma=3.0, 
                 beta_iqr=1.5, pct_limit=10, n_interval=100):
        
        self.method = method
        self.pct_alpha = pct_alpha
        self.beta_sigma = beta_sigma
        self.beta_iqr = beta_iqr
        
        # Variables for "gamma".
        self.n_interval = n_interval
        self.pct_limit = pct_limit
        
        # Index that devides left and right tails (middle)
        self.mid_index = int(n_interval*0.5)
        self.low_pct = int(n_interval*pct_limit/100)
        self.incr = 100/n_interval
     
    def __iqr_cap(self, a):
        '''
        Returns lower and upper bounds from sample median 
        (interquartile range).
        '''
        nanpct = lambda x : np.nanpercentile(x,[25,75])
        iqr = lambda x,b : np.array([b*np.diff(nanpct(x))[0]]*2)
        q = nanpct(a) + iqr(a,self.beta_iqr) * np.array([-1,1])
        return q.tolist()
    
    def __sigma_cap(self, a):
        '''
        Returns lower and upper bounds from sample mean 
        (standard deviation).
        '''
        sigma = lambda x,c : np.nanmean(x) + np.nanstd(x)*c
        return [sigma(a,n*self.beta_sigma) for n in [-1,1]]
  
    def __pct_cap(self, a):
        '''
        Returns lower and upper bounds from sample median
        (percentile).
        '''
        return np.nanpercentile(a,[self.pct_alpha,100-self.pct_alpha])
    
    def __delta_gamma(self, X, delta_asec=True, gamma_asec=True):

        '''
        Determine change (delta,δ), and rate of change (gamma,γ).
        
        Parameters
        ----------
        X : 1D-array
        \t An array, whose members are arranged in 
        \t monotonically increasing manner. 
        
        delta_asec : `bool`, optional, (default: True)
        \t If True, deltas are ascendingly ordered. 
        
        gamma_asec : `bool`, optional, (default: True)
        \t If `True`, gammas are ascendingly ordered.
        
        Returns
        -------
        delta : `list` object
        gamma : `list` object
        '''
        # Slope (1st derivative, delta)
        diff_X = np.diff(X)
        divisor = abs(X.copy())
        divisor[divisor==0] = 1
        if delta_asec: delta = diff_X/divisor[:len(diff_X)]
        else: delta = diff_X/-divisor[1:]

        # Change in slope (2nd derivative, gamma)
        diff_del = np.diff(delta)
        divisor = abs(delta)
        divisor[divisor==0] = 1
        if gamma_asec: gamma = diff_del/divisor[:len(diff_del)]
        else: gamma = diff_del/-divisor[1:]
        return delta, gamma
  
    def __gamma_cap(self, X):
        '''
        Returns lower and upper bounds from sample median.
        '''
        # Create range of percentiles.
        a = np.nanpercentile(X,q=np.arange(0,100+self.incr,self.incr))
        r = self.__iqr_cap(X)

        # Low side delta and gamma. Gamma is arranged in reversed order 
        # as change is determined towards the lower number (right to left).
        gamma = self.__delta_gamma(a, gamma_asec=False)[1]
        chg_rate = gamma[:(self.mid_index-1)]
      
        # Low cut-off and index of maximum change (one before).
        min_index = np.argmax(chg_rate[:self.low_pct]) + 1
        
        # Convert to percentile.
        low_cut = min_index/self.n_interval*100 
        low = min(np.percentile(a, low_cut), r[0])

        # Recalculate for high-side delta and gamma (ascending order)
        chg_rate = self.__delta_gamma(a)[1] 
        chg_rate = gamma[self.mid_index:]
        
        # High cut-off and index of maximum change (one before).
        max_index = np.argmax(chg_rate[-self.low_pct:])-1
        max_index = self.mid_index + max_index - self.low_pct
        
        # Convert to percentile.
        high_cut = (max_index/self.n_interval*100) + 50 
        high = max(np.percentile(a, high_cut), r[1])
        
        return low, high
    
    def _check_method(self):
        '''
        Check method of capping outliers.
        '''
        all_methods = ['gamma', 'interquartile', 'sigma', 'percentile']
        if self.method not in all_methods:
            raise ValueError("outliers only supports one of the following " 
                             "methods i.e. 'gamma', 'interquartile', 'sigma'"
                             ", and 'percentile', got %s" % self.method)
    
    def _check_values(self):
        
        def is_num(num):
            return isinstance(num, numbers.Number)
        
        if not is_num(self.pct_alpha) or not (0<self.pct_alpha<50):
            raise ValueError("α (alpha) for percentile must be in "
                             "the range [1, 49]; got {:.4g}".
                             format(self.pct_alpha))
                             
        if not is_num(self.beta_sigma) or not (0<self.beta_sigma):
            raise ValueError("β (beta) is a multiplier of standard "
                             "deviation (σ) and must be greater than "
                             "0; got {:.4g}".format(self.beta_sigma))
        
        if not is_num(self.beta_iqr) or not (0<self.beta_iqr):
            raise ValueError("β (beta) is a multiplier of Interquartile "
                             "Range (IQR) and must be greater than 0; "
                             "got {:.4g}".format(self.beta_iqr))
        
        if not is_num(self.n_interval) or not (50<self.n_interval):
            raise ValueError("n_interval must be greater than 50; got "
                             "{:.4g}".format(self.n_interval))
        
        if not is_num(self.pct_limit) or not (0<self.pct_limit<50):
            raise ValueError("pct_limit must be in the range [1, 49]; "
                             "got {:.4g}".format(self.pct_limit))
  
    def fit(self, X):
        
        '''
        Fits the model to the dataset `X` and returns capped
        `X` and its limits.
        
        Parameters
        ----------
        X : `array-like` or `pd.DataFrame` object
        \t Sample data.
        
        Returns
        -------
        self.limit_ : `dict` object
        \t `dict` of lower and upper limits for all variables.
        
        self.capped_X : `dict` object
        \t `dict` object of capped variables. 
        '''
        # Validate Input.
        self._check_method()
        self._check_values()
        
        # Convert `X` to `pd.DataFrame`
        df = _to_DataFrame(X).copy()
        a, columns = df.values, df.columns
        self.limit_ = dict(variable=columns, 
                           lower=[None]*df.shape[1], 
                           upper=[None]*df.shape[1])
        
        get_x = lambda x,n : x[:,n][~np.isnan(x[:,n])]
        compare = lambda r,x : (max(r[0], min(x)), min(r[1], max(x)))
        
        for n in range(a.shape[1]):
            k = get_x(a,n)
            if self.method == 'gamma': 
                r = self.__gamma_cap(k)   
            elif self.method == 'interquartile': 
                r = self.__iqr_cap(k)
            elif self.method == 'sigma': 
                r = self.__sigma_cap(k)
            elif self.method == 'percentile': 
                r = self.__pct_cap(k)

            # Compare against actual data.
            low, high = compare(r,k)
            
            # Replace data in array.
            k[(k>high)], k[(k<low)] = high, low
            a[:,n][~np.isnan(a[:,n])] = k
            self.limit_['lower'][n] = low
            self.limit_['upper'][n] = high
         
        self.capped_X = pd.DataFrame(a,columns=columns).to_dict(orient='list')
