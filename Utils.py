'''
Available methods are the followings:
[1] _get_argument
[2] _to_DataFrame
[3] _is_array_like
Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 02-09-2020
'''
from inspect import signature
import pandas as pd, numpy as np, inspect
from warnings import warn

__all__ = ['_get_argument', '_is_array_like', '_to_DataFrame']

def _get_argument(func):
    
    '''
    Parameters
    ----------
    func : function object
    
    Returns
    -------
    - `array` of parameter names in required 
       arguments.
    - `dict` of parameter names in positional 
      arguments and their default value.
    '''
    # Get all parameters from `func`.
    params = signature(func).parameters.items()
        
    # Internal functions for `parameters`.
    is_empty = lambda p : p.default==inspect._empty
    to_dict = lambda p : (p[1].name, p[1].default)
        
    # Take argument(s) from `func`.
    args = np.array([k[1].name for k in params if is_empty(k[1])])
    kwgs = dict([to_dict(k) for k in params if not is_empty(k[1])])
    return args[args!='self'], kwgs

def _is_array_like(X):
    '''
    Returns whether the input is array-like.
    '''
    return (hasattr(X, 'shape') or hasattr(X, '__array__'))

def _to_DataFrame(X):
    '''
    If `X` is not `pd.DataFrame`, column(s) will be
    automatically created with "Unnamed_XX" format.
    
    Parameters
    ----------
    X : array-like or `pd.DataFrame` object
    
    Returns
    -------
    `pd.DataFrame` object.
    '''
    digit = lambda n : int(np.ceil(np.log(n)/np.log(10)))             
    if _is_array_like(X)==False:
        raise TypeError('Data must be array-like.')
    elif isinstance(X, (pd.Series, pd.DataFrame)): 
        return pd.DataFrame(X)
    elif isinstance(X, np.ndarray) & (len(X.shape)==1):
        return pd.DataFrame(X.reshape(-1,1),columns=['Unnamed'])
    elif isinstance(X, np.ndarray) & (len(X.shape)==2):
        z = digit(X.shape[1])
        columns = ['Unnamed_{}'.format(str(n).zfill(z)) 
                   for n in range(X.shape[1])]
        return pd.DataFrame(X,columns=columns)
