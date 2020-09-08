'''
Available methods are the followings:
[1] pca
[2] ...

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 26-08-2020
'''
from sklearn.preprocessing import StandardScaler
import pandas as pd, numpy as np
from warnings import warn
from Utils import _to_DataFrame

__all__ = ['pca']

def pca(X, val_cutoff=1.0, vec_cutoff=0.5, stdev=(-3,3)):
    
    '''
    Perform `Principal Component Analysis` (PCA) to `X`.
    
    .. versionadded:: 26-08-2020
    
    Parameters
    ----------
    X : `pd.DataFrame` object
    \t Input array.
    
    val_cutoff : `float`, optional, default: 1.0
    \t Any eigen pair (`eigenvalue`,`eigenvector`), whose
    \t `eiginvalue` is greater than or eqaul to `cutoff` is 
    \t selected as one of the principal components. 
    \t The minimum number of `PC` is set to be 2.
    
    vec_cutoff : `float`, optional, default: 0.5
    \t Minimum value of eigenvectors that is used to determine 
    \t importance of variables to principal components.
    
    stdev : tuple of `float`, optional, default: (-3,3)
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
    
    eigenvalue : `list` of `float` 
    \t The corresponding `eigenvalue` is the factor by which 
    \t the eigenvector is scaled either stretched or shrinked.
    
    var : `list` of `str`
    \t List of variables that are important to principal 
    \t components.
    
    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> X = load_breast_cancer().data
    >>> pc_df, eigen_vectors, eigen_values, var = pca(X)
    '''
    # Standardized X.
    X = _to_DataFrame(X)
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
    def pc_columns(func):
        def columns(*args):
            return ['PC{}'.format(str(n).zfill(2)) 
                    for n in range(1,args[0]+1)]
        return columns
    
    @pc_columns
    def digit(n): 
        return np.ceil(np.log(n)/np.log(10)).astype(int)
    
    to_dict = lambda x : dict((n,k[1]) for n,k in enumerate(x))
    eigen_val = np.array([k[0] for k in eigen])
    eigen_vec = pd.DataFrame(to_dict(eigen))
    eigen_vec.index = list(X)
    eigen_vec.columns = digit(eigen_vec.shape[1])
    
    # Determine principal components given `val_cutoff`.
    n_pc = (eigen_val>=val_cutoff).sum()
    if n_pc<=1:
        warn("There is only one principal component, whose "
             "eigen value is greater than or equal to {:.0f}. "
             "Reducing `val_cutoff` help increase number of"
             "principal components. "
             "The minimum is default to 2.".format(val_cutoff), Warning)
        n_pc = 2
    pc = X_std.dot(eigen_vec.values[:,:n_pc])
    pc = pd.DataFrame(pc, columns=digit(pc.shape[1]))
    
    # Select vector, whose magnitude is above `vec_cutoff`.
    loadings = eigen_vec.iloc[:,:(eigen_val>=val_cutoff).sum()]
    var = list(X.columns[(abs(loadings).max(axis=1)>vec_cutoff).values])
   
    return pc, eigen_vec, eigen_val, var
