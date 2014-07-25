# -*- coding: utf-8 -*-
"""
Functions useful for kernel space decompositions.

These kernel functions are translations of code by:

Allan Aasbjerg Nielsen
aa@space.dtu.dk, www.imm.dtu.dk/~aa

Therefore all credit should be directed to him and appropriate papers 
should be cited when using these for academic publications.

Created on Thu Jul 24 13:06:06 2014

@author: schackv

"""

import numpy as np
from scipy.spatial.distance import pdist

def center(K,Ktrain=None):
    """Center kernel with training data.
    
    If only K is given, the Shawe-Taylor way of centering is used.
    
    Returns a centered kernel matrix as a numpy.matrix
    """
    
    K = np.matrix(K)
    if Ktrain is None:
        n = K.shape[0]
        mu_row = K.mean(axis=0) # K is symmetrix, so column mean=row mean
        mu_tot = mu_row.sum()/n
        Kc = K - mu_row - mu_row.T + mu_tot
    else:
        Ktrain = np.matrix(Ktrain)
        n1, n2 = K.shape
        mu_row = Ktrain.mean(axis=1)    # Row means
        mu_tot = mu_row.sum()/n1        # Global training mean
        mu_col = K.mean(axis=0)         # Column test mean
        
        Kc = K - mu_row - mu_col + mu_tot
    
    return Kc
    
    
    
def rule_of_thumb_scale(X):
    """Estimate the scale as the average distance between observations. 
    Rows in X are observations and columns are variables."""
    
    D = pdist(X)
    scale = np.mean(D)
    return scale
    
def kernelize(Xtrain,Xtest=None,kernel='Gaussian',scale=None):
    """Kernelize the observations in Xtest using the observations in Xtrain.
    If Xtest is omitted, Xtrain will be kernelized with itself.
    The i,j element of K will be k(x_i,y_j).
    
    
    The possible values for kernel are:
        'Gaussian' (Default)    Gaussian kernel
        'linear'                Linear kernel
        'multiquadric'          Multiquadric kernel
        'inv multiquadric'      Inverse multiquadric kernel
    
    If scale==None (default) this is automatically set to the avg distance between training observations.
    
    Returns numpy.matrix K with shape (Xtrain.shape[0],Xtest.shape[0]).
    """
    Xtrain = np.matrix(Xtrain)

    # Avoid row-sums for linear kernel
    if kernel == 'linear':
        if Xtest is None: Xtest = Xtrain.view()
        K = Xtrain * Xtest.T
        return K

    if scale is None:
        scale = rule_of_thumb_scale(Xtrain)

    Xtrainsum = np.sum(np.power(Xtrain,2),axis=1)    # Squared row sums
    
    if Xtest is None:
        Xtest = Xtrain.view()       # .view() to look at same data as in Xtrain
        Xtestsum = Xtrainsum.view()
    else:
        Xtest = np.matrix(Xtest)
        Xtestsum = np.sum(np.power(Xtest,2),axis=1)      # Squared test sums
        
    K = - 2*Xtrain*Xtest.T + Xtestsum.T + Xtrainsum
        
    if kernel == 'Gaussian':
        K = np.exp(-K/(2*scale**2))
    elif kernel == 'multiquadric':
        K = np.sqrt(K + scale**2)
    elif kernel == 'inv multiquadric':
        K = 1/np.sqrt(K + scale**2)
    else:
        raise NameError(kernel)
    
    return K
    
        
    
    
    


