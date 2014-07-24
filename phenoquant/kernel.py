# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 13:06:06 2014

These kernel functions are translations of code by:

Allan Aasbjerg Nielsen
aa@space.dtu.dk, www.imm.dtu.dk/~aa

Therefore all credit should be directed to him and appropriate papers 
should be cited when using these for academic publications.

@author: schackv

"""

import numpy as np

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
    
    
def kernelize(Xtrain,Xtest,kernel='Gaussian',scale=None):
    """Kernelize the observations in Xtest using the observations in Xtrain.
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
    Xtest = np.matrix(Xtest)
    
    if kernel == 'linear':
        K = Xtrain * Xtest.T
        return K
    
    ntrain = Xtrain.shape[0]
    ntest = Xtest.shape[0]
    Xtrainsum = np.sum(np.power(Xtrain,2),axis=1)    # Squared row sums
    Xtestsum = np.sum(np.power(Xtest,2),axis=1)      # Squared test sums
    K = - 2*Xtrain*Xtest.T + Xtestsum.T + Xtrainsum
    
    if scale is None:
        scale = np.nansum(np.sqrt(K[K>0]))/(ntest*(ntrain-1))
    
    if kernel == 'Gaussian':
        K = np.exp(-K/(2*scale**2))
    elif kernel == 'multiquadric':
        K = np.sqrt(K + scale**2)
    elif kernel == 'inv multiquadric':
        K = 1/np.sqrt(K + scale**2)
    else:
        raise NameError(kernel)
    
    return K
    
        
    
    
    


