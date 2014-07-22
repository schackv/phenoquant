# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:49:23 2014

@author: schackv
"""

import numpy as np
from scipy.linalg import solve


def fisherlda(X,y, alpha=0, stretch=True):
    """Fisher linear discriminant analysis for two classes 
    with ridge regularization.
    
    X is a N x p numpy array of features
    
    Class one has y==1
    Class two has y==2
    
    Returns
        z   Numpy vector with projected values
        w   Numpy vector with projection direction
    """
    
    X = np.matrix(X)
    X1 = X[y==1,:]
    X2 = X[y==2,:]
    l1, nvar = X1.shape
    l2 = X2.shape[0]
    N = l1 + l2
    
    # Mean vectors and covariance matrices
    m1 = np.mean(X1,axis=0).T
    m2 = np.mean(X2,axis=0).T
    X1c = X1.T - m1
    X2c = X2.T - m2
    S_B = (m2-m1)*(m2-m1).T     # Between-class covariance
    S_W = X1c*X1c.T + X2c*X2c.T # Within-class covariance
    
    # Add regularization
    S_W = S_W + alpha*np.eye(nvar)
    
    # Solve (faster than eigenvalue solution and direction is the same)
    w = solve(S_W,(m2-m1))

    # Project onto direction    
    z = X*w
    
    # Stretch between -1 and 1
    if stretch:
        z = (z- np.mean(z[y==1])) / (np.mean(z[y==2]) - np.mean(z[y==1])) * 2 - 1
    
    
    return np.array(z), np.array(w)
    
    
