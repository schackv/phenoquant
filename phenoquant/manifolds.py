# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:49:23 2014

@author: schackv
"""

import numpy as np
from scipy.linalg import solve
import phenoquant.kernel as knl

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
        z = stretch_two_classes(z,y)
    
    
    return np.array(z), np.array(w)
    
    
def fisherkda(X,y,kernel='Gaussian',scale=None,alpha=0,regtype='identity',stretch=True):
    """Two-class Fishers discriminant analysis with kernels.
    
    Inputs:
        X           Numpy array with observations by row.
        y           Numpy vector with integers (0, 1, 2, ...). Observations with y==0 are not used for training.
        kernel      Choice of kernel. See kernel.kernelize for possible choices. Default='Gaussian'.
        scale       Scale parameter for kernel. If scale==None (default) this is automatically set to the avg distance between training observations.
        alpha       Regularization parameter
        regtype     Choose to add alpha times the 'identity' or 'kernel' matrix as regularization.
                    Default is 'identity'.
    
    Outputs:
        z           N-vector of projected values
        w           Eigenvector solution
        
    
    References
    Mika, S., & Ratsch, G. (1999). Fisher discriminant analysis with kernels. In Neural Networks for … (pp. 41–48). doi:10.1109/TCYB.2013.2273355
    Muller, K., Mika, S., & Ratsch, G. (2001). An introduction to kernel-based learning algorithms. … IEEE Transactions on, 12(2), 181–202. Retrieved from http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=914517
    Mika, S., Rätsch, G., Weston, J., & Schölkopf, B. (1999). Invariant Feature Extraction and Classification in Kernel Spaces. In NIPS (Vol. 89,
    """
    
    X = np.matrix(X)
    l1 = np.sum(y==1)
    l2 = np.sum(y==2)
    N = l1 + l2
    
    # Kernelize all with training data
    idx = y > 0     # All training ids
    Xtrain = X[idx,:]
    Ktrain, scale_train = knl.kernelize(Xtrain,Xtrain,kernel,scale)
    Ktest, _ = knl.kernelize(Xtrain,X,kernel,scale_train)

    # Center kernels with training data    
    K = knl.center(Ktrain)
    Ktest = knl.center(Ktest,Ktrain)
    Ktrain = None       # Ktrain is not used anymore
    
    # Mean vectors
    m1 = np.mean(K[:,y[idx]==1],axis=1)
    m2 = np.mean(K[:,y[idx]==2],axis=1)
    
    # Covariance matrices and regularization
    S_W = K*K.T - (l1*(m1*m1.T) + l2 * (m2*m2.T) )
    S_B = (m2-m1)*(m2-m1).T # Between-class covariance
    
    if regtype=='identity':
        S_W = S_W + alpha*np.eye(N) # Regularize with identity matrix (penalize ||w||^2)
    else:
        S_W = S_W + alpha*K         #  Regularize with kernel matrix (penalize ||a||^2)
        
    # Solve evp
    aux = solve(S_W,S_B)
    eigvals, V = np.linalg.eig(aux)
    sortidx = eigvals.argsort()[::-1]
    eigvals = np.real(eigvals[sortidx])
    V = np.real(V[:,sortidx])
    w = V[:,0][:,np.newaxis]     # First eigenvector
    
    # Project training observations on w
    z = Ktest.T * w
    
    if stretch:
        z = stretch_two_classes(z,y)
    
    return np.array(z), np.array(w)
    
    
    
def stretch_two_classes(x,grp):
    """Stretch the values of x, such that 
    mean(x[grp==1])==-1 and mean(x[grp==2])==1.
    """
    
    x = (x- np.mean(x[grp==1])) / (np.mean(x[grp==2]) - np.mean(x[grp==1])) * 2 - 1
    return x
    
    
