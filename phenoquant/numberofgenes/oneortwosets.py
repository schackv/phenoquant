# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 22:14:36 2014

@author: schackv
"""

from __future__ import division
import numpy as np
from . import estimation
from scipy.optimize import minimize
import scipy.stats as stats


class Estimator():
    
    
    def __init__(self, f, z1, z2, K=1):
        """Initialize an estimator with given admixture proportions f
        and phenotypes z1 and z2.
        """
        self.f = f
        self.K = K
        self.p_G = None
        self.z1 = z1
        self.z2 = z2
        
        
    def maximize_likelihood(self):
        maxll1, params1 = self.profile_likelihood(same_set=True)
        maxll2, params2 = self.profile_likelihood(same_set=False)
       
        return maxll1, maxll2, params1, params2
        
    def profile_likelihood(self,same_set):
        """Maximize and return the log-likelihood under the hypothesis of same_set
        
        Returns log-likelihood (not negative log-likelihood!) and parameter estimates as
            loglikelihood, params = profile_likelihood(...)
        """
        
        fun = lambda x: self.negll(x,same_set)  # function to minimize
        
        self.X0 = starting_points(self.z1,self.z2)

        R = []
        for x0 in self.X0:
            res = minimize(fun,x0,method='nelder-mead', options = {'disp': True})
            res.update({'x0': x0})
            R.append(res)
        
        # Find minimum negative log-likelihood
        min_idx = np.argmin([r.fun for r in R])
        negll = R[min_idx].fun
        params = R[min_idx].x
        
        return -negll, params
        
    def likelihood(self,same_set,mu1,sigma1,mu2,sigma2):
        """Get the likelihood of the parameters.
        
        mu1 and mu2 are 3 element np.arrays
        sigma1 and sigma2 are scalars.
        """
        
        if self.p_G is None:
            self.p_G = estimation.get_pG(self.f,self.K)
            
    
        # Sum over all conditional probabilities times genotype probabilities 
        perms = estimation.permutations(self.K)     # Generator function
        p_z1 = 0
        p_z2 = 0
        i = 0
        for P in perms:
            p_mu1 = 1/self.K * sum([ P.count(a)*mu1[a] for a in range(3)  ])
            p_mu2 = 1/self.K * sum([ P.count(a)*mu2[a] for a in range(3)  ])

            aux1 = estimation.normpdf(self.z1,p_mu1,sigma1)
            aux2 = estimation.normpdf(self.z2,p_mu2,sigma2)
            
            if same_set:
                p_z1 = p_z1 + aux1.T * aux2.T * self.p_G[:,i]
            else:
                p_z1 = p_z1 + aux1.T * self.p_G[:,i]
                p_z2 = p_z2 + aux2.T * self.p_G[:,i]            
            i += 1
        
        # Take sum of logarithms of conditional probabilities    
        eps = 1e-50
        if same_set:
            ll = np.log(p_z1 + eps)
        else:
            ll = np.log(p_z1 * p_z2 + eps)
        
        return np.sum(ll)
        
        
    def negll(self,x,same_set):
        """Wrapper function unwrapping the optimization parameters to meaningful 
        parameters for the likelihood function (and negating the function value.
        """
        mu1 = x[0:3]
        sigma1 = x[3]
        mu2 = x[4:7]
        sigma2 = x[7]
        
        ll = self.likelihood(same_set,mu1,sigma1,mu2,sigma2)
        # Return negated log-likelihood
        return -ll
        
        
def starting_points(z1,z2):
    """Generate starting points based on the data percentiles"""
    percentiles = [[10,50,90],
                   [10,10,90],
                    [10,90,90],
                    [40,50,60],
                    [40,40,60],
                    [40,60,60]]
    z1_std = np.std(z1)
    z2_std = np.std(z2)
    sigma_frac = [1,5,10]
    
    
    x0 = []
    for prc1 in percentiles:
        x1 = stats.scoreatpercentile(z1,prc1)
        for prc2 in percentiles:
            x2 = stats.scoreatpercentile(z2,prc2)
            for s in sigma_frac:
                s1 = z1_std/s
                s2 = z2_std/s
                x0.append(np.hstack([x1,s1,x2,s2]))
    
    return x0
    
    
    