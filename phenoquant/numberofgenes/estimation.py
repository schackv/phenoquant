
from __future__ import division
import numpy as np
import scipy.stats as stats
import itertools
import simulation as ps
import matplotlib.pyplot as p
from scipy.optimize import minimize
from math import sqrt, exp, pi, isnan

#from joblib import Parallel, delayed

class GeneEstimator:
    
    data = [];
    
    def estimate(self,f,z,kmax):
        """Estimate the maximum likelihood for 1:kmax genes given the admixture
        proportions in f and the phenotypic quantities in z.
        """
        results = []
        for k in range(1,kmax+1):
            res = self.profile_likelihood(f,z,k)
            minfval = min([R.fun for R in res])
            print("   K = {}, fval = {}".format(k,minfval))
            results.append(res)
            
        return results

    def profile_likelihood(self,f,z,K):
        """Maximize and return the log-likelihood under the hypothesis of K genes
        given the admixture proportions in f and the phenotypic quantities in z.
        """
        
        self.X0 = starting_points(z)

        # Precompute genotype probabilities        
        p_G = get_pG(f,K)

        R = []
        for x0 in self.X0:
            res = minimize(minllnumgenes,x0,method='nelder-mead', args=(K,z,f,p_G), options = {'disp': False})
            res.update({'x0': x0})
            R.append(res)
        return R

def starting_points(z):
    """Generate starting points based on the data percentiles"""
    percentiles = [[10,50,90],
                   [10,10,90],
                    [10,90,90],
                    [40,50,60],
                    [40,40,60],
                    [40,60,60]]
    z_std = np.std(z)                
    sigmas = z_std / [1, 5, 10]
    x0 = []                        
    for prc in percentiles:
        x = stats.scoreatpercentile(z,prc)
        for s in sigmas:
            x0.append(np.hstack([x,s]))
    return x0
        

def minllnumgenes(x,K,z,f,p_G):
    """Wrapper function unwrapping the optimization parameters to meaningful 
    parameters for llnumgenes (and negating the function value.
    """
    mu = x[0:3]
    sigma = x[3]
    
    ll = llnumgenes(mu,sigma,K,z,f,p_G)
    # Return negated log-likelihood
    return -ll

def get_pG(f,K):
    """Get the genotype probabilities."""
    N = f.shape[0]
    B = f.shape[1]
    # Calculate genotype probabilities
    perms = permutations(K)     # Generator function

    i = 0        
    p_G = np.zeros([N,3**K])
    for P in perms:
        for b in range(B):
            pg = 1
            for g in P:               
                pg = pg * prob_of_g(g,f[:,b])   # product over K genes' genotype is probability of multi-locus genotype
            p_G[:,i] = p_G[:,i] + pg            # sum up bootstrap-probabilities
        p_G[:,i] = p_G[:,i] / B                 # Average of bootstraps
        i += 1
    return p_G


def llnumgenes(mu,sigma,K,z,f,p_G):
    """Returns the negative log-likelihood of the parameters, 
    given the number of genes is K and the data z and admixture proportions f"""
    
    ll_perobs = llperobs(mu,sigma,K,z,f,p_G)

    ll= np.sum(ll_perobs)

    return ll
    
def llperobs(mu,sigma,K,z,f,p_G):
    """Calculate the log-likelihood per observation under the model."""
    if p_G is None:
        p_G = get_pG(f,K)
    
    # Sum over all conditional probabilities times genotype probabilities 
    perms = permutations(K)     # Generator function
    p_z= 0
    i = 0
    for P in perms:
        p_mu = 1/K * sum([ P.count(a)*mu[a] for a in range(3)  ])
        aux = normpdf(z,p_mu,sigma)
        p_z  = p_z + aux.T * p_G[:,i]
        i += 1
    
    # Take sum of logarithms of conditional probabilities    
    eps = 1e-50
    nansum = np.sum(np.isnan(p_z))
    if (np.count_nonzero(p_z==np.inf)>0): 
        print('Inf');
    if (np.count_nonzero(p_z==-np.inf)>0):
        print("-inf")
    if (nansum>0):
        print(nansum)
    negcount = np.sum(p_z<0)
    if (negcount>0):
        print(negcount)
    
    return np.log(p_z + eps)

def normpdf(x, mu, sigma):
    """Normal pdf"""
    y = (1/(sqrt(2*pi*sigma**2)))*np.exp(-(x-mu)**2/(2*sigma**2))
    return y

def prob_of_g(g,f):
    """Genotype probabilities (g=0, 1 or 2)"""
    if g==0:
        prob = f**2
    elif g==1:
        prob = 2*(1-f)*f
    elif g==2:
        prob = (1-f)**2
    return prob
    

def permutations(K):
    P = itertools.product(range(3),repeat=K)
    return P
    

if __name__ == '__main__':
    """Test runs for varying number of true K"""
    for ktrue in range(1,4):
        print("K_true = {}".format(ktrue))
        PS = ps.PhenotypeSimulator(ktrue)
        PS.simulateData(1000,0.2,0.05,100)
    
        GE = GeneEstimator()
        
        for k in range(1,4):
            res = GE.profile_likelihood(PS.f,PS.z,k)
            minfval = min([R.fun for R in res])
            print("   K = {}, fval = {}".format(k,minfval))

    
    
    
    
    
    
    
    
    
    