# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:52:26 2014

@author: schackv
"""

# -*- coding: utf-8 -*-
"""
Demonstrates the use of the phenoquant.numberofgenes subpackage

Created on Thu Jun 26 13:12:05 2014

@author: schackv
"""
import matplotlib.pyplot as plt
import numpy as np
from phenoquant.numberofgenes.simulation import PhenotypeSimulator
from phenoquant.numberofgenes.oneortwosets import Estimator

def demo_oneortwosets():
    """Demonstrate the estimation of the same or separate sets of genes using simulated data
    for a variety of parameters.
    """
    
    # Phenotype and admixture proportion simulation
    IS_SAME = False   # True hypothesis 
    K = 2
    PS = PhenotypeSimulator(K)
    PS.simulateTwoPhenotypes(N=1000,same_set=IS_SAME,sigma_e=(0.2, 0.05),sigma_f=0)

    ## Plot the simulated phenotypes
    plt.figure()
    plt.plot(PS.f,np.hstack(PS.z),'x')
    plt.xlabel('Admixture proportion f')
    plt.ylabel('Phenotypic quantity z')
    plt.show(block=False)
    
    E = Estimator(PS.f,PS.z[0],PS.z[1],K=2)
    maxll1, maxll2, params1, params2 = E.maximize_likelihood()
    
    print('Log-likelihood of same set: {:.3f}'.format(maxll1))
    print('Log-likelihood of two sets: {:.3f}'.format(maxll2))
    
        
def square_plot(ax):
    """Make a plot square"""
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))

if __name__=='__main__':
    demo_oneortwosets()