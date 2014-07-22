# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:12:05 2014

Demonstrates the use of the numberofgenes subpackage

@author: schackv
"""
import matplotlib.pyplot as plt
import numpy as np
from phenoquant.numberofgenes.simulation import PhenotypeSimulator
from phenoquant.numberofgenes.estimation import GeneEstimator

def demo_numberofgenes():
    """Demonstrate the estimation of the number of genes using simulated data
    for a variety of parameters. The parameters are repeatedly estimated to
    get a distribution of the estimates.
    """
    
    # Phenotype and admixture proportion simulation
    K = 1   # True number of genes
    PS = PhenotypeSimulator(K)
    PS.simulateData(N=1000,sigma_e=0.2,sigma_f=0)

    ## Plot the simulated phenotypes
    plt.figure()
    plt.plot(PS.f,PS.z,'x')
    plt.xlabel('Admixture proportion f')
    plt.ylabel('Phenotypic quantity z')
    plt.show(block=False)
    
    # Estimate maximum likelihood for K=1:3 and show likelihood ratios
    repeats = 10
    kmax = 3
    maxll = []
    params = []
    GE = GeneEstimator()
    for r in range(repeats):
        maxll_, params_ = GE.estimate(PS.f,PS.z,kmax=kmax)
        maxll.append(maxll_)
        params.append(np.vstack(params_))
    maxll = np.vstack(maxll)
    params = np.dstack(params)
    
    ## Make boxplots
    fig, axs = plt.subplots(1,2)
    plt.sca(axs[0])
    plt.boxplot(maxll[:,1:]-maxll[:,0,np.newaxis])
    plt.xticks(np.arange(1,kmax),np.arange(2,kmax+1))
    plt.xlabel('k')
    plt.ylabel('ll(K=k/K=1)')
    
    # Second boxplot contains parameter estimates for k=1
    plt.sca(axs[1])
    plt.hist(params[0,:,:].T)
    plt.legend((r'$\mu_1$',r'$\mu_2$',r'$\mu_3$',r'$\sigma_e$'))
    plt.title('Parameters for k=1')
    square_plot(axs[1])
    
    plt.show()
        
def square_plot(ax):
    """Make a plot square"""
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))

if __name__=='__main__':
    demo_numberofgenes()