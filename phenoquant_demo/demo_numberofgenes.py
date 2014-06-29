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
    for a variety of parameters.
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
    maxll = []
    params = []
    GE = GeneEstimator()
    for r in range(repeats):
        PS.simulateData(N=1000,sigma_e=0.2,sigma_f=0)
        maxll_, params_ = GE.estimate(PS.f,PS.z,kmax=3)
        maxll.append(maxll_)
        params.append(np.vstack(params_))
    maxll = np.vstack(maxll)
    params = np.dstack(params)
    
    ## Make boxplots
    fig, axs = plt.subplots(1,2)
    plt.sca(axs[0])
    plt.boxplot(maxll[:,1:]-maxll[:,0])
    plt.xlabel('k')
    plt.ylabel('ll(K=k/K=1))
    
    plt.show()
        
    

if __name__=='__main__':
    demo_numberofgenes()