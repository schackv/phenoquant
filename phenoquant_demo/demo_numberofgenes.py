# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:12:05 2014

Demonstrates the use of the numberofgenes subpackage

@author: schackv
"""
import matplotlib.pyplot as plt
from phenoquant.numberofgenes.simulation import PhenotypeSimulator

def demo_numberofgenes():
    """Demonstrate the estimation of the number of genes using simulated data
    for a variety of parameters.
    """
    
    # Phenotype and admixture proportion simulation
    K = 1   # True number of genes
    PS = PhenotypeSimulator(K)
    PS.simulateData(N=1000,sigma_e=0.2,sigma_f=0)

    ## Plot the simulated phenotypes
    plt.plot(PS.f,PS.z,'x')
    plt.xlabel('Admixture proportion f')
    plt.ylabel('Phenotypic quantity z')
    plt.show()

if __name__=='__main__':
    demo_numberofgenes()