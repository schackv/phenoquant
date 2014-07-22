# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:12:05 2014

@author: schackv
"""

import numpy as np
import matplotlib.pyplot as plt
import phenoquant.manifolds as manifolds

def demo_mimicryphenotype():
    """Reduce a collection of extracted phenotypes to derive a mimicry-related
    phenotype, using the model species as training data.
    
    This demo is meant to show how to handle data extracted from a collection 
    of images (as shown in demo_features.py for a single image).
    
    It does not illustrate how to do parameter selection in a meaningful way.
    """
    
    # Read data
    raw = np.genfromtxt('phenotypes.csv',delimiter=';',dtype=None)
    ids = [int(x) for x in raw[1:,0]]
    locations = [x.decode("utf-8") for x in raw[1:,1]]
    var_names = [x.decode("utf-8") for x in raw[0,2:]]
    data = raw[1:,2:].astype(np.float32)
    N, p = data.shape
    
    # Find model species individuals
    is_model1 = [s == 'Ranitomeya summersi' for s in locations]
    is_model2 = [s == 'Ranitomeya variabilis' for s in locations]
    
    # Reduce to one dimension using linear discriminant analysis
    y = np.array(is_model1,dtype=int) + 2*np.array(is_model2,dtype=int)
    X = (data-data.mean(axis=0))/data.std(axis=0,ddof=1)
    z, w = manifolds.fisherlda(X,y,alpha=0.5)   # Arbitrary reg. param. choice
    
    # Plot variation of phenotype per locality
    unique_locations = set(locations)
    phenotypes = []
    for loc in unique_locations:
        idx = np.array([l==loc for l in locations])
        phenotypes.append(z[idx])
    
    plt.figure()
    plt.boxplot(phenotypes)
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.xticks(range(1,len(unique_locations)+1),list(unique_locations),rotation=90)
    plt.show(block=False)    
    
    # Show variables correlation with found direction
    C = np.corrcoef( np.hstack((data,z)).T)[:-1,-1]
    plt.figure()
    plt.grid()
    plt.bar(range(p),C,align='center')
    plt.xticks( range(p), var_names, rotation=90 )
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.autoscale()
    plt.title('Variable correlation with mimicry phenotype')
    plt.show(block=True)
        

if __name__=='__main__':
    demo_mimicryphenotype()