# -*- coding: utf-8 -*-
"""
Demo using phenoquant.discrim.py and phenoquant.kernel.py.

Created on Thu Jun 26 13:12:05 2014

@author: schackv
"""

import numpy as np
import matplotlib.pyplot as plt
import phenoquant.discrim as discrim

def demo_mimicryphenotype():
    """Reduce a collection of extracted phenotypes to derive a mimicry-related
    phenotype, using the model species as training data.
    
    This demo is meant to show how to handle data extracted from a collection 
    of images (as shown in demo_features.py for a single image).
    
    It does not illustrate how to do parameter selection in a meaningful way.
    """
    
    manifold = 'kda'        # Choose 'lda' or 'kda'
    
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
    
    y = np.array(is_model1,dtype=int) + 2*np.array(is_model2,dtype=int) # Class vector
    X = (data-data.mean(axis=0))/data.std(axis=0,ddof=1)        # Standardize data column wise
    
    # Reduce to one dimension
    if manifold=='lda':
         # Linear discriminant analysis    
        z, w = discrim.fisherlda(X,y,alpha=0.5)   # Arbitrary reg. param. choice
    elif manifold =='kda':
        #  Kernel discriminant analysis
        z, w = discrim.fisherkda(X,y,kernel='Gaussian',scale=None,alpha=0.5)   # Arbitrary reg. param. choice. scale=None defaults to rule-of-thumb scale choices
    
    # Plot variation of phenotype per locality
    unique_locations, phenotypes = groupvalues(z, locations)
    
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
        
        
def groupvalues(x, grp):
    """Group the values of x by the values in grp. 
    Returns the unique values from grp and a list of the same length with arrays.
    """
    unq_grp = set(grp)
    values = []
    for loc in unq_grp:
        idx = np.array([l==loc for l in grp])
        values.append(x[idx])
    
    return unq_grp, values
    
if __name__=='__main__':
    demo_mimicryphenotype()