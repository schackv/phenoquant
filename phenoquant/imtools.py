# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 17:14:25 2014

@author: schackv
"""

import numpy as np

def rgb_to_gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    


def standardize(X,mask=[]):
    if not mask.size:
        mask = np.ones_like(X)==1
    
    return (X-np.mean(X[mask]))/np.std(X[mask])