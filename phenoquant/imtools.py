# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 17:14:25 2014

@author: schackv
"""

import numpy as np
import skimage.morphology as morph

def rgb_to_gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    


def standardize(X,mask=[]):
    if not mask.size:
        mask = np.ones_like(X)==1
    
    return (X-np.mean(X[mask]))/np.std(X[mask])
    
    
"""
Scale the contents of X linearly such that oldmin equals zero and oldmax equals one.
Values below oldmin and above oldmax are set equal to 0 and 1 respectively
"""
def linear_scaling(X,oldmin=-np.inf,oldmax=np.inf):
    if oldmin==-np.inf:
        oldmin = np.min(X)
    if oldmax == np.inf:
        oldmax = np.max(X)
    X = (X-oldmin)/(oldmax-oldmin)
    X[X<0]=0
    X[X>1]=1
    return X




"""
Detect pixels in the image where all channels have values above 0.95.
These pixels are dilated with a disk with radius 5
"""
def detect_glare(im):
    saturated_pixels = np.all(im>0.95*np.iinfo(im.dtype).max,axis=2)
    
    glare_mask = morph.binary_dilation(saturated_pixels, morph.disk(5)).astype('bool')
    return glare_mask
    

"""Do a morphological opening followed by a closing"""    
def open_and_close(bw,openradius=10,closeradius=3):
    bw = morph.binary_opening(bw,morph.disk(openradius))
    bw = morph.binary_closing(bw,morph.disk(3)).astype('bool')
    return bw