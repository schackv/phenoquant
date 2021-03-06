# -*- coding: utf-8 -*-
"""
Various auxilliary functions useful for images.

Created on Thu Jun 26 17:14:25 2014

@author: schackv
"""

import numpy as np
import skimage.morphology as morph

def rgb_to_gray(rgb):
    """Convert an rgb image to a gray scale image using the same weighting as Matlab."""
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    

def standardize(X,mask=[]):
    """Standardize a matrix X to zero mean and unit variance. 
    
    If mask is supplied, only values with mask==True are used to calculate mean 
    and variance. All elements of X are standardized, though.
    """
    if not mask.size:
        mask = np.ones_like(X)==1
    
    return (X-np.mean(X[mask]))/np.std(X[mask])
    
    
def linear_scaling(X,oldmin=-np.inf,oldmax=np.inf):
    """Scale the contents of X linearly such that oldmin equals zero and oldmax equals one.
    Values below oldmin and above oldmax are set equal to 0 and 1 respectively
    """
    if oldmin==-np.inf:
        oldmin = np.min(X)
    if oldmax == np.inf:
        oldmax = np.max(X)
    X = (X-oldmin)/(oldmax-oldmin)
    X[X<0]=0
    X[X>1]=1
    return X



def circular_mask(shape, center, radius):
    """Create a true/false mask with pixels closer than 'radius' to 'center' being True
    and everything else False. The output array is of size 'shape'.
    """
    mask = np.ones(shape, dtype=bool)
    xx, yy = np.meshgrid(range(shape[1]),range(shape[0]))
    D = np.sqrt( (xx-center[0])**2 + (yy-center[1])**2)
    mask[D>radius]=False
    return mask



def detect_glare(im):
    """Detect pixels in the image where all channels have values above 0.95.
    These pixels are dilated with a disk with radius 5
    """
    saturated_pixels = np.all(im>0.95*np.iinfo(im.dtype).max,axis=2)
    
    glare_mask = morph.binary_dilation(saturated_pixels, morph.disk(5)).astype('bool')
    return glare_mask
    

def open_and_close(bw,openradius=10,closeradius=3):
    """Do a morphological opening followed by a closing"""
    bw = morph.binary_opening(bw,morph.disk(openradius))
    bw = morph.binary_closing(bw,morph.disk(3)).astype('bool')
    return bw