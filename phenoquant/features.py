# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:51:44 2014

@author: schackv
"""

from .scalespace import scale, gradient_orientation, shape_index
from scipy.special import jn
from scipy.ndimage.filters import gaussian_filter
import numpy as np

""" Extract features according to the supplied options"""
def extract(im, mask, opts):

    
    
    
    f = []
    
    # Extract black/white ratio proportions
    if 'bwratio' in opts:
        bw = bwratio(im,opts['bwratio'])
        f.append( spatial_pooling(bw, mask, opts['interest_points']) )
        
    # Gradient orientation histograms
    if 'gradient_histograms' in opts:
        goh = gradient_histograms(im,opts['gradient_histograms'])
        f.append( spatial_pooling(goh ,mask, opts['interest_points']) )
        
    # Shape index histograms
    if 'shape_histograms' in opts:
        sih = shape_histograms(im, opts['shape_histograms'])
        f.append( spatial_pooling(sih, mask, opts['interest_points']))
        
    return f
    
    
""" Do a spatial average for each interest point over each image"""
def spatial_pooling(images,mask,opts):
    f = []
    for xy, radius in enumerate(opts['xy'],opts['radii']):
        print(xy)
        print(radius)
    

def bwratio(im,opts):
    pass



"""
Gradient orientation histograms in a scale space formulation.
The scales in the array sigmas controls the spatial scale space.
The gradient orientations live in the range [0,2pi] and this range is binned in 
nbins bins. The smoothing over this range is controlled by the tonal scale.
The aperture scale controls the spatial smoothing in each bin.
"""
def gradient_histograms(im,nbins,sigmas,tonal_scale=1.0,aperture_scale=1.0):    
    tonal_scale *= np.ones_like(sigmas)
    aperture_scale *= np.ones_like(sigmas)
    
    bin_centers = np.linspace(0,2*np.pi*(nbins-1)/nbins,nbins)
    
    hists  = []
    labels = []
    for idx,s in enumerate(sigmas):
        go, go_m = gradient_orientation(im, s)
        for b in bin_centers:
            go_R = go_m * np.exp(tonal_scale[idx]**(-2) * np.cos(go - b ) ) / (2*np.pi * jn(0,tonal_scale[idx]))
            h = gaussian_filter(go_R, aperture_scale[idx])
            hists.append(h)
            labels.append('sigma={:.2f}, b={:.2f}'.format(s,b))
    hists = np.dstack(hists)
    return hists, labels
    
"""Shape index histograms"""
def shape_histograms(im, nbins, sigmas, tonal_scale=1.0, aperture_scale=1.0):
    tonal_scale *= np.ones_like(sigmas)
    aperture_scale *= np.ones_like(sigmas)
    
    bin_centers = 1/nbins + np.linspace(-1,1,num=nbins,endpoint=False)
    
    xx = np.linspace(-1,1,num=100)
    dxx = xx[2]-xx[1]
    
    hists =[]
    labels = []
    for idx, s in enumerate(sigmas):
        si, si_c = shape_index(im, s)   # Shape index at scale s
        for b in bin_centers:
            z = np.sum( np.exp( - (xx - b)**2/(2*tonal_scale[idx]**2) )*dxx)
            si_R = si_c/z * np.exp( -(si - b)**2 / (2*tonal_scale[idx]**2) )
            h = gaussian_filter(si_R, aperture_scale[idx])
            hists.append(h)
            labels.append('sigma={:.2f}, b={:.2f}'.format(s,b))
    hists = np.dstack(hists)
    return hists,labels
    
    
    