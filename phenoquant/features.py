# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:51:44 2014

@author: schackv
"""

from .scalespace import scale, gradient_orientation, shape_index
from scipy.special import jn
from scipy.ndimage.filters import gaussian_filter
from skimage import filter
import numpy as np
from . import imtools

def extract(im, mask, opts):
    """ Extract features according to the supplied options"""
    
    f = []
    labels = []
    im_gray = imtools.standardize(imtools.rgb_to_gray(im),mask=mask)    
    
    # Extract black/white ratio proportions
    if 'bwratio' in opts:
        bw, glaremask = segment_stripes(im,mask)
        val, labels_ = spatial_pooling(bw, glaremask, labels_in=('bw',), **opts['interest_points'])
        f.append(val)
        labels.extend(labels_)
        
    # Gradient orientation histograms
    if 'gradient_histograms' in opts:
        goh, labels_ = gradient_histograms(im_gray,**opts['gradient_histograms'])
        val, labels_ = spatial_pooling(goh, mask,labels_in=labels_, **opts['interest_points'])
        f.append(val)
        labels.extend(labels_)
        
    # Shape index histograms
    if 'shape_histograms' in opts:
        sih, labels_ = shape_histograms(im_gray,**opts['shape_histograms'])
        val, labels_ = spatial_pooling(sih, mask,labels_in=labels_, **opts['interest_points'])
        f.append(val)
        labels.extend(labels_)

        
    return np.hstack(f), labels
    
    
def spatial_pooling(images,mask,xy,radii,labels_in=[]):
    """Do a spatial average for each interest point over each layer"""

    imshp = images.shape
    m, n = imshp[0:2]
    if len(imshp)>2:
        p = imshp[2]
    else:
        images = images[:,:,np.newaxis]
        p = 1
        
    if not labels_in:
        labels_in = ['f{}'.format(i) for i in range(p)]
    
    f = []
    labels = []
    for c, (xy, radius) in enumerate(zip(xy,radii)):
        circ_mask = imtools.circular_mask((m,n),xy,radius)  # Create mask around interest point
        for i in range(p):
            im = images[:,:,i]
            val = np.mean(im[circ_mask & mask])     # Average over region
            f.append(val)
            labels.append('c{}, {}'.format(c,labels_in[i]))
    f = np.hstack(f)
    return f, labels
            
            
    

def segment_stripes(imrgb,mask):
    """Segment image into black and colored regions (stripe-or-not)"""
    
    mask = mask & ~imtools.detect_glare(imrgb)
    im = imtools.rgb_to_gray(imrgb)
    # Do thresholding in grayscale image
    im = imtools.linear_scaling(im,oldmin=np.mean(im[mask])-3*np.std(im[mask]),oldmax=np.mean(im[mask])+3*np.std(im[mask]))
    tau = filter.threshold_otsu(im)
    stripes = (im>tau) & mask
    
    return stripes, mask
    



def gradient_histograms(im,nbins,sigmas,tonal_scale=1.0,aperture_scale=1.0):
    """
    Gradient orientation histograms in a scale space formulation.
    The scales in the array sigmas controls the spatial scale space.
    The gradient orientations live in the range [0,2pi] and this range is binned in 
    nbins bins. The smoothing over this range is controlled by the tonal scale.
    The aperture scale controls the spatial smoothing in each bin.
    """
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
    
def shape_histograms(im, nbins, sigmas, tonal_scale=1.0, aperture_scale=1.0):
    """Shape index histograms"""
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
    
    
    