# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:12:05 2014

@author: schackv
"""


import scipy.misc
import matplotlib.pyplot as plt
import phenoquant as pq

"""Use a single image as an example of feature extraction"""
def demo_features():
    
    # Load image and mask
    im = scipy.misc.imread('warpedImage.png')
    mask = scipy.misc.imread('mask.png')
        
    # Define the features to be extracted
    feature_opts = {'interest_points': {
                        'xy': [[]],
                        'radii': [50, 50, 100, 100]},
                    'bwratio': True,
                    'gradient_histograms': {
                        'nbins': 4,
                        'sigmas': [2, 8]},
                    'shape_histograms':{
                        'nbins': 5,
                        'sigmas': [2, 8]}
                    }

    
    
    # Extract all features
    features = pq.features.extract(im,mask,feature_opts)
    print(features)
    
    # Show examples of features
    F_bwratio = pq.features.bwratio(im,feature_opts['bwratio'])
    F_gradient = pq.features.gradient_histograms(im,feature_opts['gradient_histograms'])
    F_shape = pq.features.gradient_histograms(im,feature_opts['shape_histograms'])
    
    


if __name__=='__main__':
    demo_features()