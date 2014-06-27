# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:12:05 2014

@author: schackv
"""


import scipy.misc
import matplotlib.pyplot as plt
import phenoquant.features as features
import phenoquant.imtools as imtools

"""Use a single image as an example of feature extraction"""
def demo_features():
    
    # Load image and mask
    im = scipy.misc.imread('warpedImage.png')
    mask = scipy.misc.imread('mask.png')>0
        
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
#    features = pq.features.extract(im,mask,feature_opts)
#    print(features)
    
    # Show examples of features
    im_gray = imtools.rgb_to_gray(im)
    im_gray = imtools.standardize(im_gray,mask=mask)
#    F_bwratio = pq.features.bwratio(im,feature_opts['bwratio'])
    F_gradient, labels_gradient = features.gradient_histograms(im_gray,**feature_opts['gradient_histograms'])
    F_shape, labels_shape = features.shape_histograms(im_gray,**feature_opts['shape_histograms'])
    
    # Gradient orientations
    fig, axs = plt.subplots(2,4)
    for ax, F, label in zip(axs.flat,F_gradient.T,labels_gradient):
        ax.imshow(F.T)
        ax.axis('image')
        ax.axis('off')
        ax.set_title(label)
    fig.suptitle('Gradient orientation histograms')
    plt.show(block=False)
    
    # Shape index histograms
    fig, axs = plt.subplots(len(feature_opts['shape_histograms']['sigmas']),feature_opts['shape_histograms']['nbins'])
    for ax, F, label in zip(axs.flat,F_shape.T,labels_shape):
        ax.imshow(F.T)
        ax.axis('image')
        ax.axis('off')
        ax.set_title(label)
    fig.suptitle('Shape index histograms')
    
    plt.show(block=True)

if __name__=='__main__':
    demo_features()