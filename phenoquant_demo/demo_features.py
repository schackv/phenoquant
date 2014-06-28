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
    
    ## Load image and mask
    im = scipy.misc.imread('warpedImage.png')
    mask = scipy.misc.imread('mask.png')>0
    im_gray = imtools.standardize(imtools.rgb_to_gray(im),mask=mask)
    
    ## Define the features to be extracted
    feature_opts = {'interest_points': {
                        'xy': [(75.875, 76.625), (202.625, 299.25), (365.125, 144.875), (630.0, 154.625)],
                        'radii': [75, 75, 150, 150]},
                    'bwratio': True,
                    'gradient_histograms': {
                        'nbins': 4,
                        'sigmas': [2, 8]},
                    'shape_histograms':{
                        'nbins': 5,
                        'sigmas': [2, 8]}
                    }

    
    
    # Extract all features
    F, labels = features.extract(im,mask,feature_opts)
    print('\n'.join('{}:\t {:.5f}'.format(*k) for k in zip(labels,F)))
    
    # Show examples of features
    stripesbw, glaremask = features.segment_stripes(im,mask)
    F_gradient, labels_gradient = features.gradient_histograms(im_gray,**feature_opts['gradient_histograms'])
    F_shape, labels_shape = features.shape_histograms(im_gray,**feature_opts['shape_histograms'])
    
    ## Stripe segmentation
    fig,axs = plt.subplots(3,1)
    for ax, title, img in zip(axs,('Original image','Stripe segmentation','Glare mask'),(im,stripesbw,glaremask)):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('image')
    
    ## Plot interest points circles to illustrate positions
    for xy, radius in  zip(feature_opts['interest_points']['xy'], feature_opts['interest_points']['radii']):
        circle = plt.Circle(xy,radius,color='r',fill=False)
        axs[0].add_artist(circle)
    
    ## Gradient orientations
    fig, axs = plt.subplots(2,4)
    for ax, F, label in zip(axs.flat,F_gradient.T,labels_gradient):
        ax.imshow(F.T)
        ax.axis('image')
        ax.axis('off')
        ax.set_title(label)
    fig.suptitle('Gradient orientation histograms')
    plt.show(block=False)
    
    ## Shape index histograms
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