# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:51:44 2014

@author: schackv
"""



""" Extract features according to the supplied options"""
def extract(im, mask, opts):
    f = []
    
    # Extract black/white ratio proportions
    if 'bwratio' in opts:
        bw = bwratio(im,opts['bwratio'])
        f.append( spatial_pooling(bw, mask, opts['interest_points'])
        
    # Gradient orientation histograms
    if 'gradient_histograms' in opts:
        goh = gradient_histograms(im,opts['gradient_histograms'])
        f.append( spatial_pooling(goh ,mask, opts['interest_points']))
        
    # Shape index histograms
    if 'shape_histograms' in opts:
        sih = shape_histograms(im, opts['shape_histograms'])
        f.append( spatial_pooling(sih, mask, opts['interest_points']))
        
    return f
    
    
def spatial_pooling(images,mask,opts):
    

def bwratio(im,opts):
    

def gradient_histograms(im,opts):
    

def shape_histograms(im, opts):