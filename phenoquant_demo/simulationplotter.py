# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:45:25 2014

@author: jsve
"""
from matplotlib import pyplot as pp
import pickle
import os
import numpy as np
import init
from utils import *



opts_md5 = '2a75615e3127bbc39c376827171353cf'
output_dir = os.path.join(init.output_dir, opts_md5)
graphics_dir = os.path.join(output_dir,'plots')
if os.path.exists(graphics_dir)==False:
    os.makedirs(graphics_dir)

    
        

def plot_simulations(output_file,K,f_sigma,opts):
    with open(output_file,'rb') as f:
        results = pickle.load(f)

#    suffix = "sim_K{}_fsigma{}".format(K,f_sigma)
    
    sigmas = opts['sigmas']
#    print(results)

    ll = np.empty([len(sigmas),opts['kmax']])
    i = 0
    for r, sigma in zip(results,sigmas):
        print('sigma: {}'.format(sigma))
        for k,k_result in zip(range(opts['kmax']),r):
            minfval = min([kr.fun for kr in k_result])
            print(' K = {}, min. fval = {}'.format(k+1,minfval))
            ll[i,k] = -minfval
        i += 1

    
    ratiosplot(sigmas,ll,K,f_sigma)            
    likelihoodplot(sigmas,ll,K,f_sigma)
    
    

def likelihoodplot(sigmas,ll,K,f_sigma):
    kmax = ll.shape[1]
    suffix = "sim_K{}_fsigma{}".format(K,f_sigma)    
    h1 = pp.plot(sigmas,ll,'o')
    pp.legend(h1,[str(i+1) for i in range(kmax)])
    pp.xlabel('$\sigma_e$')
    pp.ylabel('ll(K)')
    pp.title('$K = {}, \sigma_f = {}$'.format(K,f_sigma))
    saveplot(graphics_dir,'likelihoods',suffix,'png')


def ratiosplot(sigmas,ll,Ktrue,f_sigma):
    N = ll.shape[0]
    kmax = ll.shape[1]
    llratios = ll - ll[:,Ktrue-1].reshape(N,1)
    
    suffix = "sim_K{}_fsigma{}".format(Ktrue,f_sigma)    
    
    h1 = pp.plot(sigmas,llratios,'o')
    pp.legend(h1,[str(i+1) for i in range(kmax)],loc='upper left')
    pp.xlabel('$\sigma_e$')
    pp.ylabel('ll(K) - ll({})'.format(Ktrue))
    pp.title('$K = {}, \sigma_f = {}$'.format(Ktrue,f_sigma))
    pp.axis([min(sigmas), max(sigmas), -100, 100])
    pp.grid(which='both')
    saveplot(graphics_dir,'ratios',suffix,'png','pdf')

if __name__=='__main__':
    from options import options
#    import init
#    import hashlib
#    from pprint import pformat
#    opts_md5 = hashlib.md5(pformat(options).encode('utf-8')).hexdigest()
    
    K = 3
    fsigma = 0.1
    filename = os.path.join(output_dir,'sim_K{}_fsigma{}.pic'.format(K,fsigma))
    
    plot_simulations(filename,K,fsigma,options)