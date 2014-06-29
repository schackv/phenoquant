# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:37:04 2014

Adds command line support for the number-of-genes simulations.

@author: schackv
"""

import numpy as np
import simulation as PS
import estimation as GE
import pickle
import argparse

def run_simulations(K,N,sigmas,f_sigma,B,kmax):
    """Run simulations with the given parameters."""
    # Initialize simulator
    ps = PS.PhenotypeSimulator(K)
    
    results = []
    for sigma in sigmas:
        # Simulate data
        ps.simulateData(N,sigma,f_sigma,B)
        
        # Estimate the number of genes
        ge = GE.GeneEstimator()
        res = ge.estimate(ps.f,ps.z,kmax)
        results.append(res)
        
        print("sigma = {}".format(sigma))
    
    return results
    

if __name__ == '__main__':
    """Command line support for the simulations"""
    
    with open('test.pic','rb') as f:
        args, results = pickle.load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument("-K","--K", help="Number of true genes",type=int)
    parser.add_argument("-N","--N", help="Number of observations",type=int)
    parser.add_argument("-kmax","--kmax", help="Maximum number of genes to estimate likelihood for.",type=int)
    parser.add_argument("-minsig","--minsigma", help="Minimum environmental noise to simulate.",type=float)
    parser.add_argument("-maxsig","--maxsigma", help="Maximum environmental noise to simulate.",type=float)
    parser.add_argument("-numsig","--numsigma", help="Number of sigmas between minsigma and maxsigma to simulate data for.",type=int)
    parser.add_argument("-fs","--fsigma", help="Noise on f",type=float)
    parser.add_argument("-B","--B", help="Number of admixture proportion estimates simulated to have.", type=int)
    parser.add_argument("-outputfile","--outputfile", help="Output filename to pickle the result to.",type=float)
    args = parser.parse_args()
    if (args.fsigma is None):
        args.fsigma = 0
    if (args.K is None):
        args.K = 1
    if (args.N is None):
        args.N = 1000
    if (args.B is None):
        args.B = 50
    if (args.kmax is None):
        args.kmax = 3
    if (args.minsigma is None):
        args.minsigma = 0.05
    if (args.maxsigma is None):
        args.maxsigma = 0.2
    if (args.numsigma is None):
        args.numsigma = 4
    
    print("Simulator called with K={} and fsigma={}".format(args.K,args.fsigma))        

    results = run_simulations(args.K, args.N, np.linspace(args.minsigma,args.maxsigma,args.numsigma), args.fsigma, args.B, args.kmax)
    
    if (args.outputfile is not None):
        # Save results to a file
        with open(args.outputfile,'wb') as f:
            pickle.dump([args, results], f)


    