# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:22:04 2014

@author: schackv
"""

import numpy as np
import matplotlib.pyplot as p

class PhenotypeSimulator:
    
    def __init__(self, K):
        """Initialize the phenotype simulator with K genes"""
        self.K = K    
    
    def simulateData(self,N, sigma_e, sigma_f, B, seed=None):
        """Simulate N phenotype observations with sigma_e being the st.d. on the
        environmental variance, sigma_f the estimation noise on the admixture proportions
        and B the number of samples of each admixture proportion.
        """
        
        self.sigma_e = sigma_e
        self.sigma_f = sigma_f
        self.B = B   
        self.N = N
        self.mu = np.array([[0],
                           [1],
                           [2]])
        
        if seed is not None:
            np.random.seed(seed)
        
        # Draw a uniform distribution of f
        self.f_true = np.random.uniform(0,1,N)
        
        # Assign genotypes based on probabilities
        G = self.assign_genotypes(self.f_true,self.K)
        
        # Convert to genotype proportions
        H = np.column_stack((np.sum(G==0,axis=1),np.sum(G==1,axis=1),np.sum(G==2,axis=1)))
        H = H/ self.K
        
        # Assign phenotypic values
        z_true = np.dot(H,self.mu)
        # print(z_true)
        z_noise = sigma_e * np.random.normal(0,1,size=[N,1])
        self.z = z_true + z_noise
        
        # Add noise
        f_noise = sigma_f * np.random.normal(0,1,size=[N,B])
        f = self.f_true[:,np.newaxis] + f_noise  # Broadcast
        f = np.maximum(np.minimum(f,1),0)
        
        self.f = np.array(f);
        
        
    @classmethod
    def assign_genotypes(cls,f,K):
        """Randomly assign K-locis genotypes given the admixture proportions f"""
        N = len(f)
        G = np.empty([N,K])
        for i in range(0,N):
            p_G = [f[i]**2, 2*(1-f[i])*f[i], (1-f[i])**2]
            for k in range(0,K):
                G[i,k] = np.random.choice(3,p=p_G)
        return G
                
    

        

if __name__ == '__main__':
    PS = PhenotypeSimulator(2)
    PS.simulateData(1000,0.1,0,3)
    
    p.plot(PS.f, PS.z,'ko')
    p.show()
    
    
    