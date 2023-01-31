# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:00:35 2022

@author: chinn
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import f, chi2

import matplotlib.pyplot as plt

class outlierDection_PLS():
    # Ref1: https://nirpyresearch.com/outliers-detection-pls-regression-nir-spectroscopy-python/
    # Ref2: https://www.sciencedirect.com/science/article/pii/S0378517314004980
    def __init__(self, ncomp = 2, conf = 0.99):
        self.ncomp = ncomp
        self.conf = conf
    
    def fit(self, X, y):
        ncomp = self.ncomp
        self.plsModel = PLSRegression(n_components=ncomp)
        self.plsModel.fit(X, y)
        
        return self
    
    def detect(self, X, y):
        ncomp = self.ncomp
        conf = self.conf
        plsModel = self.plsModel
        # Get X scores
        T = plsModel.transform(X)

        # Calculate error array
        Err = X - plsModel.inverse_transform(T)
         
        # Calculate Q-residuals (sum over the rows of the error array)
        Q = np.sum(Err**2, axis=1)
        
        # Estimate the confidence level for the Q-residuals
        Q_conf = (np.var(Q)/2/np.mean(Q))*chi2.ppf(conf, 2*np.mean(Q)**2/np.var(Q))
        
        # Calculate Hotelling's T-squared (note that data are normalised by default)
        Tsq = np.sum((plsModel.x_scores_/np.std(plsModel.x_scores_, axis=0))**2, axis=1)
        
        # Calculate confidence level for T-squared from the ppf of the F distribution
        Tsq_conf = f.ppf(q=conf,dfn=ncomp,dfd=X.shape[0]-ncomp)
        Tsq_conf = Tsq_conf*ncomp*(X.shape[0]-1)*(X.shape[0]+1)/X.shape[0]/(X.shape[0]-ncomp)
        idxOutlier = np.logical_and(Q>Q_conf, Tsq>Tsq_conf)
         
        return Q, Tsq, Q_conf, Tsq_conf, idxOutlier
    
    def plot_HotellingT2_Q(self, Q, Tsq, Q_conf, Tsq_conf, ax=None):
        if ax ==None:
            fig, ax = plt.subplots(figsize=(8,4.5))
    
        ax.plot(Tsq, Q, 'o')
     
        ax.plot([Tsq_conf,Tsq_conf],[plt.axis()[2],plt.axis()[3]],  '--')
        ax.plot([plt.axis()[0],plt.axis()[1]],[Q_conf,Q_conf],  '--')
        ax.set_xlabel("Hotelling's T-squared")
        ax.set_ylabel('Q residuals')
     
        plt.show()