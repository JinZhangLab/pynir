# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:00:35 2022

@author: J Zhang (jzhang@chemoinfolab.com)
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import f, chi2

import matplotlib.pyplot as plt

class outlierDetection_PLS():
    """
    A class for performing outlier sample detection using Partial Least Squares (PLS)
    regression.

    Parameters
    ----------
    ncomp : int, optional
        The number of components to use in the PLS regression (default is 2).
    conf : float, optional
        The confidence level for the outlier detection (default is 0.99).

    Attributes
    ----------
    plsModel : PLSRegression object
        The PLS regression model fitted to the input data.

    Methods
    -------
    fit(X, y)
        Fits the PLS regression model to the input data X and output data y.
    detect(X, y)
        Performs outlier detection on the input data X and output data y using
        the PLS model.
    plot_HotellingT2_Q(Q, Tsq, Q_conf, Tsq_conf, ax=None)
        Plots the Q-residuals against Hotelling's T-squared, with the confidence
        levels indicated by dashed lines.
    
    References
    ----------
    Ref1:
    https://nirpyresearch.com/outliers-detection-pls-regression-nir-spectroscopy-python/
    Ref2: https://www.sciencedirect.com/science/article/pii/S0378517314004980
    """

    def __init__(self, ncomp = 2, conf = 0.99):
        self.ncomp = ncomp
        self.conf = conf
    
    def fit(self, X, y):
        """
        Fits the PLS regression model to the input data X and output data y.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of data.
        y : numpy.ndarray
            The output vector of data.

        Returns
        -------
        self : outlierDetection_PLS
            The fitted outlier detection object.
        """
        ncomp = self.ncomp
        self.plsModel = PLSRegression(n_components=ncomp)
        self.plsModel.fit(X, y)
        
        return self
    
    def detect(self, X, y):
        """
        Performs outlier detection on the input data X and output data y using
        the PLS model.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of data.
        y : numpy.ndarray
            The output vector of data.

        Returns
        -------
        Q : numpy.ndarray
            The Q-residuals.
        Tsq : numpy.ndarray
            Hotelling's T-squared.
        Q_conf : float
            The confidence level for the Q-residuals.
        Tsq_conf : float
            The confidence level for Hotelling's T-squared.
        idxOutlier : numpy.ndarray
            A boolean array indicating which samples are outliers.
        """
        ncomp = self.ncomp
        conf = self.conf
        plsModel = self.plsModel

        T = plsModel.transform(X)

        Err = X - plsModel.inverse_transform(T)
         
        Q = np.sum(Err**2, axis=1)
        
        Q_conf = (np.var(Q)/2/np.mean(Q))*chi2.ppf(conf, 2*np.mean(Q)**2/np.var(Q))
        
        Tsq = np.sum((plsModel.x_scores_/np.std(plsModel.x_scores_, axis=0))**2, axis=1)
        
        Tsq_conf = f.ppf(q=conf,dfn=ncomp,dfd=X.shape[0]-ncomp)
        Tsq_conf = Tsq_conf*ncomp*(X.shape[0]-1)*(X.shape[0]+1)/X.shape[0]/(X.shape[0]-ncomp)
        idxOutlier = np.logical_and(Q>Q_conf, Tsq>Tsq_conf)
         
        return Q, Tsq, Q_conf, Tsq_conf, idxOutlier
    
    def plot_HotellingT2_Q(self, Q, Tsq, Q_conf, Tsq_conf, ax=None):
        """
        Plots the Q-residuals against Hotelling's T-squared, with the confidence
        levels indicated by dashed lines.

        Parameters
        ----------
        Q : numpy.ndarray
            The Q-residuals.
        Tsq : numpy.ndarray
            Hotelling's T-squared.
        Q_conf : float
            The confidence level for the Q-residuals.
        Tsq_conf : float
            The confidence level for Hotelling's T-squared.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the figure (default is None).

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object containing the plotted figure.
        """
        if ax == None:
            fig, ax = plt.subplots(figsize=(8,4.5))
    
        ax.plot(Tsq, Q, 'o')
     
        ax.plot([Tsq_conf,Tsq_conf],[plt.axis()[2],plt.axis()[3]],  '--')
        ax.plot([plt.axis()[0],plt.axis()[1]],[Q_conf,Q_conf],  '--')
        ax.set_xlabel("Hotelling's T-squared")
        ax.set_ylabel('Q residuals')
    
        return ax