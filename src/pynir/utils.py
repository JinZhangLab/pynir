# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:02:36 2022

@author: Jin Zhang (zhangjin@mail.nankai.edu.cn)
"""
import numpy as np
from scipy.stats import norm

def simulateNIR(nSample = 100, n_components = 3, refType = 1, noise = 0.0, seeds = 1):
    """
    simulating NIR spectra

    Parameters
    ----------
    nSample : int, optional
        number of samples. The default is 100.

    n_components : int, optional
        number of componnet for spectral simulation. The default is 3.

    refType : int, optional
        type of reference value
        None for no reference value output
        1 for contious values as reference value output
        2 or the larger integer for binary or class output.

    seeds : int, optimal
        random seed for generating spectra and reference values. The default is 1.

    Returns
    -------
    X:  matrix, simulated NIR spectra matrix.
    y: array, concentration or class of all samples.

    """
    wv = np.linspace(1000,2500,500) #wavelength
    np.random.seed(seeds)
    conc = np.random.random((nSample,n_components))
    mu = np.random.random(n_components)*1500+1000
    sigma = np.random.random(n_components)*100+100
    spcBase = [norm.pdf(wv, mu[i],sigma[i]) for i in range(n_components)]
    X = np.dot(conc,spcBase)
    X = X + np.random.randn(*X.shape)*noise
    conc = conc + np.random.randn(*conc.shape)*noise
    if refType == 0:
        y = None
    elif refType == 1:
        y = conc[:,1]
    elif refType > 1:
        y = np.zeros((conc[:,1].shape),dtype=int)
        yquantile = np.linspace(0,1,refType+1)
        for i in range(refType):
            if i == refType-1:
                conditioni = np.logical_and(conc[:,1] >= np.quantile(conc[:,1],yquantile[i]), conc[:,1] <= np.quantile(conc[:,1],yquantile[i+1]))
            else:
                conditioni = np.logical_and(conc[:,1] >= np.quantile(conc[:,1],yquantile[i]), conc[:,1] < np.quantile(conc[:,1],yquantile[i+1]))
            y = y + conditioni*i
    else:
        raise ValueError("refType only allow integer larger than 0 as input")

    return X, y, wv


def simulateNIR_calibrationTransfer(nSample = 100, n_components = 3,shifts = 0.01, refType = 1, noise = 0.0, seeds = 1):
    """
    simulating NIR spectra for calibration transfer

    Parameters
    ----------
    nSample : int, optional
        number of samples. The default is 100.

    n_components : int, optional
        number of componnet for spectral simulation. The default is 3.

    shifts: float, optimal
        shift level of base peaks for simulte secondary NIR spectra data

    refType : int, optional
        type of reference value
        None for no reference value output
        1 for contious values as reference value output
        2 or the larger integer for binary or class output.

    seeds : int, optimal
        random seed for generating spectra and reference values. The default is 1.

    Returns
    -------
    X:  matrix, simulated NIR spectra matrix.
    y: array, concentration or class of all samples.

    """
    wv = np.linspace(1000,2500,500) #wavelength
    np.random.seed(seeds)
    conc = np.random.random((nSample,n_components))
    mu1 = np.random.random(n_components)*1500+1000
    sigma1 = np.random.random(n_components)*100+100
    spcBase1 = [norm.pdf(wv, mu1[i],sigma1[i]) for i in range(n_components)]
    X1 = np.dot(conc,spcBase1)
    X1 = X1 + np.random.randn(*X1.shape)*noise


    mu2 = mu1 + np.random.random(mu1.shape)*shifts
    sigma2 = sigma1 + np.random.random(sigma1.shape)*shifts
    spcBase2 = [norm.pdf(wv, mu2[i],sigma2[i]) for i in range(n_components)]
    X2 = np.dot(conc,spcBase2)
    X2 = X2 + np.random.randn(*X2.shape)*noise


    conc = conc + np.random.randn(*conc.shape)*noise
    if refType == 0:
        y = None
    elif refType == 1:
        y = conc[:,1]
    elif refType > 1:
        y = np.zeros((conc[:,1].shape),dtype=int)
        yquantile = np.linspace(0,1,refType+1)
        for i in range(refType):
            if i == refType-1:
                conditioni = np.logical_and(conc[:,1] >= np.quantile(conc[:,1],yquantile[i]), conc[:,1] <= np.quantile(conc[:,1],yquantile[i+1]))
            else:
                conditioni = np.logical_and(conc[:,1] >= np.quantile(conc[:,1],yquantile[i]), conc[:,1] < np.quantile(conc[:,1],yquantile[i+1]))
            y = y + conditioni*i
    else:
        raise ValueError("refType only allow integer larger than 0 as input")

    return X1, X2, y, wv
