# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:01:16 2022

@author: chinn
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal
import pywt

class snv():
    def __init__(self):
        pass
    
    def fit(self, X):
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)
        return self
    
    def transform(self, X):
        return (X-self.mean[None,:])/self.std[None,:]
    
    def fit_transform(self, X):
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)
        return self.transform(X)
    
class cwt():
    def __init__(self, wavelet = "morl", scale = 20):
        """
        Parameters
        ----------
        wavelet : string, optional
            Wavelet object or name:
            ['cgau1'-'cgau8','cmor','fbsp',
             'gaus1'-'gaus8','mexh','morl','shan'].
            For details about this wavelet, refer to https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
            The default is "morl".
        scale : array_like, optional
            Wavelet scale to use. The default is 20.

        Returns
        -------
        None.

        """
        self.wavelet = wavelet
        self.scale = scale
    
    
    def getContinuousWavelet(self):
        return pywt.wavelist(kind = 'continuous')
    
    def getDiscreteWavelet(self):
        return pywt.wavelist(kind = 'discrete')
        
    def transform(self, X):
        nrow, ncol = X.shape
        
        Xcwt = pywt.cwt(X, self.scale, self.wavelet,axis=1)
        Xcwt = Xcwt[0].squeeze()

        return Xcwt

class msc():
    def __init__(self):
        pass
    
    def fit(self, X):
        self.mean = np.mean(X,axis=0)
        return self
        
    def transform(self, X):
        for i in range(X.shape[0]):
            X[i,:] = LinearRegression().fit(X[i,:][:,None], self.mean[:,None]).predict(X[i,:][:,None]).ravel()
        return X
        
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class SG_filtering():
    def __init__(self,window_length = 13, polyorder=2, **kwargs):
        self.window_length = window_length
        self.polyorder = polyorder
        self.kwargs = kwargs
    
    def transform(self, X):
        return signal.savgol_filter(X, window_length = self.window_length, 
                             polyorder = self.polyorder, **self.kwargs)
    

    
class centralization():
    def __init__(self):
        pass
    
    def fit(self, X):
        self.mean = np.mean(X,axis=0)
        return self
    
    def transform(self, X):
        return X-self.mean[None,:]
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
# High level preprocessing function   
class derivate():
    def __init__(self,deriv = 1, window_length = 13, polyorder=2, **kwargs):
        self.deriv = deriv
        self.window_length = window_length
        self.polyorder = polyorder
        self.kwargs = kwargs
    
    def transform(self, X):
        return signal.savgol_filter(X, deriv=self.deriv, window_length = self.window_length, 
                             polyorder = self.polyorder,
                             delta = 1.0, **self.kwargs)
    
    
class smooth():
    def __init__(self,window_length = 13, polyorder=2, **kwargs):
        self.window_length = window_length
        self.polyorder = polyorder
        self.kwargs = kwargs
    
    def transform(self, X):
        return signal.savgol_filter(X,  window_length = self.window_length, 
                             polyorder = self.polyorder, deriv=0, **self.kwargs)