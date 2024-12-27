# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:01:16 2022

@author: J Zhang (jzhang@chemoinfolab.com)
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal
import pywt

class snv():
    """
    A class for performing Standard Normal Variate (SNV) normalization on a matrix of spectral data.

    Parameters
    ----------
    None

    Attributes
    ----------
    mean : numpy.ndarray
        The mean of each feature in the input matrix.
    std : numpy.ndarray
        The standard deviation of each feature in the input matrix.
    """
    def __init__(self):
        pass
    
    def fit(self, X):
        """
        Compute the mean and standard deviation of each feature in the input matrix.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of spectral data.

        Returns
        -------
        self : snv
            The fitted snv object.
        """
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)
        return self
    
    def transform(self, X):
        """
        Normalize the input matrix using the mean and standard deviation computed during fitting.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of spectral data.

        Returns
        -------
        X_norm : numpy.ndarray
            The normalized input matrix.
        """
        mean_expanded = np.tile(self.mean, (X.shape[0], 1))
        std_expanded = np.tile(self.std, (X.shape[0], 1))
        X_norm = (X - mean_expanded) / std_expanded
        return X_norm
    
    def fit_transform(self, X):
        """
        Compute the mean and standard deviation of each feature in the input matrix, and then normalize the input matrix.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of spectral data.

        Returns
        -------
        X_norm : numpy.ndarray
            The normalized input matrix.
        """
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)
        X_norm = self.transform(X)
        return X_norm
    
class cwt():
    """
    A class for performing Continuous Wavelet Transform (CWT) on a matrix of data.

    Parameters
    ----------
    wavelet : str, optional
        The name of the wavelet to use for the CWT (default is "morl").
    scale : int, optional
        The number of scales to use for the CWT (default is 20).

    Attributes
    ----------
    wavelet : str
        The name of the wavelet to use for the CWT.
    scale : int
        The number of scales to use for the CWT.
    """
    def __init__(self, wavelet = "morl", scale = 20):
        self.wavelet = wavelet
        self.scale = scale
    
    
    def getContinuousWavelet(self):
        """
        Get a list of available continuous wavelets.

        Returns
        -------
        wavelets : list of str
            A list of available continuous wavelets.
        """
        return pywt.wavelist(kind = 'continuous')
    
    def getDiscreteWavelet(self):
        """
        Get a list of available discrete wavelets.

        Returns
        -------
        wavelets : list of str
            A list of available discrete wavelets.
        """
        return pywt.wavelist(kind = 'discrete')
        
    def transform(self, X):
        """
        Perform Continuous Wavelet Transform on the input matrix.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of data.

        Returns
        -------
        Xcwt : numpy.ndarray
            The transformed matrix of data.
        """
        nrow, ncol = X.shape
        Xcwt = pywt.cwt(X, self.scale, self.wavelet,axis=1)
        Xcwt = Xcwt[0].squeeze()
        return Xcwt

class msc():
    """
    A class for performing Multiplicative Scatter Correction (MSC) on a matrix of data.

    Parameters
    ----------
    None

    Attributes
    ----------
    mean : numpy.ndarray
        The mean spectrum of the input matrix.
    """
    def __init__(self):
        pass
    
    def fit(self, X):
        """
        Compute the mean spectrum of the input matrix.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of spectral data.

        Returns
        -------
        self : msc
            The fitted msc object.
        """
        self.mean = np.mean(X,axis=0)
        return self
        
    def transform(self, X):
        """
        Normalize the input matrix using the mean spectrum computed during fitting.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of spectral data.

        Returns
        -------
        X_msc : numpy.ndarray
            The normalized input matrix.
        """
        for i in range(X.shape[0]):
            X_row = X[i, :].reshape(-1, 1)
            mean_reshaped = self.mean.reshape(-1, 1)
            X[i, :] = LinearRegression().fit(X_row, mean_reshaped).predict(X_row).ravel()
        return X
        
    
    def fit_transform(self, X):
        """
        Compute the mean spectrum of the input matrix, and then normalize the input matrix.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of spectral data.

        Returns
        -------
        X_msc : numpy.ndarray
            The normalized input matrix.
        """
        return self.fit(X).transform(X)

class SG_filtering():
    """
    A class for performing Savitzky-Golay filtering on a matrix of data.

    Parameters
    ----------
    window_length : int, optional
        The length of the filter window (default is 13).
    polyorder : int, optional
        The order of the polynomial used to fit the samples (default is 2).
    **kwargs : dict, optional
        Additional keyword arguments to pass to the `signal.savgol_filter` function.

    Attributes
    ----------
    window_length : int
        The length of the filter window.
    polyorder : int
        The order of the polynomial used to fit the samples.
    kwargs : dict
        Additional keyword arguments to pass to the `signal.savgol_filter` function.
    """
    def __init__(self,window_length = 13, polyorder=2, **kwargs):
        self.window_length = window_length
        self.polyorder = polyorder
        self.kwargs = kwargs
    
    def transform(self, X):
        """
        Apply Savitzky-Golay filtering to the input matrix.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of data.

        Returns
        -------
        X_filtered : numpy.ndarray
            The filtered matrix of data.
        """
        return signal.savgol_filter(X, window_length = self.window_length, 
                             polyorder = self.polyorder, **self.kwargs)
    

    
class centralization():
    """
    A class for performing centralization on a matrix of data.

    Parameters
    ----------
    None

    Attributes
    ----------
    mean : numpy.ndarray
        The mean of each feature in the input matrix.
    """
    def __init__(self):
        pass
    
    def fit(self, X):
        """
        Compute the mean of each feature in the input matrix.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of data.

        Returns
        -------
        self : centralization
            The fitted centralization object.
        """
        self.mean = np.mean(X,axis=0)
        return self
    
    def transform(self, X):
        """
        Subtract the mean of each feature from the input matrix.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of data.

        Returns
        -------
        X_centered : numpy.ndarray
            The centered input matrix.
        """
        return X-self.mean[None,:]
    
    def fit_transform(self, X):
        """
        Compute the mean of each feature in the input matrix, and then subtract the mean from the input matrix.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of data.

        Returns
        -------
        X_centered : numpy.ndarray
            The centered input matrix.
        """
        return self.fit(X).transform(X)
    
 
class derivate():
    """
    A class for performing Savitzky-Golay filtering with derivative on a matrix of data.

    Parameters
    ----------
    deriv : int, optional
        The order of the derivative to compute (default is 1).
    window_length : int, optional
        The length of the filter window (default is 13).
    polyorder : int, optional
        The order of the polynomial used to fit the samples (default is 2).
    **kwargs : dict, optional
        Additional keyword arguments to pass to the `signal.savgol_filter` function.

    Attributes
    ----------
    deriv : int
        The order of the derivative to compute.
    window_length : int
        The length of the filter window.
    polyorder : int
        The order of the polynomial used to fit the samples.
    kwargs : dict
        Additional keyword arguments to pass to the `signal.savgol_filter` function.
    """
    def __init__(self,deriv = 1, window_length = 13, polyorder=2, **kwargs):
        self.deriv = deriv
        self.window_length = window_length
        self.polyorder = polyorder
        self.kwargs = kwargs
    
    def transform(self, X):
        """
        Apply Savitzky-Golay filtering with derivative to the input matrix.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of data.

        Returns
        -------
        X_filtered : numpy.ndarray
            The filtered matrix of data.
        """
        return signal.savgol_filter(X, deriv=self.deriv, window_length = self.window_length, 
                             polyorder = self.polyorder,
                             delta = 1.0, **self.kwargs)
    
    
class smooth():
    """
    A class for performing smoothing on a matrix of data using the Savitzky-Golay filter.

    Parameters
    ----------
    window_length : int, optional
        The length of the filter window (default is 13).
    polyorder : int, optional
        The order of the polynomial used to fit the samples (default is 2).
    **kwargs : dict, optional
        Additional keyword arguments to pass to the `signal.savgol_filter` function.

    Attributes
    ----------
    window_length : int
        The length of the filter window.
    polyorder : int
        The order of the polynomial used to fit the samples.
    kwargs : dict
        Additional keyword arguments to pass to the `signal.savgol_filter` function.
    """
    def __init__(self,window_length = 13, polyorder=2, **kwargs):
        self.window_length = window_length
        self.polyorder = polyorder
        self.kwargs = kwargs
    
    def transform(self, X):
        """
        Apply Savitzky-Golay smoothing to the input matrix.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix of data.

        Returns
        -------
        X_smoothed : numpy.ndarray
            The smoothed matrix of data.
        """
        return signal.savgol_filter(X,  window_length = self.window_length, 
                             polyorder = self.polyorder, deriv=0, **self.kwargs)