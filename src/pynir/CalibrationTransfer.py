# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:00:35 2022

@author: J Zhang (jzhang@chemoinfolab.com)
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD

from scipy.optimize import (BFGS, SR1, Bounds,
                            NonlinearConstraint, minimize)


class PDS():
    """
    Partial Direct Standardization (PDS) for spectral calibration.

    This class implements the PDS algorithm for spectral calibration, which is a method
    for transferring calibration models between instruments with different spectral
    characteristics.

    Parameters
    ----------
    halfWindowSize : int, optional
        The half window size for selecting the spectral bands.
    regType : str, optional
        The regression type to use for modeling the calibration transfer function.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the regression model.

    Attributes
    ----------
    Models : list
        A list of regression models for each spectral band.

    Notes
    -----
    This implementation is based on the algorithm described in:
    Li, H., Xu, Q., Liang, Y., & Ying, Y. (2010).
    Partial direct standardization for calibration transfer between near-infrared spectrometers.
    Analytica Chimica Acta, 665(1), 77-82.
    """
    def __init__(self, halfWindowSize = 7, regType = 'mlr',**kwargs):
        self.halfWindowSize = halfWindowSize
        self.regType = regType
        self.kwargs = kwargs

    def get_windowRange(halfWindowSize, nFeatrues, i):
        if i < halfWindowSize:
            band = np.arange(i)
        elif i > nFeatrues-halfWindowSize:
            band = np.arange(i)
        else:
            band = np.arange(i-halfWindowSize, i+halfWindowSize)
        return band

    def fit(self, X1, X2):
        """
        Fit the PDS model to the training data.

        Parameters
        ----------
        X1 : numpy.ndarray
            The standard spectra of the master instrument.
        X2 : numpy.ndarray
            The standard spectra of the slave instrument.

        Returns
        -------
        self : PDS
            The fitted PDS model.
        """
        if X1.shape != X2.shape:
            raise("The dimension of two spectral matrix doesn't match.")

        Models = []

        for i in range(X1.shape[1]):
            if i < self.halfWindowSize:
                X2i = X2[:,:2*self.halfWindowSize+1]
            elif i >= self.halfWindowSize and i < X2.shape[1]-self.halfWindowSize:
                X2i = X2[:,i-self.halfWindowSize:i+self.halfWindowSize+1]
            elif i > X2.shape[1]-self.halfWindowSize:
                X2i = X2[:,X2.shape[1]-2*self.halfWindowSize-1:]



            if self.regType == 'mlr':
                Models.append(LinearRegression().fit(X2i,X1[:,i][:,None]))
            elif self.regType == 'pls':
                Models.append(PLSRegression(n_components=self.n_components).fit(X2i,X1[:,i][:,None]))

        self.Models = Models
        return self

    def transform(self, X):
        """
        Apply the PDS model to new data.

        Parameters
        ----------
        X : numpy.ndarray
            The spectra to calibrate.

        Returns
        -------
        Xnew : numpy.ndarray
            The calibrated spectra.
        """
        Xnew = np.zeros(X.shape)
        for i in range(X.shape[1]):
            if i < self.halfWindowSize:
                Xi = X[:,:2*self.halfWindowSize+1]
            elif i >= self.halfWindowSize and i < X.shape[1]-self.halfWindowSize:
                Xi = X[:,i-self.halfWindowSize:i+self.halfWindowSize+1]
            elif i > X.shape[1]-self.halfWindowSize:
                Xi = X[:,X.shape[1]-2*self.halfWindowSize-1:]

            Xnew[:,i] = self.Models[i].predict(Xi).ravel()
        return Xnew


class SST():
    """
    Spectral Space Transformation (SST) for spectral calibration.

    This class implements the SST algorithm for spectral calibration, which is a method
    for transferring calibration models between instruments with different spectral
    characteristics.

    Parameters
    ----------
    n_components : int, optional
        The number of components to use for the truncated SVD.

    Attributes
    ----------
    F : numpy.ndarray
        The transformation matrix learned from the training data.
    
    Notes
    -----
    This implementation is based on the algorithm described in:
    Du W, Chen Z P, Zhong L J, et al. Maintaining the predictive abilities of multivariate calibration models by spectral space transformation[J]. Analytica Chimica Acta, 2011, 690(1): 64-70.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self,X1,X2):
        """
        Parameters
        ----------
        X1 : 2D array
            Standard spectra of the first instrument.

        X2 : 2D array
            Standard spectra of the second instrument.
        """
        svd = TruncatedSVD(n_components=self.n_components).fit(np.hstack((X1,X2)))
        P = svd.components_
        nFeature_X1 = X1.shape[1]
        W1 = P[:,:nFeature_X1]
        W2 = P[:,nFeature_X1:]
        self.F = np.dot(np.linalg.pinv(W2),  (W1-W2))
        return self

    def transform(self,X):
        return X + np.dot(X,self.F)


class BS():
    """
    Implementation of the Osborne and Fearn Back-Shift (BS) method for spectral calibration.

    This class implements the BS algorithm for spectral calibration, which is a method
    for transferring calibration models between instruments with different spectral
    characteristics.

    Notes
    ----------
    Osborne, B. G., & Fearn, T. (1983).
    Collaborative evaluation of universal calibrations for the measurement of protein and moisture
    in flour by near infrared reflectance. International Journal of Food Science & Technology, 18(4), 453-460.
    """
    def __init__(self):
        pass

    def fit(self, y1, y2):
        """
        Fit the BS model to the training data.

        Parameters
        ----------
        y1 : numpy.ndarray
            The predictions of standard spectra from the master instrument.
        y2 : numpy.ndarray
            The predictions of standard spectra from the slave instrument.

        Returns
        -------
        self : BS
            The fitted BS model.
        """
        self.lrModel = LinearRegression().fit(y2.reshape(-1,1), y1.reshape(-1,1))
        return self

    def transform(self, y2):
        """
        Apply the BS model to new prediction of spectra from slave instrument.

        Parameters
        ----------
        y2 : numpy.ndarray
            The predictions of spectra measured on the slave instrument.

        Returns
        -------
        y1 : numpy.ndarray
            The predicted reference values with correction
        """
        return self.lrModel.predict(y2.reshape(-1,1))


class NS_PFCE():
    """
    Non-Supervised Parameter-Free Framework for Calibration Enhancement (NS-PFCE) for spectral calibration enhancement.

    This class implements the NS-PFCE algorithm for spectral calibration enhancement, which is a method
    for transferring calibration models between instruments with different spectral
    characteristics.

    Parameters
    ----------
    thres : float, optional
        The threshold for tconstraint.
    constrType : int, optional
        The type of constraint to use for optimization.

    Attributes
    ----------
    b2 : numpy.ndarray
        The coefficients learned from the training data.
    
    Notes
    -----
    This implementation is based on the algorithm described in:
    [1] Zhang J., Zhou X, Li B. Y.*, PFCE2: A versatile parameter-free calibration enhancement framework for near-infrared spectroscopy, Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 2023, 301: 122978.
    [2] Zhang J., Li B. Y.*, Hu Y., Zhou L. X., Wang G. Z., Guo G., Zhang Q. H., Lei S. C.*, Zhang A. H., A Parameter-Free Framework for Calibration Enhancement of Near-Infrared Spectroscopy Based on Correlation Constraint, Anal. Chim. Acta, 2021, 1142: 169-178.
    """
    def __init__(self,thres = 0.98, constrType = 1):
        self.thres = thres
        self.constrType = constrType

    def fit(self, X1, X2, b1):
        """
        Fit the NS-PFCE model to the training data.

        Parameters
        ----------
        X1 : numpy.ndarray
            The standard spectra of the master instrument.
        X2 : numpy.ndarray
            The standard spectra of the slave instrument.
        b1 : numpy.ndarray
            The coefficients learned from the master instrument.

        Returns
        -------
        self : NS_PFCE
            The fitted NS-PFCE model for slave instrument.
        """
        ntask = 2
        lb = -np.inf
        ub = 0.0
        if self.constrType == 1:
            nlc = NonlinearConstraint(
                lambda b2:pfce_constr1(np.hstack((b1.reshape(-1,1), b2.reshape(-1,1))),
                                       self.thres), lb, ub)
        elif self.constrType == 2:
            nlc = NonlinearConstraint(
                lambda b2:pfce_constr2(np.hstack((b1.reshape(-1,1), b2.reshape(-1,1))),
                                       self.thres), lb, ub)
        elif self.constrType == 3:
            nlc = NonlinearConstraint(
                lambda b2:pfce_constr3(np.hstack((b1.reshape(-1,1), b2.reshape(-1,1))),
                                       self.thres), lb, ub)


        b2 = minimize(lambda b2:cost_NS_PFCE(b1,b2,X1,X2),
                      x0 = b1, method='SLSQP', constraints=[nlc])
        self.b2 = b2
        return self

    def transform(self, X):
        """
        Apply the NS-PFCE model to spectra measured on slave instruments.

        Parameters
        ----------
        X : numpy.ndarray
            The spectra to calibrate.

        Returns
        -------
        y : numpy.ndarray
            The prediction of spectra from slave instrument with the NS-PFCE enhanced model.
        """
        return np.dot(np.hstack((np.ones([X.shape[0],1]), X)),
                      np.reshape(self.b2.x,(-1,1)))

class SS_PFCE():
    """
    Semi-Supervised Parameter-Free Calibration Enhancement (SS-PFCE) framework for spectral calibration enhancement.

    This class implements the SS-PFCE algorithm for spectral calibration enhancement, which is a method
    for transferring calibration models between instruments with different spectral
    characteristics.

    Parameters
    ----------
    thres : float, optional
        The threshold for tconstraint.
    constrType : int, optional
        The type of constraint to use for optimization.

    Attributes
    ----------
    b2 : numpy.ndarray
        The coefficients learned from the training data.
    
    Notes
    -----
    This implementation is based on the algorithm described in:
    [1] Zhang J., Zhou X, Li B. Y.*, PFCE2: A versatile parameter-free calibration enhancement framework for near-infrared spectroscopy, Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 2023, 301: 122978.
    [2] Zhang J., Li B. Y.*, Hu Y., Zhou L. X., Wang G. Z., Guo G., Zhang Q. H., Lei S. C.*, Zhang A. H., A Parameter-Free Framework for Calibration Enhancement of Near-Infrared Spectroscopy Based on Correlation Constraint, Anal. Chim. Acta, 2021, 1142: 169-178.
    """
    def __init__(self,thres = 0.98, constrType = 1):
        self.thres = thres
        self.constrType = constrType

    def fit(self, X2, y, b1):
        """
        Fit the SS-PFCE model to the training data.

        Parameters
        ----------
        X2 : numpy.ndarray
            The standard spectra of the slave instrument.
        y : numpy.ndarray
            The reference values of the slave instrument.
        b1 : numpy.ndarray
            The coefficients learned from the master instrument.

        Returns
        -------
        self : SS_PFCE
            The fitted SS-PFCE model for slave instrument.
        """
        ntask = 2
        lb = -np.inf
        ub = 0.0
        if self.constrType == 1:
            nlc = NonlinearConstraint(
                lambda b2:pfce_constr1(np.hstack((b1.reshape(-1,1), b2.reshape(-1,1))),
                                       self.thres), lb, ub)
        elif self.constrType == 2:
            nlc = NonlinearConstraint(
                lambda b2:pfce_constr2(np.hstack((b1.reshape(-1,1), b2.reshape(-1,1))),
                                       self.thres), lb, ub)
        elif self.constrType == 3:
            nlc = NonlinearConstraint(
                lambda b2:pfce_constr3(np.hstack((b1.reshape(-1,1), b2.reshape(-1,1))),
                                       self.thres), lb, ub)
        else:
            raise("PFCE only support the constrint type of 1, 2, or 3")

        b2 = minimize(lambda b2:cost_SS_PFCE(b2,X2, np.ravel(y)),
                      x0 = b1, method='SLSQP', constraints=[nlc])
        self.b2 = b2
        return self

    def transform(self, X):
        """
        Apply the SS-PFCE model to spectra measured on slave instruments.

        Parameters
        ----------
        X : numpy.ndarray
            The spectra to calibrate.

        Returns
        -------
        y : numpy.ndarray
            The prediction of spectra from slave instrument with the SS-PFCE enhanced model.
        """
        return np.dot(np.hstack((np.ones([X.shape[0],1]), X)),
                      np.reshape(self.b2.x,(-1,1)))


class FS_PFCE():
    """
    Full-Supervised Parameter-Free Calibration Enhancement (FS-PFCE) framework for spectral calibration enhancement.

    This class implements the FS-PFCE algorithm for spectral calibration enhancement, which is a method
    for transferring calibration models between instruments with different spectral
    characteristics.

    Parameters
    ----------
    thres : float, optional
        The threshold for tconstraint.
    constrType : int, optional
        The type of constraint to use for optimization.

    Attributes
    ----------
    b2 : numpy.ndarray
        The coefficients learned from the training data.
    
    Notes
    -----
    This implementation is based on the algorithm described in:
    [1] Zhang J., Zhou X, Li B. Y.*, PFCE2: A versatile parameter-free calibration enhancement framework for near-infrared spectroscopy, Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 2023, 301: 122978.
    [2] Zhang J., Li B. Y.*, Hu Y., Zhou L. X., Wang G. Z., Guo G., Zhang Q. H., Lei S. C.*, Zhang A. H., A Parameter-Free Framework for Calibration Enhancement of Near-Infrared Spectroscopy Based on Correlation Constraint, Anal. Chim. Acta, 2021, 1142: 169-178.
    """
    def __init__(self,thres = 0.98, constrType = 1):
        self.thres = thres
        self.constrType = constrType

    def fit(self, X1, X2, y, b1):
        """
        Fit the FS-PFCE model to the training data.

        Parameters
        ----------
        X1 : numpy.ndarray
            The standard spectra of the master instrument.
        X2 : numpy.ndarray
            The standard spectra of the slave instrument.
        y : numpy.ndarray
            The reference values of the slave instrument.
        b1 : numpy.ndarray
            The coefficients learned from the master instrument.

        Returns
        -------
        self : FS_PFCE
            The fitted FS-PFCE model for slave instrument.
        """
        ntask = 2
        lb = -np.inf
        ub = 0.0
        if self.constrType == 1:
            nlc = NonlinearConstraint(
                lambda b2:pfce_constr1(np.hstack((b1.reshape(-1,1), b2.reshape(-1,1))),
                                       self.thres), lb, ub)
        elif self.constrType == 2:
            nlc = NonlinearConstraint(
                lambda b2:pfce_constr2(np.hstack((b1.reshape(-1,1), b2.reshape(-1,1))),
                                       self.thres), lb, ub)
        elif self.constrType == 3:
            nlc = NonlinearConstraint(
                lambda b2:pfce_constr3(np.hstack((b1.reshape(-1,1), b2.reshape(-1,1))),
                                       self.thres), lb, ub)

        b2 = minimize(lambda b2:cost_FS_PFCE(b1,b2,X1,X2,np.ravel(y)),
                      x0 = b1,  method='SLSQP', constraints=[nlc])
        self.b2 = b2
        return self

    def transform(self, X):
        """
        Apply the FS-PFCE model to spectra measured on slave instruments.

        Parameters
        ----------
        X : numpy.ndarray
            The spectra to calibrate.

        Returns
        -------
        y : numpy.ndarray
            The prediction of spectra from slave instrument with the FS-PFCE enhanced model.
        """
        return np.dot(np.hstack((np.ones([X.shape[0],1]), X)),
                      np.reshape(self.b2.x,(-1,1)))


class MT_PFCE():
    """
    Multi-Task Parameter-Free Calibration Enhancement (MT-PFCE) framework for spectral calibration enhancement.

    This class implements the MT-PFCE algorithm for spectral calibration enhancement, which is a method
    for transferring calibration models between instruments with different spectral
    characteristics.

    Parameters
    ----------
    thres : float, optional
        The threshold for tconstraint.
    constrType : int, optional
        The type of constraint to use for optimization.

    Attributes
    ----------
    B : numpy.ndarray
        The coefficients learned from the training data.
    ntask : int
        The number of tasks.

    Notes
    -----
    This implementation is based on the algorithm described in:
    [1] Zhang J., Zhou X, Li B. Y.*, PFCE2: A versatile parameter-free calibration enhancement framework for near-infrared spectroscopy, Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 2023, 301: 122978.
    [2] Zhang J., Li B. Y.*, Hu Y., Zhou L. X., Wang G. Z., Guo G., Zhang Q. H., Lei S. C.*, Zhang A. H., A Parameter-Free Framework for Calibration Enhancement of Near-Infrared Spectroscopy Based on Correlation Constraint, Anal. Chim. Acta, 2021, 1142: 169-178.
    """
    def __init__(self,thres = 0.98, constrType = 1):
        self.thres = thres
        self.constrType = constrType

    def fit(self, X, y, b1):
        """
        Fit the MT-PFCE model to the training data.

        Parameters
        ----------
        X : list of numpy.ndarray
            The standard spectra of the master instrument for each task.
        y : list of numpy.ndarray
            The reference values of the slave instrument for each task.
        b1 : numpy.ndarray
            The coefficients learned from the master instrument.

        Returns
        -------
        self : MT_PFCE
            The fitted MT-PFCE model for slave instrument.
        """
        ntask = len(X)
        y = [np.ravel(yi) for yi in y]
        self.ntask = ntask
        lb = -np.inf
        ub = 0.0
        if self.constrType == 1:
            nlc = NonlinearConstraint(
                lambda B:pfce_constr1(np.transpose(np.reshape(B,(ntask,-1))),
                                       self.thres), lb, ub)
        elif self.constrType == 2:
            nlc = NonlinearConstraint(
                lambda B:pfce_constr2(np.transpose(np.reshape(B,(ntask,-1))),
                                       self.thres), lb, ub)
        elif self.constrType == 3:
            nlc = NonlinearConstraint(
                lambda B:pfce_constr3(np.transpose(np.reshape(B,(ntask,-1))),
                                       self.thres), lb, ub)

        B = minimize(lambda B:cost_MT_PFCE(B.reshape(ntask,-1).transpose(),X,y),
                      x0 = np.tile(b1,ntask),method='SLSQP',constraints=[nlc])
        self.B = B
        return self

    def transform(self, X, itask):
        """
        Apply the MT-PFCE model to spectra measured on slave instruments.

        Parameters
        ----------
        X : numpy.ndarray
            The spectra to calibrate.
        itask : int
            The index of the task to apply the model to.

        Returns
        -------
        y : numpy.ndarray
            The prediction of spectra from ith task with the MT-PFCE enhanced model.
        """
        return np.dot(np.hstack((np.ones([X.shape[0],1]), X)),
                      ((self.B.x).reshape(self.ntask,-1).transpose()[:,itask]).reshape((-1,1)))


def cost_NS_PFCE(b1, b2, X1, X2):
    yhat1 = np.dot(np.hstack((np.ones([X1.shape[0],1]), X1)), b1)
    yhat2 = np.dot(np.hstack((np.ones([X2.shape[0],1]), X2)), b2)
    error = yhat1-yhat2
    return np.sum(error**2)



def cost_SS_PFCE(b2, X2, y):
    yhat2 = np.dot(np.hstack((np.ones([X2.shape[0],1]), X2)), b2)
    error = y-yhat2
    return np.sum(error**2)


def cost_FS_PFCE(b1, b2, X1, X2, y):
    yhat1 = np.dot(np.hstack((np.ones([X1.shape[0],1]), X1)), b1)
    yhat2 = np.dot(np.hstack((np.ones([X2.shape[0],1]), X2)), b2)
    error1 = yhat1-yhat2
    error2 = y-yhat2
    return np.sum(error1**2/len(error1)+error2**2/len(error2))


def cost_MT_PFCE(B, X, y):
    Error = []
    for i in range(B.shape[1]):
        yhati = np.dot(np.hstack((np.ones([X[i].shape[0],1]), X[i])), B[:,i])
        errori = y[i]-yhati
        Error.append(np.mean(errori**2))
    return np.sum(Error)


def pfce_constr1(B,thres):
    corr = np.corrcoef((B[1:,].transpose()))
    constr_cost  = thres - np.sum(np.triu(corr,1))/np.sum(np.arange(B.shape[1]))
    return constr_cost

def pfce_constr2(B,thres):
    B = B[1:,]
    Vars = np.diag(np.dot(B.transpose(),B))
    VarErr = 0
    for i in range(B.shape[1]):
        Err = B-B[:,i][:,None]
        Erri = Err;
        Erri = np.delete(Erri, i, axis =1)
        Varsi = Vars
        Varsi = np.delete(Varsi, i)
        VarErr += np.sum(Erri**2,axis = 0)/np.sqrt(Vars[i]*Varsi)
    dist = VarErr/(B.shape[1] * (B.shape[1]-1))
    constr_cost = thres - 1 + dist
    return constr_cost

def pfce_constr3(B,thres):
    B = B[1:,]
    norm1s = np.sum(np.abs(B),axis=0)
    norm1Err = 0
    for i in range(B.shape[1]):
        Err = B-B[:,i][:,None]
        Erri = Err
        Erri = np.delete(Erri, i,axis = 1)
        norm1si = norm1s
        norm1si = np.delete(norm1si, i)
        norm1Err += np.sum(np.abs(Erri),axis = 0)/np.sqrt(norm1s[i]*norm1si)
    dist = norm1Err/(B.shape[1] * (B.shape[1]-1))
    constr_cost = thres - 1 + dist
    return constr_cost
