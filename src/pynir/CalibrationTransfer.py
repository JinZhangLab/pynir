# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:00:35 2022

@author: chinn
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD

from scipy.optimize import (BFGS, SR1, Bounds,
                            NonlinearConstraint, minimize)


class PDS():
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
        Parameters
        ----------
        X1 : 2D array
            Standard spectra of master.

        X2 : 2D array
            Standard spectra of master.
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
    Ref Osborne B. G., Fearn T. Collaborative evaluation of universal
        calibrations for the measurement of protein and moisture in flour by near
        infrared reflectance [J]. International Journal of Food Science & Technology,
        1983, 18(4): 453-460.
    """
    def __init__(self):
        pass

    def fit(self, y1, y2):
        self.lrModel = LinearRegression().fit(y2.reshape(-1,1), y1.reshape(-1,1))
        return self

    def transform(self, y2):
        return self.lrModel.predict(y2.reshape(-1,1))


class NS_PFCE():
    def __init__(self,thres = 0.98, constrType = 1):
        self.thres = thres
        self.constrType = constrType

    def fit(self, X1, X2, b1):
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
        return np.dot(np.hstack((np.ones([X.shape[0],1]), X)),
                      np.reshape(self.b2.x,(-1,1)))

class SS_PFCE():
    def __init__(self,thres = 0.98, constrType = 1):
        self.thres = thres
        self.constrType = constrType

    def fit(self, X2, y, b1):
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
        return np.dot(np.hstack((np.ones([X.shape[0],1]), X)),
                      np.reshape(self.b2.x,(-1,1)))


class FS_PFCE():
    def __init__(self,thres = 0.98, constrType = 1):
        self.thres = thres
        self.constrType = constrType

    def fit(self, X1, X2, y, b1):
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
        return np.dot(np.hstack((np.ones([X.shape[0],1]), X)),
                      np.reshape(self.b2.x,(-1,1)))


class MT_PFCE():
    def __init__(self,thres = 0.98, constrType = 1):
        self.thres = thres
        self.constrType = constrType

    def fit(self, X, y, b1):
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
