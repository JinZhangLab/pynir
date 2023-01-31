#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019.4.30
# @Author  : FrankEl
# @File    : Feature_selection_demo_rt.py

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from numpy.linalg import matrix_rank as rank
from pynir.FeatureSelection import MSVC
from pynir.utils import simulateNIR

if __name__ == "__main__":
    # set your number of component in PLS caliration, and 7 is default
    ncomp = 7

    # simulate NIR data
    X,y,wv = simulateNIR(nSample=200,n_components=10,noise=1e-5)



    # Building normal pls model
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2)
    plsModel = PLSRegression(n_components=ncomp)
    plsModel.fit(Xtrain, Ytrain)
    T, P, U, Q, W, C, beta = plsModel.x_scores_, plsModel.x_loadings_, plsModel.y_scores_, plsModel.y_loadings_, plsModel.x_weights_, plsModel.y_weights_, plsModel.coef_
    plt.plot(wv, beta[0:])
    plt.xlabel("Wavelength")
    plt.ylabel("Intensity")
    plt.title("Regression Coefficients")
    plt.show()

    # Prediction result of pls model
    Ytrain_hat = plsModel.predict(Xtrain)
    Ytest_hat = plsModel.predict(Xtest)
    plt.plot([Ytrain.min(), Ytrain.max()], [Ytrain.min(), Ytrain.max()], 'k--', lw=4)
    plt.scatter(Ytrain, Ytrain_hat, marker='*')
    plt.scatter(Ytest, Ytest_hat, marker='*')
    plt.xlabel("Prediction")
    plt.ylabel("Reference")
    plt.title("Prediction of normal pls model")
    plt.show()

    # C value of MSVC
    # It will take a few minits due to the intensitve calculation of C values
    vcModel = MSVC(Xtrain, Ytrain, ncomp, nrep=500)
    vcModel.calcCriteria()
    plt.imshow(vcModel.criteria, aspect='auto')
    plt.xlabel("Wavelength")
    plt.ylabel("Iteration")
    plt.title("C value")
    plt.show()

    # Feature ranking efficienty by stability of VC
    vcModel.evalCriteria(cv=3)
    plt.plot(vcModel.featureR2)
    plt.xlabel("Wavelength")
    plt.ylabel("Intensity")
    plt.title("R2")
    plt.show()

    # Prediction results after feature selection by VC
    XtrainNew, XtestNew = vcModel.cutFeature(Xtrain, Xtest)
    plsModelNew = PLSRegression(n_components=min([ncomp, rank(XtrainNew)]))
    plsModelNew.fit(XtrainNew, Ytrain)
    YtrainNew_hat = plsModelNew.predict(XtrainNew)
    YtestNew_hat = plsModelNew.predict(XtestNew)
    plt.plot([Ytrain.min(), Ytrain.max()], [Ytrain.min(), Ytrain.max()], 'k--', lw=4)
    plt.scatter(Ytrain, YtrainNew_hat, marker='*')
    plt.scatter(Ytest, YtestNew_hat, marker='*')
    plt.xlabel("Prediction")
    plt.ylabel("Reference")
    plt.title("Prediction after MSVC")
    plt.show()
