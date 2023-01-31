#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019.4.30
# @Author  : FrankEl
# @File    : Feature_selection_demo_mcuve.py

import matplotlib.pyplot as plt
import numpy as np

from pynir.utils import simulateNIR
from pynir.Calibration import pls
from pynir.FeatureSelection import MCUVE,RT,VC,plotFeatureSelection

# simulate NIR data
X,y,wv = simulateNIR(nSample=200,n_components=10,noise=1e-5)



# estabilish PLS model
n_components = 10
plsModel = pls(n_components = n_components)
plsModel.fit(X,y)

# 10 fold cross validation for selecting optimal n_components
yhat = plsModel.predict(X,n_components=7)

fig, ax = plt.subplots(2)
ax[0].plot(wv, plsModel.model["B"][1:,-1])
ax[0].set_xlabel("Wavelength")
ax[0].set_ylabel("Intensity")
ax[0].set_title("Regression Coefficients")
plsModel.plot_prediction(y, yhat,ax = ax[1])


nSel = 100
nrep = 500
# Feature selection
vip = plsModel.get_vip()
featureSelected_vip = np.where(vip>1)

mcModel = MCUVE(X, y, n_components, nrep=nrep).fit()
featureSelected_MC_UVE = mcModel.featureRank[:nSel]

rtModel = RT(X, y, n_components, nrep=nrep).fit()
featureSelected_RT = rtModel.featureRank[:nSel]

vcModel = VC(X, y, n_components, nrep=nrep).fit()
featureSelected_VC = vcModel.featureRank[:nSel]



plotFeatureSelection(wv, X,
                     [featureSelected_vip,
                         featureSelected_MC_UVE,
                         featureSelected_RT,
                         featureSelected_VC],
                     methodNames=["VIP","MC-UVE","RT","VC"])

fig, ax = plt.subplots()
mc = mcModel.criteria
rt = rtModel.criteria
vc = vcModel.criteria
vip = (vip - min(vip))/(max(vip)-min(vip))
mc = (mc-min(mc))/(max(mc)-min(mc))
rt = (rt-min(rt))/(max(rt)-min(rt))
vc = (vc-min(vc))/(max(vc)-min(vc))
ax.plot(wv,vip, label ='VIP')
ax.plot(wv,mc, label ='MC-UVE')
ax.plot(wv,rt, label ='RT')
ax.plot(wv,vc, label ='VC')
plt.legend()
