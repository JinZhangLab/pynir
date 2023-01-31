import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
pynir_dir = os.path.join(parent_dir, 'src')
if pynir_dir not in sys.path:
    sys.path.insert(0, pynir_dir)


import numpy as np
import matplotlib.pyplot as plt

from pynir.utils import simulateNIR
from pynir.Calibration import pls
from pynir.Calibration import regresssionReport

from sklearn.model_selection import train_test_split

# simulate NIR data
X,y,wv = simulateNIR(nSample=200,n_components=10,noise=1e-5)

Xtrain, Xtest, ytrain,ytest = train_test_split(X,y,test_size=0.2)

# estabilish PLS model
n_components = 10
plsModel = pls(n_components = n_components)
plsModel.fit(Xtrain,ytrain)

# 10 fold cross validation for selecting optimal n_components
yhat_train = plsModel.predict(Xtrain,n_components=np.arange(n_components)+1)
yhat_cv    = plsModel.crossValidation_predict(nfold = 10)
yhat_test  = plsModel.predict(Xtest,n_components=np.arange(n_components)+1)

rmsec = []
rmsecv = []
rmsep = []

r2c = []
r2cv = []
r2p = []

for i in range(n_components):
    report_train = regresssionReport(ytrain, yhat_train[:,i])
    report_cv = regresssionReport(ytrain, yhat_cv[:,i])
    report_test = regresssionReport(ytest, yhat_test[:,i])

    rmsec.append(report_train["rmse"])
    rmsecv.append(report_cv["rmse"])
    rmsep.append(report_test["rmse"])

    r2c.append(report_train["r2"])
    r2cv.append(report_cv["r2"])
    r2p.append(report_test["r2"])

fig,ax = plt.subplots()
ax.plot(np.arange(n_components)+1,rmsec, marker = ".",label = "RMSEC")
ax.plot(np.arange(n_components)+1,rmsecv, marker = "*",label = "RMSECV")
ax.plot(np.arange(n_components)+1,rmsep, marker = "^",label = "RMSEP")
ax.set_xlabel("n_components")
ax.set_ylabel("RMSE")
ax.legend()
plt.show()

optLV = plsModel.get_optLV()  # optimized n_components based on cross validation
yhat_test_opt = plsModel.predict(Xtest, n_components = optLV)

fig, ax = plt.subplots()
ax.plot(ytest, yhat_test_opt, "*", label = "Optimal prediction of test set")
ax.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], label = "y = x")
ax.set_xlabel("Reference")
ax.set_ylabel("Prediction")
ax.legend()
plt.show()


vip = plsModel.get_vip()
fig, ax = plt.subplots()
ax.plot(vip)

fig, ax = plt.subplots()
ax.plot(wv,plsModel.model["B"][1:,optLV-1], label = "PLS regresssion coefficients")
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Coefficients")
ax.legend()
plt.show()
