from pynir.utils import simulateNIR_calibrationTransfer
from pynir.Calibration import pls, regresssionReport
from pynir.CalibrationTransfer import NS_PFCE,SS_PFCE,FS_PFCE,MT_PFCE
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

import time


nSample = 100
X1, X2, y, wv = simulateNIR_calibrationTransfer(nSample=nSample,n_components=10,shifts=5e1)
idxTrain,idxTest = train_test_split(np.arange(nSample),test_size=0.6)
idxTransfer,idxTest = train_test_split(idxTest,test_size=0.5)


fig, ax = plt.subplots(2,sharex=True)
ax[0].plot(wv, np.transpose(X1))
ax[0].set_title("First")

ax[1].plot(wv, np.transpose(X2))
ax[1].set_title("Second")

fig.supxlabel("Wavelength (nm)")
fig.supylabel("Intensity(A.U.)")
plt.show()

n_components = 7
plsModel1 = pls(n_components=n_components).fit(X1[idxTrain,:], y[idxTrain])

yhat1 = plsModel1.predict(X1[idxTest,:],n_components=n_components)
yhat2= plsModel1.predict(X2[idxTest,:],n_components=n_components)

fig, ax = plt.subplots(2,sharex=True,figsize=(8,16))
plsModel1.plot_prediction(y[idxTest], yhat1,title = "First", ax = ax[0])
plsModel1.plot_prediction(y[idxTest], yhat2, title= "Second", ax = ax[1])

fig, ax = plt.subplots()
ax.plot(wv, plsModel1.model['B'][1:,-1])
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Regression Coefficients")


## PFCE
thres = 0.98
constrType = 1

tic = time.time()
b1 = plsModel1.model['B'][:,-1]
NS_PFCE_model = NS_PFCE(thres=thres, constrType=constrType).fit(X1[idxTransfer,:],X2[idxTransfer,:],b1)
yhat2_NS_PFCE = NS_PFCE_model.transform(X2[idxTest,:])
plsModel1.plot_prediction(y[idxTest], yhat2_NS_PFCE, title= "NS-PFCE")

fig, ax = plt.subplots()
ax.plot(wv, NS_PFCE_model.b2.x[1:])
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Regression Coefficients")
ax.set_title("NS-PFCE")
print("cost {:.2f} seconds for NS-PFCE".format(time.time()-tic))


tic = time.time()
b1 = plsModel1.model['B'][:,-1]
SS_PFCE_model = SS_PFCE(thres=thres, constrType=constrType).fit(X2[idxTransfer,:],y[idxTransfer],b1)
yhat2_SS_PFCE = SS_PFCE_model.transform(X2[idxTest,:])
plsModel1.plot_prediction(y[idxTest], yhat2_SS_PFCE, title= "SS-PFCE")

fig, ax = plt.subplots()
ax.plot(wv, SS_PFCE_model.b2.x[1:])
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Regression Coefficients")
ax.set_title("SS-PFCE")
print("cost {:.2f} seconds for SS-PFCE".format(time.time()-tic))


tic = time.time()
b1 = plsModel1.model['B'][:,-1]
FS_PFCE_model = FS_PFCE(thres=thres, constrType=constrType).fit(X1[idxTransfer,:],X2[idxTransfer,:],y[idxTransfer],b1)
yhat2_FS_PFCE = FS_PFCE_model.transform(X2[idxTest,:])
plsModel1.plot_prediction(y[idxTest], yhat2_FS_PFCE, title= "FS-PFCE")

fig, ax = plt.subplots()
ax.plot(wv, FS_PFCE_model.b2.x[1:])
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Regression Coefficients")
ax.set_title("FS-PFCE")
print("cost {:.2f} seconds for FS-PFCE".format(time.time()-tic))


tic = time.time()
b1 = plsModel1.model['B'][:,-1]
MT_PFCE_model = MT_PFCE(thres=thres, constrType=constrType)
MT_PFCE_model.fit([X1[idxTrain,:],X2[idxTransfer,:]],(y[idxTrain],y[idxTransfer]),b1)
yhat1_MT_PFCE = MT_PFCE_model.transform(X1[idxTest,:],0) # task 1
yhat2_MT_PFCE = MT_PFCE_model.transform(X2[idxTest,:],1) # task 2

fig, ax = plt.subplots(2,sharex=True,figsize=(8,16))
plsModel1.plot_prediction(y[idxTest], yhat1_MT_PFCE, title= "MT-PFCE_First", ax= ax[0])
plsModel1.plot_prediction(y[idxTest], yhat2_MT_PFCE, title= "MT-PFCE_Second", ax= ax[1])

fig, ax = plt.subplots()
ax.plot(wv, MT_PFCE_model.B.x.reshape(2,-1)[:,1:].transpose())
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Regression Coefficients")
ax.set_title("MT-PFCE")
print("cost {:.2f} seconds for MT-PFCE".format(time.time()-tic))
