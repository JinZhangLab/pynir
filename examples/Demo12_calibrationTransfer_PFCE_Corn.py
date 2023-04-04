import sys
from pathlib import Path

pynir_path = str(Path(__file__).parent.parent / 'src')

if pynir_path not in sys.path:
    sys.path.append(pynir_path)
    print("add pynir path")

from pynir.utils import simulateNIR_calibrationTransfer
from pynir.Calibration import pls, regresssionReport
from pynir.CalibrationTransfer import NS_PFCE,SS_PFCE,FS_PFCE,MT_PFCE
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import loadmat

from sklearn.model_selection import train_test_split

import time


RawData = loadmat("./Data_Corn.mat")
wv = RawData["wv"].ravel()
Xcal1 = RawData["Xcal1"]
Xcal2 = RawData["Xcal2"]
Xcal3 = RawData["Xcal3"]
Xtrans1 = RawData["Xtrans1"]
Xtrans2 = RawData["Xtrans2"]
Xtrans3 = RawData["Xtrans3"]
Xtest1 = RawData["Xtest1"]
Xtest2 = RawData["Xtest2"]
Xtest3 = RawData["Xtest3"]
ycal = RawData["ycal"]
ytrans = RawData["ytrans"]
ytest = RawData["ytest"]


fig, ax = plt.subplots(3,sharex=True)
ax[0].plot(wv, np.transpose(Xcal1))
ax[0].set_title("First")

ax[1].plot(wv, np.transpose(Xcal2))
ax[1].set_title("Second")

ax[2].plot(wv, np.transpose(Xcal3))
ax[2].set_title("Third")

fig.supxlabel("Wavelength (nm)")
fig.supylabel("Intensity(A.U.)")
plt.show()

nComp = 4
plsModel1 = pls(nComp=nComp).fit(Xcal1, ycal)

yhat1 = plsModel1.predict(Xtest1,nComp=nComp)
yhat2= plsModel1.predict(Xtest2,nComp=nComp)
yhat3= plsModel1.predict(Xtest3,nComp=nComp)

fig, ax = plt.subplots(3,sharex=True,figsize=(8,16))
plsModel1.plot_prediction(ytest, yhat1,title = "First", ax = ax[0])
plsModel1.plot_prediction(ytest, yhat2, title= "Second", ax = ax[1])
plsModel1.plot_prediction(ytest, yhat3, title= "Third", ax = ax[2])

fig, ax = plt.subplots()
ax.plot(wv, plsModel1.model['B'][1:,-1])
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Regression Coefficients")

## PFCE setting
thres = 0.98
constrType = 3

## NS-PFCE
tic = time.time()
b1 = plsModel1.model['B'][:,-1]
NS_PFCE_model = NS_PFCE(thres=thres, constrType=constrType).fit(Xtrans1,Xtrans2,b1)
yhat2_NS_PFCE = NS_PFCE_model.transform(Xtest2)
NS_PFCE_model = NS_PFCE(thres=thres, constrType=constrType).fit(Xtrans1,Xtrans3,b1)
yhat3_NS_PFCE = NS_PFCE_model.transform(Xtest3)
plsModel1.plot_prediction(ytest, yhat2_NS_PFCE, title= "NS-PFCE_slave1")
plsModel1.plot_prediction(ytest, yhat3_NS_PFCE, title= "NS-PFCE_slave2")

fig, ax = plt.subplots()
ax.plot(wv, NS_PFCE_model.b2.x[1:])
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Regression Coefficients")
ax.set_title("NS-PFCE")
print("cost {:.2f} seconds for NS-PFCE".format(time.time()-tic))


## SS-PFCE
tic = time.time()
b1 = plsModel1.model['B'][:,-1]
SS_PFCE_model = SS_PFCE(thres=thres, constrType=constrType).fit(Xtrans2,ytrans,b1)
yhat2_SS_PFCE = SS_PFCE_model.transform(Xtest2)
SS_PFCE_model = SS_PFCE(thres=thres, constrType=constrType).fit(Xtrans3,ytrans,b1)
yhat3_SS_PFCE = SS_PFCE_model.transform(Xtest3)
plsModel1.plot_prediction(ytest, yhat2_SS_PFCE, title= "SS-PFCE_slave1")
plsModel1.plot_prediction(ytest, yhat3_SS_PFCE, title= "SS-PFCE_slave2")

fig, ax = plt.subplots()
ax.plot(wv, SS_PFCE_model.b2.x[1:])
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Regression Coefficients")
ax.set_title("SS-PFCE")
print("cost {:.2f} seconds for SS-PFCE".format(time.time()-tic))

## FS-PFCE
tic = time.time()
b1 = plsModel1.model['B'][:,-1]
FS_PFCE_model = FS_PFCE(thres=thres, constrType=constrType).fit(Xtrans1,Xtrans2,ytrans,b1)
yhat2_FS_PFCE = FS_PFCE_model.transform(Xtest2)
FS_PFCE_model = FS_PFCE(thres=thres, constrType=constrType).fit(Xtrans1,Xtrans3,ytrans,b1)
yhat3_FS_PFCE = FS_PFCE_model.transform(Xtest3)

plsModel1.plot_prediction(ytest, yhat2_FS_PFCE, title= "FS-PFCE_slave1")
plsModel1.plot_prediction(ytest, yhat3_FS_PFCE, title= "FS-PFCE_slave2")

fig, ax = plt.subplots()
ax.plot(wv, FS_PFCE_model.b2.x[1:])
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Regression Coefficients")
ax.set_title("FS-PFCE")
print("cost {:.2f} seconds for FS-PFCE".format(time.time()-tic))


## MT-PFCE
tic = time.time()
plsModel_global = pls(nComp=nComp).fit(np.vstack((Xcal1,Xtrans2,Xtrans3)),
                                       np.vstack((ycal,ytrans,ytrans)))
optLV_global = plsModel_global.get_optLV()
plsModel_global = pls(nComp=optLV_global).fit(np.vstack((Xcal1,Xtrans2,Xtrans3)),
                                       np.vstack((ycal,ytrans,ytrans)))

bglb = plsModel_global.model['B'][:,-1]
MT_PFCE_model = MT_PFCE(thres=thres, constrType=constrType)
MT_PFCE_model.fit([Xcal1,Xtrans2,Xtrans3],(ycal,ytrans,ytrans),bglb)
yhat1_MT_PFCE = MT_PFCE_model.transform(Xtest1,0) # task 1
yhat2_MT_PFCE = MT_PFCE_model.transform(Xtest2,1) # task 2
yhat3_MT_PFCE = MT_PFCE_model.transform(Xtest3,2) # task 3

fig, ax = plt.subplots(3,sharex=True,figsize=(8,16))
plsModel1.plot_prediction(ytest, yhat1_MT_PFCE, title= "MT-PFCE_First", ax= ax[0])
plsModel1.plot_prediction(ytest, yhat2_MT_PFCE, title= "MT-PFCE_Second", ax= ax[1])
plsModel1.plot_prediction(ytest, yhat3_MT_PFCE, title= "MT-PFCE_Second", ax= ax[2])

fig, ax = plt.subplots()
ax.plot(wv, MT_PFCE_model.B.x.reshape(3,-1).transpose()[1:,:])
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Regression Coefficients")
ax.set_title("MT-PFCE")
print("cost {:.2f} seconds for MT-PFCE".format(time.time()-tic))
