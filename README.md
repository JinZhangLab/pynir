# NIR calibration toolbox in python

This is a Python library for handling Near infrared (NIR) spectral calibration.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install [pynir](https://pypi.org/project/pynir/). 

```bash
pip install pynir
```

In addition, we also provide an online version at [this link](https://nir.chemoinfolab.com)

## Supported Algorithms and Modules

- **Readers**: Import .CSV and .SPA files through `InnoSpectraNIRReader` and `spaReader`.
- **Regression**: PLS, etc.
- **Classification**: PLSDA, etc.
- **Feature Selection**: MCUVE, RT, etc.
- **Outlier Detection**
- **Calibration Transfer**: PDS, SST, BS, PFCE variants
- **Data Preprocessing**: SNV, CWT, MSC, etc.

## Usage

### Simulate NIR spectra (spc) and reference values (conc)

```python
from pynir.utils import simulateNIR

spc, conc = simulateNIR()
```

### Regression

```python
from pynir.utils import simulateNIR
from pynir.Calibration import pls

# establish PLS model
n_components = 10
plsModel = pls(n_components = n_components)
plsModel.fit(X,y)

yhat = plsModel.predict(X)
```

### Classification

```python

# simulate NIR data
from pynir.utils import simulateNIR
from pynir.Calibration import plsda

nclass = 4
X,y,wv = simulateNIR(nSample=200,n_components=10,refType=nclass, noise=1e-5)

# estabilish PLS model
n_components = 10
plsdaModel = plsda(n_components = n_components)
plsdaModel.fit(X,y)

yhat = plsdaModel.predict(X)

```

### Feature selection

```python
# Feature selection
from pynir.utils import simulateNIR
from pynir.Calibration import pls
from pynir.FeatureSelection import MCUVE

# simulate NIR data
X,y,wv = simulateNIR(nSample=200,n_components=10,noise=1e-5)

mcModel = MCUVE(X, y, n_components, nrep=nrep).fit()
featureSelected_MC_UVE = mcModel.featureRank[:nSel]

```

### Outlier detection

```python
import numpy as np
import matplotlib.pyplot as plt

from pynir.utils import simulateNIR

from pynir.OutlierDetection import outlierDetection_PLS

# simulate NIR data
X,y,wv = simulateNIR(nSample=200,n_components=10,noise=1e-5)

ODModel = outlierDection_PLS(ncomp=3)
Q, Tsq, Q_conf, Tsq_conf, idxOutlier = ODModel.fit(X, y).detect(X,y)
ODModel.plot_HotellingT2_Q(Q, Tsq, Q_conf, Tsq_conf)
```

### Calibration Transfer

```python
from pynir.utils import simulateNIR_calibrationTransfer
from pynir.Calibration import pls, regresssionReport
from pynir.CalibrationTransfer import PDS,SST, BS
import matplotlib.pyplot as plt
import numpy as np

# Simulate NIR spectra for calibration transfer
nSample = 100
X1, X2, y, wv = simulateNIR_calibrationTransfer(nSample=nSample,n_components=10,shifts=5e1)
idxTrain,idxTest = train_test_split(np.arange(nSample),test_size=0.6)
idxTransfer,idxTest = train_test_split(idxTest,test_size=0.5)


# Multivariate calibration
n_components = 7
plsModel1 = pls(n_components=n_components).fit(X1[idxTrain,:], y[idxTrain])

yhat1 = plsModel1.predict(X1[idxTest,:],n_components=n_components)
yhat2= plsModel1.predict(X2[idxTest,:],n_components=n_components)

fig, ax = plt.subplots(2,sharex=True,figsize=(8,16))
plsModel1.plot_prediction(y[idxTest], yhat1,title = "First", ax = ax[0])
plsModel1.plot_prediction(y[idxTest], yhat2, title= "Second", ax = ax[1])

# Calibration transfer on spectra
## PDS
X2_PDS = PDS(halfWindowSize=3).fit(X1[idxTransfer,:], X2[idxTransfer,:]).transform(X2[idxTest,:])
yhat2_PDS= plsModel1.predict(X2_PDS,n_components=n_components)
plsModel1.plot_prediction(y[idxTest], yhat2_PDS, title= "PDS")

fig, ax = plt.subplots()
ax.plot(wv, np.transpose(X2_PDS))


## SST
X2_SST = SST(n_components=n_components).fit(X1[idxTransfer,:], X2[idxTransfer,:]).transform(X2[idxTest,:])
yhat2_SST= plsModel1.predict(X2_SST,n_components=n_components)
plsModel1.plot_prediction(y[idxTest], yhat2_SST, title= "SST")

fig, ax = plt.subplots()
ax.plot(wv, np.transpose(X2_SST))


# Calibration transfer on prediction
## BS
yhat2_BS = BS().fit(yhat1, yhat2).transform(yhat2)
plsModel1.plot_prediction(y[idxTest], yhat2_BS, title= "BS")

```

### Calibration Enhancement

```python
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


n_components = 7
plsModel1 = pls(n_components=n_components).fit(X1[idxTrain,:], y[idxTrain])



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

```

## Demo

First, execute

```bash
git clone https://github.com/JinZhangLab/pynir.git
cd ./pynir/examples
```

Then, execute code in your python coding environment or just in terminal as follows:

```bash
python Demo1_SimulateNIR.py
python Demo2_Regression.py
python Demo3_Binary_Classification.py
python Demo4_Multiclass_Classification.py
python Demo5_dataPreprocessing.py
python Demo6_outlierDetection.py
python Demo7_FeatureSelection_oneStep
python Demo8_FeatureSelection_multiSteps.py
python Demo9_calibrationTransfer.py
python Demo10_calibrationTransfer_PFCE_simulateNIR.py
python Demo11_calibrationTransfer_PFCE_Tablet.py
python Demo12_calibrationTransfer_PFCE_Corn.py
```

## Ref

- Zhang, J.; Cui, X. Y.; Cai, W. S.; Shao, X. G., A variable importance criterion for variable selection in near-infrared spectral analysis. Sci. China Chem. 2018, 62, 271-79.[link](https://link.springer.com/article/10.1007%2Fs11426-018-9368-9)

- Zhang J., Li B. Y., Hu Y., Zhou L. X., Wang G. Z., Guo G., Zhang Q. H., Lei S. C., Zhang A. H. A parameter-free framework for calibration enhancement of near-infrared spectroscopy based on correlation constraint [J]. Analytica Chimica Acta, 2021, 1142: 169-178.
  [link](<https://linkinghub.elsevier.com/retrieve/pii/S0003-2670(20)31110-7>)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
