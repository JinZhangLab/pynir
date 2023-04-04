import numpy as np
import matplotlib.pyplot as plt

from pynir.utils import simulateNIR

from pynir.OutlierDection import outlierDection_PLS

# simulate NIR data
X,y,wv = simulateNIR(nSample=200,n_components=10,noise=1e-5)

ODModel = outlierDection_PLS(ncomp=3)
Q, Tsq, Q_conf, Tsq_conf, idxOutlier = ODModel.fit(X, y).detect(X,y)
ODModel.plot_HotellingT2_Q(Q, Tsq, Q_conf, Tsq_conf)
