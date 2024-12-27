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

from pynir.OutlierDetection import outlierDetection_PLS

# simulate NIR data
X,y,wv = simulateNIR(nSample=200,n_components=10,noise=1e-5)

ODModel = outlierDetection_PLS(ncomp=3)
Q, Tsq, Q_conf, Tsq_conf, idxOutlier = ODModel.fit(X, y).detect(X,y)
ODModel.plot_HotellingT2_Q(Q, Tsq, Q_conf, Tsq_conf)
