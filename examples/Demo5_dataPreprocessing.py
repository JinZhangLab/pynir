import numpy as np
import matplotlib.pyplot as plt

from pynir.utils import simulateNIR

from pynir.Preprocessing import snv,cwt,msc,SG_filtering

# simulate NIR data
X,y,wv = simulateNIR(nSample=200,n_components=10,noise=1e-5)

X_snv = snv().fit_transform(X)
X_cwt = cwt(wavelet="mexh",scale=20).transform(X)
X_msc = msc().fit_transform(X)
X_sg = SG_filtering(deriv=1).transform(X)

fig, ax = plt.subplots(5,sharex=True)
ax[0].plot(wv,np.transpose(X))
ax[0].set_title("Raw spectra")

ax[1].plot(wv,np.transpose(X_snv))
ax[1].set_title("Spectra preprocessed with snv")

ax[2].plot(wv,np.transpose(X_cwt))
ax[2].set_title("Spectra preprocessed with cwt")

ax[3].plot(wv,np.transpose(X_msc))
ax[3].set_title("Spectra preprocessed with msc")

ax[4].plot(wv,np.transpose(X_sg))
ax[4].set_title("Spectra preprocessed with SG filtering")


plt.show()
