import numpy as np
import matplotlib.pyplot as plt
from pynir.utils import simulateNIR

# Generating X, y and wv variables using simulateNIR function
X,y,wv = simulateNIR(nSample=200,n_components=10,noise=1e-5)


fig, ax = plt.subplots(2)
# Plotting intensity values against wavelength
ax[0].plot(wv,np.transpose(X))
ax[0].set_xlabel("wavelength (nm)")
ax[0].set_ylabel("Intesntiy (a.u.)")

# Plotting histogram of reference values
ax[1].hist(y)
ax[1].set_xlabel("Reference values")
ax[1].set_ylabel("Count")
