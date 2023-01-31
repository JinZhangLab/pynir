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
from pynir.Calibration import plsda
from pynir.Calibration import binaryClassificationReport,plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# simulate NIR data
X,y,wv = simulateNIR(nSample=200,nComp=10,refType=2, noise=1e-5)

Xtrain, Xtest, ytrain,ytest = train_test_split(X,y,test_size=0.2)

# estabilish PLS model
nComp = 10
plsdaModel = plsda(n_components = nComp).fit(Xtrain,ytrain)

# 10 fold cross validation for selecting optimal nlv
accuracy_cv = []
yhat_cv  = plsdaModel.crossValidation_predict(nfold = 10)
for i in range(nComp):
    report_cv = binaryClassificationReport(ytrain, yhat_cv[:,i])
    accuracy_cv.append(report_cv["accuracy"])


fig,ax = plt.subplots()
ax.plot(np.arange(nComp)+1,accuracy_cv, marker = "*",label = "Accuracy$_c$$_v$")
ax.set_xlabel("nLV")
ax.set_ylabel("Accuracy")
ax.legend()
plt.show()


optLV = plsdaModel.get_optLV()  # optimized nLV based on cross validation

plsdaModel_opt = plsda(n_components = optLV)
plsdaModel_opt.fit(Xtrain,ytrain)
yhat_train_opt = plsdaModel_opt.predict(Xtrain)
yhat_test_opt = plsdaModel_opt.predict(Xtest)

cm_train_opt = confusion_matrix(y_true=ytrain, y_pred=yhat_train_opt)
cm_test_opt = confusion_matrix(y_true=ytest, y_pred=yhat_test_opt)



plot_confusion_matrix(cm_train_opt,np.unique(y),normalize=False,
                      title="Confusion matrix for prediction on training set")
plot_confusion_matrix(cm_test_opt,np.unique(y),normalize=False,
                      title="Confusion matrix for prediction on testing set")

vip =plsdaModel_opt.get_vip()
fig,ax = plt.subplots()
ax.plot(vip)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

q2, r2, permutation_ratios = plsdaModel_opt.permutation_test(X, y, n_repeats = 1000, n_jobs=-1)

# Calculate the observed Q2 and R2 values
y_pred = cross_val_predict(plsdaModel_opt, X, y, cv=10)
q2_observed = plsdaModel_opt.score(X, y)
r2_observed = r2_score(y, y_pred)

permutation_ratios = np.append(permutation_ratios,0)
q2 = np.append(q2,q2_observed)
r2 = np.append(r2,r2_observed)

# Fit a linear regression model to the permutation test data
q2_coef, q2_intercept = np.polyfit(permutation_ratios, q2, 1)
r2_coef, r2_intercept = np.polyfit(permutation_ratios, r2, 1)

# Calculate the fitted line values
q2_fit = q2_coef * permutation_ratios + q2_intercept
r2_fit = r2_coef * permutation_ratios + r2_intercept

# Plot the fitted lines
fig,ax = plt.subplots()
plt.scatter(1-permutation_ratios, q2, label='Q2')
plt.scatter(1-permutation_ratios, r2, label='R2')
plt.plot(1-permutation_ratios, q2_fit, '--', color='black', label='Q2 Fit')
plt.plot(1-permutation_ratios, r2_fit, '--', color='gray', label='R2 Fit')


plt.xlabel('Retain Ratio')
plt.ylabel('Model Performance')
plt.legend()
plt.show()
