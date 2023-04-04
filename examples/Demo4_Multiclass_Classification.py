import numpy as np
import matplotlib.pyplot as plt

from pynir.utils import simulateNIR
from pynir.Calibration import plsda
from pynir.Calibration import multiClassificationReport,plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# simulate NIR data
nclass = 4
X,y,wv = simulateNIR(nSample=200,n_components=10,refType=nclass, noise=1e-5)

Xtrain, Xtest, ytrain,ytest = train_test_split(X,y,test_size=0.2)

# estabilish PLS model
n_components = 10
plsdaModel = plsda(n_components = n_components)
plsdaModel.fit(Xtrain,ytrain)

# 10 fold cross validation for selecting optimal nlv
accuracy_cv = []
yhat_cv  = plsdaModel.crossValidation_predict(nfold = 10)
for i in range(n_components):
    report_cv = multiClassificationReport(ytrain, yhat_cv[:,i])
    accuracy_cv.append(np.mean([rep["accuracy"] for rep in report_cv.values()]))


fig,ax = plt.subplots()
ax.plot(np.arange(n_components)+1,accuracy_cv, marker = "*",label = "Accuracy$_c$$_v$")
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
