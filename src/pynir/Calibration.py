# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:00:35 2022

@author: chinn
"""
import numpy as np
from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_predict

import matplotlib.pyplot as plt


class pls:
    def __init__(self, n_components = 2):
        """
        Initialize the number of components in PLS model.
        """

        self.n_components = n_components

    def fit(self, X, y):
        """
        Fit the PLS model to training data.

        Parameters:
        X (np.array): independent variable(s) of the training data
        y (np.array): dependent variable of the training data

        Returns:
        self: the PLS model
        """
        self.X = X
        self.y = y
        meanX = np.mean(X,axis = 0)
        meany = np.mean(y)
        Xcentered = X - meanX
        ycentered = y - meany
        model = simpls(Xcentered, ycentered, self.n_components)
        meanX_hat = -1 * np.dot(meanX, model['B']) + meany
        model['B'] = np.append(meanX_hat[np.newaxis,:], model['B'],axis=0)
        self.model = model
        return self

    def predict(self, Xnew, n_components = None):
        """
        Predict the dependent variable based on independent variable(s).

        Parameters:
        Xnew (np.array): independent variable(s) of the new data to predict
        n_components (int): number of components used to predict
                            (default is None, which uses the optimal number of
                             components obtained by cross validation)

        Returns:
        ynew_hat (np.array): predicted dependent variable based on Xnew
        """

        if n_components is None:
            B = self.model['B'][:,-1]
        else:
            B = self.model['B'][:, n_components-1]
        if Xnew.shape[1] != B.shape[0]-1:
            raise ValueError('The feature number of predictor is isconsistent with that of indepnentent.')
        Xnew = np.append(np.ones([Xnew.shape[0],1]), Xnew, axis=1)
        ynew_hat = np.dot(Xnew,B)
        return ynew_hat

    def crossValidation_predict(self, nfold = 10):
        """
        Predict dependent variable using cross validation method.

        Parameters:
        nfold (int): number of folds for cross validation (default is 10)

        Returns:
        yhat (np.array): predicted dependent variable based on cross validation method
        """
        X = self.X
        y = self.y
        yhat = np.zeros((y.shape[0],self.n_components))
        model = pls(n_components = self.n_components)
        for train, test in KFold(n_splits=nfold).split(X):
            model.fit(X[train,:], y[train])
            yhat[test,:] = model.predict(X[test,:],np.arange(self.n_components)+1)
        return yhat

    def get_optLV(self, nfold = 10):
        yhat_cv = self.crossValidation_predict(nfold)
        rmsecv = []
        r2cv = []
        for i in range(yhat_cv.shape[1]):
            reportcv = regresssionReport(self.y, yhat_cv[:,i])
            rmsecv.append(reportcv["rmse"])
            r2cv.append(reportcv["r2"])
        optLV = int(np.argmin(rmsecv)+1)
        self.optLV = optLV
        return optLV

    def transform(self, Xnew):
        meanX = np.mean(self.X, axis = 0)
        Xnew_c = Xnew - meanX
        Tnew = np.dot(Xnew_c,self.model['x_weights'])
        return Tnew

    def get_vip(self):
        # ref. https://www.sciencedirect.com/topics/engineering/variable-importance-in-projection
        x_scores, x_loadings,y_loadings,x_weights = \
            self.model['x_scores'],self.model['x_loadings'],\
            self.model['y_loadings'],self.model['x_weights']

        n_samples, n_components = x_scores.shape
        W0 = x_weights / np.sqrt(np.sum(x_weights**2, axis=0))
        p = x_loadings.shape[0]
        sumSq = np.sum(x_scores**2, axis=0) * np.sum(y_loadings**2, axis=1)
        vipScore = np.sqrt(p * np.sum(sumSq * (W0**2), axis=1) / np.sum(sumSq))
        return vipScore

    def plot_prediction(self, y, yhat, xlabel = "Reference", ylabel = "Prediction", title = "", ax = None):
        report = regresssionReport(y,yhat)
        if ax == None:
            fig, ax = plt.subplots()
        ax.plot([np.min(y)*0.95,np.max(y)*1.05],[np.min(y)*0.95,np.max(y)*1.05],
                color = 'black',label = "y=x")
        ax.scatter(y, yhat,color = 'tab:green', marker='*', label ='Prediction')
        ax.text(0.7, 0.03,
                "RMSEP = {:.4f}\nR$^2$ = {:.2}".format(report["rmse"],report["r2"]),
                transform = ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
        ax.set_title(title)


class plsda(PLSRegression):
    def __init__(self, n_components=2, scale=True, **kwargs):
        super().__init__(n_components=n_components, scale=scale, **kwargs)
        self.lda = LinearDiscriminantAnalysis()

    def fit(self, X, y):
        self.X = X
        self.y = y
        super().fit(X, y)
        self.lda.fit(self.x_scores_, y)
        return self

    def predict(self, X):
        return self.lda.predict(self.transform(X))

    def predict_log_proba(self, X):
        return self.lda.predict_log_proba(self.predict(X))

    def predict_proba(self, X):
        return self.lda.predict_proba(self.predict(X))

    def crossValidation_predict(self, nfold = 10):
        X = self.X
        y = self.y
        yhat = np.zeros((y.shape[0],self.n_components))
        for i in range(self.n_components):
            model = plsda(n_components = i+1)
            for train, test in KFold(n_splits=nfold).split(X):
                model.fit(X[train,:], y[train])
                yhat[test,i] = model.predict(X[test,:])
        return yhat

    def get_optLV(self, nfold = 10):
        yhat_cv = self.crossValidation_predict(nfold)
        accuracy_cv = []
        for i in range(yhat_cv.shape[1]):
            if len(self.lda.classes_) == 2 :
                report_cv = binaryClassificationReport(self.y, yhat_cv[:,i])
                accuracy_cv.append(report_cv["accuracy"])
            elif len(self.lda.classes_) > 2:
                report_cv = multiClassificationReport(self.y, yhat_cv[:,i])
                accuracy_tmp = [rep["accuracy"] for rep in report_cv.values()]
                accuracy_cv.append(sum(accuracy_tmp))

        optLV = int(np.argmax(accuracy_cv)+1)
        self.optLV = optLV
        return optLV

    def get_confusion_matrix(self, X, y):
        yhat = self.predict(X)
        cm = confusion_matrix(y, yhat)
        return cm

    def get_vip(self):
        # latex code: VIP = \sqrt{\frac{p\sum_{a=1}^{A}((q_a^2t_a^Tt_a)(w_{ja}/||w_a||)^2}{\sum_{a=1}^A{(q_a^2t_a^Tt_a)}}}
        XL = self.x_scores_
        yl = self.y_scores_
        Xw = self.x_weights_

        W0 = Xw / np.sqrt(np.sum(Xw**2, axis=0))
        p = XL.shape[0]
        sumSq = np.sum(Xw**2, axis=0) * np.sum(yl**2, axis=0)
        vipScore = np.sqrt(p * np.sum(sumSq * (W0**2), axis=1) / np.sum(sumSq))
        return vipScore

    def permutation_test(self, X, y, n_repeats=100,n_jobs = None):
        # Initialize arrays to store Q2 and R2 values
        q2 = np.zeros(n_repeats)
        r2 = np.zeros(n_repeats)
        permutation_ratio = np.zeros(n_repeats)
        # Perform the permutation test
        for i in range(n_repeats):
            # Shuffle the target variable
            y_shuffled = np.random.permutation(y)

            # Fit the model to the shuffled target variable
            self.fit(X, y_shuffled)

            # Calculate the cross-validated Q2 and R2 values
            y_pred = cross_val_predict(self, X, y_shuffled, cv=10, n_jobs = n_jobs)
            q2[i] = self.score(X, y_shuffled)
            r2[i] = r2_score(y_shuffled, y_pred)
            permutation_ratio[i] = np.sum(y_shuffled != y) / len(y)
        return q2, r2, permutation_ratio

class lsvc(LinearSVC):# linear svc
    def get_optParams(self, X, y, Params = None, nfold = 10, n_jobs = None):
        if Params is None:
            Params = {'C': np.logspace(-4, 5, 10),
                      'penalty': ('l1', 'l2')}
        self.gsh = GridSearchCV(estimator=self,  param_grid=Params,
                           cv = nfold, n_jobs = n_jobs)
        self.gsh.fit(X, y)
        return self.gsh.best_params_

    def get_confusion_matrix(self, X, y):
        yhat = self.predict(X)
        cm = confusion_matrix(y, yhat)
        return cm

class svc(SVC):# linear svc
    def get_optParams(self, X, y, Params = None, nfold = 10, n_jobs = None):
        if Params is None:
            Params = {'C': np.logspace(-4, 5, 10),
                      'gamma':np.logspace(-4, 5, 10),
                      'kernel': ('poly', 'rbf', 'sigmoid')}
        self.gsh = GridSearchCV(estimator=self,  param_grid=Params,
                           cv = nfold, n_jobs = n_jobs)
        self.gsh.fit(X, y)
        return self.gsh.best_params_

    def get_confusion_matrix(self, X, y):
        yhat = self.predict(X)
        cm = confusion_matrix(y, yhat)
        return cm

class rf(RandomForestClassifier):
    def get_optParams(self, X, y, Params = None, nfold = 10, n_jobs = None):
        if Params is None:
            Params = {'n_estimators': np.arange(100)+1,
                      'max_depth': np.arange(3)+1}
        self.gsh = GridSearchCV(estimator=self,  param_grid=Params,
                           cv = nfold, n_jobs = n_jobs)
        self.gsh.fit(X, y)
        return self.gsh.best_params_

    def get_confusion_matrix(self, X, y):
        yhat = self.predict(X)
        cm = confusion_matrix(y, yhat)
        return cm


class multiClass_to_binaryMatrix():
    def __init__(self):
        pass

    def fit(self, x):
        self.classes = np.unique(x)
        return self

    def transform(self, x):
        Xnew = np.zeros((len(x),len(self.classes)), dtype=int)
        if len(self.classes) > 2 :
            for i, classi in enumerate(self.classes):
                Xnew[:,i] = x==classi
        return Xnew

    def reTransform(self, xnew):
        x = [np.classes(np.where(xnew[i,:])) for i in range(xnew.shape[0])]
        return x

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def multiClassificationReport(ytrue,ypred):
    labels = np.unique(ytrue)
    report = dict()
    for labeli in labels:
        report[labeli] = binaryClassificationReport(ytrue = ytrue == labeli, ypred = ypred == labeli)
    return report

def binaryClassificationReport(ytrue,ypred):
    if len(np.unique(ytrue))>2:
        raise("Use the multiClassificationReport function for multiple classification.")
    else:
        tn, fp, fn, tp = confusion_matrix(ytrue,ypred).ravel()
        report = dict()
        report["accuracy"] = accuracy_score(ytrue, ypred)
        report["sensitivity"] = recall_score(ytrue, ypred)#recall
        report["specificity"] = tn/(tn+fp)
        report["f1"] = f1_score(ytrue, ypred)
        return report

def regresssionReport(ytrue,ypred):
    report = dict()
    report["rmse"] = mean_squared_error(ytrue, ypred, squared=False)
    report["r2"] =  r2_score(ytrue, ypred)
    return report

def simpls(X, y, n_components):
    '''
    Partial Least Squares, SIMPLS
    Ref https://github.com/freesiemens/SpectralMultivariateCalibration/blob/master/pypls.py
    :param X: independent variables, numpy array of shape (n_samples, n_variables)
    :param y: dependent variable, numpy array of shape (n_samples,) or (n_samples, 1)
    :param n_components: number of latent variables to decompose the data into
    :return: dictionary containing the results of the SIMPLS algorithm
    '''
    n_samples, n_variables = X.shape
    if np.ndim(y) == 1:
        y = y[:, np.newaxis]
    if n_samples != y.shape[0]:
        raise ValueError('The number of independent and dependent variable are inconsistent')

    n_components = np.min((n_components, n_samples, n_variables))
    V = np.zeros((n_variables, n_components))
    x_scores = np.zeros((n_samples, n_components))  # X scores (standardized)
    x_weights = np.zeros((n_variables, n_components))  # X weights
    x_loadings = np.zeros((n_variables, n_components))  # X loadings
    y_loadings = np.zeros((1, n_components))  # Y loadings
    y_scores = np.zeros((n_samples, n_components))  # Y scores
    s = np.dot(X.T, y).ravel()  # cross-product matrix between the X and y_data
    for i in range(n_components):
        r = s
        t = np.dot(X, r)
        tt = np.linalg.norm(t)
        t = t / tt
        r = r / tt
        p = np.dot(X.T, t)
        q = np.dot(y.T, t)
        u = np.dot(y, q)
        v = p  # P的正交基
        if i > 0:
            v = v - np.dot(V, np.dot(V.T, p))  # Gram-Schimidt orthogonal
            u = u - np.dot(x_scores, np.dot(x_scores.T, u))
        v = v / np.linalg.norm(v)
        s = s - np.dot(v, np.dot(v.T, s))
        x_weights[:, i] = r
        x_scores[:, i] = t
        x_loadings[:, i] = p
        y_loadings[:, i] = q
        y_scores[:, i] = u
        V[:, i] = v
    B = np.cumsum(np.dot(x_weights, np.diag(y_loadings.ravel())), axis=1)
    return {'B': B, 'x_scores': x_scores, 'x_loadings': x_loadings, 'y_loadings': y_loadings, \
            'x_scores_weights': x_weights, 'x_weights': x_weights, 'y_scores':y_scores}



def sampleSplit_random(X,test_size=0.25, random_state=1, shuffle=False):
    sampleIdx = np.arange(X.shape[0])
    trainIdx, testIdx = train_test_split(sampleIdx,test_size=test_size,
                                         random_state=random_state,
                                         shuffle=shuffle)
    return trainIdx, testIdx

def sampleSplit_KS(X, test_size=0.25, metric='euclidean', *args, **kwargs):
    """Kennard Stone Sample Split method
    Parameters
    ----------
    spectra: ndarray, shape of i x j
        i spectrums and j variables (wavelength/wavenumber/ramam shift and so on)
    test_size : float, int
        if float, then round(i x (1-test_size)) spectrums are selected as test data, by default 0.25
        if int, then test_size is directly used as test data size
    metric : str, optional
        The distance metric to use, by default 'euclidean'
        See scipy.spatial.distance.cdist for more infomation
    Returns
    -------
    select_pts: list
        index of selected spetrums as train data, index is zero based
    remaining_pts: list
        index of remaining spectrums as test data, index is zero based
    References
    --------
    Kennard, R. W., & Stone, L. A. (1969). Computer aided design of experiments.
    Technometrics, 11(1), 137-148. (https://www.jstor.org/stable/1266770)
    """
    Xscore = PCA(n_components=2).fit_transform(X)
    distance = cdist(Xscore, Xscore, metric=metric, *args, **kwargs)
    select_pts = []
    remaining_pts = [x for x in range(distance.shape[0])]

    # first select 2 farthest points
    first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
    select_pts.append(first_2pts[0])
    select_pts.append(first_2pts[1])

    # remove the first 2 points from the remaining list
    remaining_pts.remove(first_2pts[0])
    remaining_pts.remove(first_2pts[1])

    for i in range(round(X.shape[0]*(1-test_size)) - 2):
        # find the maximum minimum distance
        select_distance = distance[select_pts, :]
        min_distance = select_distance[:, remaining_pts]
        min_distance = np.min(min_distance, axis=0)
        max_min_distance = np.max(min_distance)

        # select the first point (in case that several distances are the same, choose the first one)
        points = np.argwhere(select_distance == max_min_distance)[:, 1].tolist()
        for point in points:
            if point in select_pts:
                pass
            else:
                select_pts.append(point)
                remaining_pts.remove(point)
                break
    return select_pts, remaining_pts
