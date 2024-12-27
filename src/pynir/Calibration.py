# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:00:35 2022

@author: J Zhang (jzhang@chemoinfolab.com)
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
    """
    Partial Least Squares (PLS) regression model.

    Parameters
    ----------
    n_components : int, optional (default=2)
        The number of PLS components to use.

    Attributes
    ----------
    model : dict
        The PLS model, containing the following keys:
        - 'x_scores': the X scores
        - 'x_loadings': the X loadings
        - 'y_loadings': the Y loadings
        - 'x_weights': the X weights
        - 'B': the regression coefficients

    optLV : int
        The optimal number of PLS components, determined by cross-validation.

    Methods
    -------
    fit(X, y)
        Fit the PLS model to the training data.

    predict(Xnew, n_components=None)
        Predict the response variable for new data.

    crossValidation_predict(nfold=10)
        Perform cross-validation and return the predicted response variable.

    get_optLV(nfold=10)
        Determine the optimal number of PLS components using cross-validation.

    transform(Xnew)
        Transform new data into the PLS space.

    get_vip()
        Compute the variable importance in projection (VIP) scores.

    plot_prediction(y, yhat, xlabel="Reference", ylabel="Prediction", title="", ax=None)
        Plot the predicted response variable against the reference variable.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        """
        Fit the PLS model to the training data.

        Parameters
        ----------
        X : numpy.ndarray
            The independent variable matrix.
        y : numpy.ndarray
            The dependent variable vector.

        Returns
        -------
        self : pls
            The fitted PLS model.
        """
        self.X = X
        self.y = y
        meanX = np.mean(X, axis=0)
        meany = np.mean(y)
        Xcentered = X - meanX
        ycentered = y - meany
        model = simpls(Xcentered, ycentered, self.n_components)
        meanX_hat = -1 * np.dot(meanX, model['B']) + meany
        model['B'] = np.append(meanX_hat[np.newaxis, :], model['B'], axis=0)
        self.model = model
        return self

    def predict(self, Xnew, n_components=None):
        """
        Predict the response variable for new data.

        Parameters
        ----------
        Xnew : numpy.ndarray
            The new independent variable matrix.
        n_components : int, optional
            The number of PLS components to use (default is None, which uses all components).

        Returns
        -------
        ynew_hat : numpy.ndarray
            The predicted response variable.
        """
        if n_components is None:
            B = self.model['B'][:, -1]
        else:
            B = self.model['B'][:, n_components-1]
        if Xnew.shape[1] != B.shape[0]-1:
            raise ValueError(
                'The feature number of predictor is isconsistent with that of indepnentent.')
        Xnew = np.append(np.ones([Xnew.shape[0], 1]), Xnew, axis=1)
        ynew_hat = np.dot(Xnew, B)
        return ynew_hat

    def crossValidation_predict(self, nfold=10):
        """
        Perform cross-validation and return the predicted response variable.

        Parameters
        ----------
        nfold : int, optional (default=10)
            The number of folds to use in cross-validation.

        Returns
        -------
        yhat : numpy.ndarray
            The predicted response variable.
        """
        X = self.X
        y = self.y
        yhat = np.zeros((y.shape[0], self.n_components))
        model = pls(n_components=self.n_components)
        for train, test in KFold(n_splits=nfold).split(X):
            model.fit(X[train, :], y[train])
            yhat[test, :] = model.predict(
                X[test, :], np.arange(self.n_components)+1)
        return yhat

    def get_optLV(self, nfold=10):
        """
        Determine the optimal number of PLS components using cross-validation.

        Parameters
        ----------
        nfold : int, optional (default=10)
            The number of folds to use in cross-validation.

        Returns
        -------
        optLV : int
            The optimal number of PLS components.
        """
        yhat_cv = self.crossValidation_predict(nfold)
        rmsecv = []
        r2cv = []
        for i in range(yhat_cv.shape[1]):
            reportcv = regressionReport(self.y, yhat_cv[:, i])
            rmsecv.append(reportcv["rmse"])
            r2cv.append(reportcv["r2"])
        optLV = int(np.argmin(rmsecv)+1)
        self.optLV = optLV
        return optLV

    def transform(self, Xnew):
        """
        Transform new data into the PLS space.

        Parameters
        ----------
        Xnew : numpy.ndarray
            The new independent variable matrix.

        Returns
        -------
        Tnew : numpy.ndarray
            The transformed data.
        """
        meanX = np.mean(self.X, axis=0)
        Xnew_c = Xnew - meanX
        Tnew = np.dot(Xnew_c, self.model['x_weights'])
        return Tnew

    def get_vip(self):
        """
        Compute the variable importance in projection (VIP) scores.

        Returns
        -------
        vipScore : numpy.ndarray
            The VIP scores.
        
        References
        ----------
        https://www.sciencedirect.com/topics/engineering/variable-importance-in-projection
        """
        x_scores, x_loadings, y_loadings, x_weights = \
            self.model['x_scores'], self.model['x_loadings'],\
            self.model['y_loadings'], self.model['x_weights']

        n_samples, n_components = x_scores.shape
        W0 = x_weights / np.sqrt(np.sum(x_weights**2, axis=0))
        p = x_loadings.shape[0]
        sumSq = np.sum(x_scores**2, axis=0) * np.sum(y_loadings**2, axis=1)
        vipScore = np.sqrt(p * np.sum(sumSq * (W0**2), axis=1) / np.sum(sumSq))
        return vipScore

    def plot_prediction(self, y, yhat, xlabel="Reference", ylabel="Prediction", title="", ax=None):
        """
        Plot the predicted response variable against the reference variable.

        Parameters
        ----------
        y : numpy.ndarray
            The reference variable.
        yhat : numpy.ndarray
            The predicted response variable.
        xlabel : str, optional (default="Reference")
            The label for the x-axis.
        ylabel : str, optional (default="Prediction")
            The label for the y-axis.
        title : str, optional (default="")
            The title for the plot.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the figure (default is None).

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object containing the plotted figure.
        """
        report = regressionReport(y, yhat)
        if ax == None:
            fig, ax = plt.subplots()
        ax.plot([np.min(y)*0.95, np.max(y)*1.05], [np.min(y)*0.95, np.max(y)*1.05],
                color='black', label="y=x")
        ax.scatter(y, yhat, color='tab:green', marker='*', label='Prediction')
        ax.text(0.7, 0.03,
                "RMSEP = {:.4f}\nR$^2$ = {:.2}".format(
                    report["rmse"], report["r2"]),
                transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
        ax.set_title(title)


class plsda(PLSRegression):
    """
    Partial Least Squares Discriminant Analysis (PLS-DA) model.

    This class extends the scikit-learn PLSRegression class to include
    Linear Discriminant Analysis (LDA) for classification.

    Parameters
    ----------
    n_components : int, optional (default = 2)
        Number of components to keep in the model.
    scale : bool, optional (default = True)
        Whether to scale the data before fitting the model.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the PLSRegression constructor.

    Attributes
    ----------
    lda : LinearDiscriminantAnalysis
        The LDA model used for classification.

    Methods
    -------
    fit(X, y)
        Fit the PLS-DA model to the training data.
    predict(X)
        Predict the class labels for new data.
    predict_log_proba(X)
        Predict the log probabilities of the class labels for new data.
    predict_proba(X)
        Predict the probabilities of the class labels for new data.
    crossValidation_predict(nfold=10)
        Perform cross-validation to predict the class labels for the training data.
    get_optLV(nfold=10)
        Find the optimal number of components using cross-validation.
    get_confusion_matrix(X, y)
        Compute the confusion matrix for the model.
    get_vip()
        Compute the Variable Importance in Projection (VIP) scores for the model.
    permutation_test(X, y, n_repeats=100, n_jobs=None)
        Perform a permutation test to assess the significance of the model.

    """
    def __init__(self, n_components=2, scale=True, **kwargs):
        super().__init__(n_components=n_components, scale=scale, **kwargs)
        self.lda = LinearDiscriminantAnalysis()

    def fit(self, X, y):
        """
        Fit the PLS-DA model to the training data.

        Parameters
        ----------
        X : numpy.ndarray
            The training data matrix.
        y : numpy.ndarray
            The target variable vector.

        Returns
        -------
        self : plsda
            The fitted PLS-DA model.
        """
        self.X = X
        self.y = y
        super().fit(X, y)
        self.lda.fit(self.x_scores_, y)
        return self

    def predict(self, X):
        """
        Predict the class labels for new data.

        Parameters
        ----------
        X : numpy.ndarray
            The new data matrix.

        Returns
        -------
        y_pred : numpy.ndarray
            The predicted class labels.
        """
        return self.lda.predict(self.transform(X))

    def predict_log_proba(self, X):
        """
        Predict the log probabilities of the class labels for new data.

        Parameters
        ----------
        X : numpy.ndarray
            The new data matrix.

        Returns
        -------
        log_proba : numpy.ndarray
            The log probabilities of the class labels.
        """
        return self.lda.predict_log_proba(self.predict(X))

    def predict_proba(self, X):
        """
        Predict the probabilities of the class labels for new data.

        Parameters
        ----------
        X : numpy.ndarray
            The new data matrix.

        Returns
        -------
        proba : numpy.ndarray
            The probabilities of the class labels.
        """
        return self.lda.predict_proba(self.predict(X))

    def crossValidation_predict(self, nfold=10):
        """
        Perform cross-validation to predict the class labels for the training data.

        Parameters
        ----------
        nfold : int, optional (default = 10)
            The number of folds to use in cross-validation.

        Returns
        -------
        yhat : numpy.ndarray
            The predicted class labels for each fold and each number of components.
        """
        X = self.X
        y = self.y
        yhat = np.zeros((y.shape[0], self.n_components))
        for i in range(self.n_components):
            model = plsda(n_components=i+1)
            for train, test in KFold(n_splits=nfold).split(X):
                model.fit(X[train, :], y[train])
                yhat[test, i] = model.predict(X[test, :])
        return yhat

    def get_optLV(self, nfold=10):
        """
        Find the optimal number of components using cross-validation.

        Parameters
        ----------
        nfold : int, optional (default = 10)
            The number of folds to use in cross-validation.

        Returns
        -------
        optLV : int
            The optimal number of components.
        """
        yhat_cv = self.crossValidation_predict(nfold)
        accuracy_cv = []
        for i in range(yhat_cv.shape[1]):
            if len(self.lda.classes_) == 2:
                report_cv = binaryClassificationReport(self.y, yhat_cv[:, i])
                accuracy_cv.append(report_cv["accuracy"])
            elif len(self.lda.classes_) > 2:
                report_cv = multiClassificationReport(self.y, yhat_cv[:, i])
                accuracy_tmp = [rep["accuracy"] for rep in report_cv.values()]
                accuracy_cv.append(sum(accuracy_tmp))

        optLV = int(np.argmax(accuracy_cv)+1)
        self.optLV = optLV
        return optLV

    def get_confusion_matrix(self, X, y):
        """
        Compute the confusion matrix for the model.

        Parameters
        ----------
        X : numpy.ndarray
            The data matrix.
        y : numpy.ndarray
            The target variable vector.

        Returns
        -------
        cm : numpy.ndarray
            The confusion matrix.
        """
        yhat = self.predict(X)
        cm = confusion_matrix(y, yhat)
        return cm

    def get_vip(self):
        """
        Compute the Variable Importance in Projection (VIP) scores for the model.

        Returns
        -------
        vipScore : numpy.ndarray
            The VIP scores.
        """
        # latex code: VIP = \sqrt{\frac{p\sum_{a=1}^{A}((q_a^2t_a^Tt_a)(w_{ja}/||w_a||)^2}{\sum_{a=1}^A{(q_a^2t_a^Tt_a)}}}
        XL = self.x_scores_
        yl = self.y_scores_
        Xw = self.x_weights_

        W0 = Xw / np.sqrt(np.sum(Xw**2, axis=0))
        p = XL.shape[0]
        sumSq = np.sum(Xw**2, axis=0) * np.sum(yl**2, axis=0)
        vipScore = np.sqrt(p * np.sum(sumSq * (W0**2), axis=1) / np.sum(sumSq))
        return vipScore

    def permutation_test(self, X, y, n_repeats=100, n_jobs=None):
        """
        Perform a permutation test to assess the significance of the model.

        Parameters
        ----------
        X : numpy.ndarray
            The data matrix.
        y : numpy.ndarray
            The target variable vector.
        n_repeats : int, optional (default = 100)
            The number of permutations to perform.
        n_jobs : int, optional (default = None)
            The number of parallel jobs to run. If None, all CPUs are used.

        Returns
        -------
        q2 : numpy.ndarray
            The Q2 values for each permutation.
        r2 : numpy.ndarray
            The R2 values for each permutation.
        permutation_ratio : numpy.ndarray
            The ratio of permuted target variable values to total target variable values for each permutation.
        """
        q2 = np.zeros(n_repeats)
        r2 = np.zeros(n_repeats)
        permutation_ratio = np.zeros(n_repeats)
        for i in range(n_repeats):
            y_shuffled = np.random.permutation(y)
            self.fit(X, y_shuffled)
            y_pred = cross_val_predict(
                self, X, y_shuffled, cv=10, n_jobs=n_jobs)
            q2[i] = self.score(X, y_shuffled)
            r2[i] = r2_score(y_shuffled, y_pred)
            permutation_ratio[i] = np.sum(y_shuffled != y) / len(y)
        return q2, r2, permutation_ratio


class lsvc(LinearSVC):  # linear svc
    """
    Linear Support Vector Classification (Linear SVC) model.

    This class extends the scikit-learn LinearSVC class to include
    methods for finding the optimal hyperparameters and computing
    the confusion matrix.

    Methods
    -------
    get_optParams(X, y, Params=None, nfold=10, n_jobs=None)
        Find the optimal hyperparameters for the model using cross-validation.
    get_confusion_matrix(X, y)
        Compute the confusion matrix for the model.

    """
    def get_optParams(self, X, y, Params=None, nfold=10, n_jobs=None):
        """
        Find the optimal hyperparameters for the model using cross-validation.

        Parameters
        ----------
        X : numpy.ndarray
            The data matrix.
        y : numpy.ndarray
            The target variable vector.
        Params : dict, optional (default = None)
            The hyperparameters to search over. If None, a default set of hyperparameters is used.
        nfold : int, optional (default = 10)
            The number of folds to use in cross-validation.
        n_jobs : int, optional (default = None)
            The number of parallel jobs to run. If None, all CPUs are used.

        Returns
        -------
        best_params : dict
            The optimal hyperparameters for the model.
        """
        if Params is None:
            Params = {'C': np.logspace(-4, 5, 10),
                      'penalty': ('l1', 'l2')}
        self.gsh = GridSearchCV(estimator=self,  param_grid=Params,
                                cv=nfold, n_jobs=n_jobs)
        self.gsh.fit(X, y)
        return self.gsh.best_params_

    def get_confusion_matrix(self, X, y):
        """
        Compute the confusion matrix for the model.

        Parameters
        ----------
        X : numpy.ndarray
            The data matrix.
        y : numpy.ndarray
            The target variable vector.

        Returns
        -------
        cm : numpy.ndarray
            The confusion matrix.
        """
        yhat = self.predict(X)
        cm = confusion_matrix(y, yhat)
        return cm


class svc(SVC):
    """
    Support Vector Classification (SVC) model.

    This class extends the scikit-learn SVC class to include methods for
    finding the optimal hyperparameters and computing the confusion matrix.

    Methods
    -------
    get_optParams(X, y, Params=None, nfold=10, n_jobs=None)
        Find the optimal hyperparameters for the model using cross-validation.
    get_confusion_matrix(X, y)
        Compute the confusion matrix for the model.

    """
    def get_optParams(self, X, y, Params=None, nfold=10, n_jobs=None):
        """
        Find the optimal hyperparameters for the model using cross-validation.

        Parameters
        ----------
        X : numpy.ndarray
            The data matrix.
        y : numpy.ndarray
            The target variable vector.
        Params : dict, optional (default = None)
            The hyperparameters to search over. If None, a default set of hyperparameters is used.
        nfold : int, optional (default = 10)
            The number of folds to use in cross-validation.
        n_jobs : int, optional (default = None)
            The number of parallel jobs to run. If None, all CPUs are used.

        Returns
        -------
        best_params : dict
            The optimal hyperparameters for the model.
        """
        if Params is None:
            Params = {'C': np.logspace(-4, 5, 10),
                      'gamma': np.logspace(-4, 5, 10),
                      'kernel': ('poly', 'rbf', 'sigmoid')}
        self.gsh = GridSearchCV(estimator=self,  param_grid=Params,
                                cv=nfold, n_jobs=n_jobs)
        self.gsh.fit(X, y)
        return self.gsh.best_params_

    def get_confusion_matrix(self, X, y):
        """
        Compute the confusion matrix for the model.

        Parameters
        ----------
        X : numpy.ndarray
            The data matrix.
        y : numpy.ndarray
            The target variable vector.

        Returns
        -------
        cm : numpy.ndarray
            The confusion matrix.
        """
        yhat = self.predict(X)
        cm = confusion_matrix(y, yhat)
        return cm


class rf(RandomForestClassifier):
    """
    Random Forest Classification (RF) model.

    This class extends the scikit-learn RandomForestClassifier class to include
    methods for finding the optimal hyperparameters and computing the confusion matrix.

    Methods
    -------
    get_optParams(X, y, Params=None, nfold=10, n_jobs=None)
        Find the optimal hyperparameters for the model using cross-validation.
    get_confusion_matrix(X, y)
        Compute the confusion matrix for the model.

    """
    def get_optParams(self, X, y, Params=None, nfold=10, n_jobs=None):
        """
        Find the optimal hyperparameters for the model using cross-validation.

        Parameters
        ----------
        X : numpy.ndarray
            The data matrix.
        y : numpy.ndarray
            The target variable vector.
        Params : dict, optional (default = None)
            The hyperparameters to search over. If None, a default set of hyperparameters is used.
        nfold : int, optional (default = 10)
            The number of folds to use in cross-validation.
        n_jobs : int, optional (default = None)
            The number of parallel jobs to run. If None, all CPUs are used.

        Returns
        -------
        best_params : dict
            The optimal hyperparameters for the model.
        """
        if Params is None:
            Params = {'n_estimators': np.arange(100)+1,
                      'max_depth': np.arange(3)+1}
        self.gsh = GridSearchCV(estimator=self,  param_grid=Params,
                                cv=nfold, n_jobs=n_jobs)
        self.gsh.fit(X, y)
        return self.gsh.best_params_

    def get_confusion_matrix(self, X, y):
        """
        Compute the confusion matrix for the model.

        Parameters
        ----------
        X : numpy.ndarray
            The data matrix.
        y : numpy.ndarray
            The target variable vector.

        Returns
        -------
        cm : numpy.ndarray
            The confusion matrix.
        """
        yhat = self.predict(X)
        cm = confusion_matrix(y, yhat)
        return cm


class multiClass_to_binaryMatrix():
    """
    Multi-class to binary matrix conversion.

    This class is used to convert a multi-class target variable into a binary matrix
    suitable for training a multi-label classifier.

    Methods
    -------
    fit(x)
        Fit the transformer to the data.
    transform(x)
        Transform the data into a binary matrix.
    reTransform(xnew)
        Convert the binary matrix back into the original target variable.

    """
    def __init__(self):
        pass

    def fit(self, x):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        x : numpy.ndarray
            The target variable vector.

        Returns
        -------
        self : object
            Returns self.
        """
        self.classes = np.unique(x)
        return self

    def transform(self, x):
        """
        Transform the data into a binary matrix.

        Parameters
        ----------
        x : numpy.ndarray
            The target variable vector.

        Returns
        -------
        Xnew : numpy.ndarray
            The binary matrix.
        """
        Xnew = np.zeros((len(x), len(self.classes)), dtype=int)
        if len(self.classes) > 2:
            for i, classi in enumerate(self.classes):
                Xnew[:, i] = x == classi
        return Xnew

    def reTransform(self, xnew):
        """
        Convert the binary matrix back into the original target variable.

        Parameters
        ----------
        xnew : numpy.ndarray
            The binary matrix.

        Returns
        -------
        x : numpy.ndarray
            The original target variable vector.
        """
        x = [np.classes(np.where(xnew[i, :])) for i in range(xnew.shape[0])]
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
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.show()


def multiClassificationReport(ytrue, ypred):
    """
    Generate a classification report for a multi-class classification problem.

    This function generates a classification report for a multi-class classification problem
    by computing binary classification reports for each class.

    Parameters
    ----------
    ytrue : numpy.ndarray
        The true target variable vector.
    ypred : numpy.ndarray
        The predicted target variable vector.

    Returns
    -------
    report : dict
        A dictionary containing binary classification reports for each class.
    """
    labels = np.unique(ytrue)
    report = dict()
    for labeli in labels:
        report[labeli] = binaryClassificationReport(
            ytrue=ytrue == labeli, ypred=ypred == labeli)
    return report


def binaryClassificationReport(ytrue, ypred):
    """
    Generate a binary classification report.

    This function generates a binary classification report for a binary classification problem
    by computing the confusion matrix and various performance metrics.

    Parameters
    ----------
    ytrue : numpy.ndarray
        The true target variable vector.
    ypred : numpy.ndarray
        The predicted target variable vector.

    Returns
    -------
    report : dict
        A dictionary containing various performance metrics.
    """
    if len(np.unique(ytrue)) > 2:
        raise ("Use the multiClassificationReport function for multiple classification.")
    else:
        tn, fp, fn, tp = confusion_matrix(ytrue, ypred).ravel()
        report = dict()
        report["accuracy"] = accuracy_score(ytrue, ypred)
        report["sensitivity"] = recall_score(ytrue, ypred)  # recall
        report["specificity"] = tn/(tn+fp)
        report["f1"] = f1_score(ytrue, ypred)
        return report


def regressionReport(ytrue, ypred):
    """
    Generate a regression report.

    This function generates a regression report for a regression problem
    by computing the root mean squared error (RMSE) and the R-squared (R2) score.

    Parameters
    ----------
    ytrue : numpy.ndarray
        The true target variable vector.
    ypred : numpy.ndarray
        The predicted target variable vector.

    Returns
    -------
    report : dict
        A dictionary containing the RMSE and R2 score.
    """
    report = dict()
    report["rmse"] = mean_squared_error(ytrue, ypred, squared=False)
    report["r2"] = r2_score(ytrue, ypred)
    return report


def simpls(X, y, n_components):
    """
    Perform SIMPLS (Partial Least Squares) regression.

    This function performs SIMPLS regression, which is a variant of PLS regression
    that uses a sequential algorithm to compute the PLS components.

    Parameters
    ----------
    X : numpy.ndarray
        The independent variable matrix.
    y : numpy.ndarray
        The dependent variable vector.
    n_components : int
        The number of PLS components to compute.

    Returns
    -------
    results : dict
        A dictionary containing the PLS components and loadings.

    Notes
    -----
    This implementation is based on the algorithm described in:
    Wold, S., Ruhe, A., Wold, H., & Dunn III, W. J. (1984).
    The collinearity problem in linear regression. The partial least squares (PLS) approach to generalized inverses.
    SIAM Journal on Scientific and Statistical Computing, 5(3), 735-743.
    """
    n_samples, n_variables = X.shape
    if np.ndim(y) == 1:
        y = y[:, np.newaxis]
    if n_samples != y.shape[0]:
        raise ValueError(
            'The number of independent and dependent variable are inconsistent')

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
        v = p  
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
    return {'B': B, 'x_scores': x_scores, 'x_loadings': x_loadings, 'y_loadings': y_loadings,
            'x_scores_weights': x_weights, 'x_weights': x_weights, 'y_scores': y_scores}


def sampleSplit_random(X, test_size=0.25, random_state=1, shuffle=False):
    """
    Randomly split a dataset into training and testing sets.

    This function randomly splits a dataset into training and testing sets
    using the train_test_split function from scikit-learn.

    Parameters
    ----------
    X : numpy.ndarray
        The dataset to split.
    test_size : float, optional
        The proportion of the dataset to include in the test split.
    random_state : int, optional
        The random seed to use for reproducibility.
    shuffle : bool, optional
        Whether or not to shuffle the dataset before splitting.

    Returns
    -------
    trainIdx : numpy.ndarray
        The indices of the training set.
    testIdx : numpy.ndarray
        The indices of the testing set.
    """
    sampleIdx = np.arange(X.shape[0])
    trainIdx, testIdx = train_test_split(sampleIdx, test_size=test_size,
                                         random_state=random_state,
                                         shuffle=shuffle)
    return trainIdx, testIdx


def sampleSplit_KS(X, test_size=0.25, metric='euclidean', *args, **kwargs):
    """
    Split a dataset into training and testing sets using the KS algorithm.

    This function splits a dataset into training and testing sets using the KS algorithm,
    which selects points that maximize the minimum distance between them and previously
    selected points.

    Parameters
    ----------
    X : numpy.ndarray
        The dataset to split.
    test_size : float, optional
        The proportion of the dataset to include in the test split.
    metric : str, optional
        The distance metric to use for computing distances between points.
    *args : tuple
        Additional arguments to pass to the distance metric function.
    **kwargs : dict
        Additional keyword arguments to pass to the distance metric function.

    Returns
    -------
    trainIdx : numpy.ndarray
        The indices of the training set.
    testIdx : numpy.ndarray
        The indices of the testing set.

    Notes
    -----
    This implementation is based on the algorithm described in:
    K. S. Lee, "Automatic thresholding for defect detection," Pattern Recognition, vol. 21, no. 3, pp. 225-238, 1988.
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
        points = np.argwhere(select_distance == max_min_distance)[
            :, 1].tolist()
        for point in points:
            if point in select_pts:
                pass
            else:
                select_pts.append(point)
                remaining_pts.remove(point)
                break
    return select_pts, remaining_pts
