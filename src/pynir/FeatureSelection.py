import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from numpy.linalg import matrix_rank as rank

import matplotlib.pyplot as plt
from .Preprocessing import smooth


# Single step feature selection method
class MCUVE:
    """
    Multi-Component Uninformative Variable Elimination (MCUVE) for feature selection.

    This class implements the MCUVE algorithm for feature selection, which is a method
    for identifying the most informative features in a dataset.

    Parameters
    ----------
    x : numpy.ndarray
        The independent variables.
    y : numpy.ndarray
        The dependent variable.
    ncomp : int, optional
        The number of latent components.
    nrep : int, optional
        The number of repetitions.
    testSize : float, optional
        The proportion of the dataset to include in the test split.

    Attributes
    ----------
    criteria : numpy.ndarray
        The criteria for feature selection.
    featureRank : numpy.ndarray
        The rank of the features based on their criteria.
    featureR2 : list
        The R^2 score for each number of selected features.
    """
    def __init__(self, x, y, ncomp=2, nrep=500, testSize=0.2):
        self.x = x
        self.y = y
        # The number of latent components should not be larger than any dimension size of independent matrix
        self.ncomp = min([ncomp, rank(x)])
        self.nrep = nrep
        self.testSize = testSize


    def fit(self):
        """
        Fit the MCUVE model to the training data.

        Returns
        -------
        self : MCUVE
            The fitted MCUVE model.
        """
        PLSCoef = np.zeros((self.nrep, self.x.shape[1]))
        ss = ShuffleSplit(n_splits=self.nrep, test_size=self.testSize)
        step = 0
        for train, test in ss.split(self.x, self.y):
            xtrain = self.x[train, :]
            ytrain = self.y[train]
            plsModel = PLSRegression(min([self.ncomp, rank(xtrain)]))
            plsModel.fit(xtrain, ytrain)
            PLSCoef[step, :] = plsModel.coef_.T
            step += 1
        meanCoef = np.mean(PLSCoef, axis=0)
        stdCoef = np.std(PLSCoef, axis=0)
        self.criteria = meanCoef / stdCoef
        self.featureRank = np.argsort(-np.abs(self.criteria))
        return self
    
    def evalFeatures(self, cv=5):
        self.featureR2 = []
        for i in range(self.x.shape[1]):
            xi = self.x[:, self.featureRank[:i + 1]]
            if i<self.ncomp:
                regModel = LinearRegression()
            else:
                regModel = PLSRegression(min([self.ncomp, rank(xi)]))

            cvScore = cross_val_score(regModel, xi, self.y, cv=cv)
            self.featureR2[i] = np.mean(cvScore)
        return self

    def Transform(self, X, nSelFeatures = None):
        if nSelFeatures is None:
            self.evalFeatures()
            nSelFeatures = np.argmax(self.featureR2)+1
            
        selFeatures = self.featureRank[:nSelFeatures]
        return X[:, selFeatures]
                
class RT(MCUVE):
    """
    Random Test (RT) feature selection method based on cross-validation.

    This class implements the RT algorithm for feature selection, which is a method for selecting the most
    informative features from a dataset based on their ability to predict the target variable using cross-validation.

    Parameters
    ----------
    ncomp : int, optional
        The number of components to use in PLS regression.
    nrep : int, optional
        The number of repetitions to use in the random test.

    Attributes
    ----------
    criteria : numpy.ndarray
        The criteria for feature selection.
    featureRank : numpy.ndarray
        The ranked indices of the selected features.
    featureR2 : list of float
        The R^2 scores for each number of selected features.

    """
    def fit(self):
        # calculate normal pls regression coefficient
        plsmodel0=PLSRegression(self.ncomp)
        plsmodel0.fit(self.x, self.y)
        # calculate noise reference regression coefficient
        plsCoef0=plsmodel0.coef_
        PLSCoef = np.zeros((self.nrep, self.x.shape[1]))
        for i in range(self.nrep):
            randomidx = list(range(self.x.shape[0]))
            np.random.shuffle(randomidx)
            ytrain = self.y[randomidx]
            plsModel = PLSRegression(self.ncomp)
            plsModel.fit(self.x, ytrain)
            PLSCoef[i, :] = plsModel.coef_.T
        plsCoef0 = np.tile(np.reshape(plsCoef0, [1, -1]), [ self.nrep, 1])
        criteria = np.sum(np.abs(PLSCoef) > np.abs(plsCoef0), axis=0)/self.nrep
        self.criteria = criteria
        self.featureRank = np.argsort(self.criteria)
        return self

    def evalFeatures(self, cv=5):
        self.featureR2 = []
        # Note: small P value indicating important feature
        
        for i in range(self.x.shape[1]):
            xi = self.x[:, self.featureRank[:i + 1]]
            if i<self.ncomp:
                regModel = LinearRegression()
            else:
                regModel = PLSRegression(min([self.ncomp, rank(xi)]))
            cvScore = cross_val_score(regModel, xi, self.y, cv=cv)
            self.featureR2[i] = np.mean(cvScore)
        return self



class VC(RT):
    """
    Variable Importance Criterion (VC) feature selection based on PLS regression.

    This class implements the VC algorithm for feature selection based on the criterion of C values,
    which is a method for selecting the most important variables in near-infrared spectral analysis.

    Parameters
    ----------
    cv : int, optional
        The number of cross-validation folds to use.
    isSmooth : bool, optional
        Whether to apply smoothing to the criteria.

    Attributes
    ----------
    criteria : numpy.ndarray
        The VC criteria for each variable.
    featureRank : numpy.ndarray
        The indices of the variables sorted by their VC criteria.

    Notes
    -----
    This implementation is based on the algorithm described in:
    [1] Zhang J., et.al. A variable importance criterion for variable selection in near-infrared spectral analysis [J]. Science China-Chemistry, 2019, 62(2): 271-279.
    """
    def fit(self, cv=5, isSmooth = True):
        """
        Fit the VC model to the training data.

        Parameters
        ----------
        cv : int, optional
            The number of cross-validation folds to use.
        isSmooth : bool, optional
            Whether to apply smoothing to the criteria.

        Returns
        -------
        self : VC
            The fitted VC model.
        """
        # calculate normal pls regression coefficient
        nVar = self.x.shape[1]
        sampleMatrix = np.ndarray([self.nrep,self.x.shape[1]], dtype=int)
        sampleMatrix[:, :] = 0
        errVector = np.ndarray([self.nrep,1])
        # The number of variable in combination should less than the total variable number
        if nVar > self.ncomp:
            nSample = max([self.ncomp, nVar//10])
        else:
            nSample = max([1, nVar-1])
        sampleidx = range(self.x.shape[1])
        for i in range(self.nrep):
            sampleidx = shuffle(sampleidx)
            seli = sampleidx[:nSample]
            plsModel = PLSRegression(n_components=min([self.ncomp, rank(self.x[:, seli])]))
            plsModel.fit(self.x[:, seli], self.y)
            sampleMatrix[i, seli] = 1
            yhati=cross_val_predict(plsModel, self.x[:, seli], self.y, cv=cv)
            errVector[i] = np.sqrt(mean_squared_error(yhati, self.y))
        plsModel = PLSRegression(n_components=self.ncomp)
        plsModel.fit(sampleMatrix, errVector)
        self.criteria = plsModel.coef_.ravel()
        if self.nrep < 5*self.x.shape[0] and isSmooth:
           self.criteria =  smooth(polyorder=1).transform(self.criteria)
        self.featureRank = np.argsort(self.criteria)
        return self

class MSVC:
    """
    Multi-Step Variable Importance Criterion (MSVC) for recursive feature selection.

    This class implements the MSVC algorithm for recursive feature selection, which is a method
    for selecting the most important features in a dataset based on their contribution to the
    prediction of a target variable.

    Parameters
    ----------
    x : numpy.ndarray
        The independent variable matrix.
    y : numpy.ndarray
        The dependent variable vector.
    ncomp : int, optional
        The number of latent components to use in the PLS regression model.
    nrep : int, optional
        The number of repetitions to use in the VC algorithm.
    ncut : int, optional
        The number of feature selection stages to perform.
    testSize : float, optional
        The proportion of the dataset to use for testing in cross-validation.

    Attributes
    ----------
    criteria : numpy.ndarray
        The VC criteria for each feature at each stage of feature selection.
    featureR2 : numpy.ndarray
        The R-squared score for each stage of feature selection.
    selFeature : numpy.ndarray
        The boolean mask indicating the selected features.

    Notes
    -----
    This implementation is based on the algorithm described in:
    [1] Zhang J., et.al. A variable importance criterion for variable selection in near-infrared spectral analysis [J]. Science China-Chemistry, 2019, 62(2): 271-279.
    """
    def __init__(self, x, y, ncomp=1, nrep=7000, ncut=50, testSize=0.2):
        self.x = x
        self.y = y
        # The number of latent components should not be larger than any dimension size of independent matrix
        self.ncomp = min([ncomp, rank(x)])
        self.nrep = nrep
        self.ncut = ncut
        self.testSize = testSize
        self.criteria = np.full([ncut, self.x.shape[1]], np.nan)
        self.featureR2 = np.empty(ncut)
        self.selFeature = None

    def calcCriteria(self):
        varidx = np.array(range(self.x.shape[1]))
        ncuti = np.logspace(np.log(self.x.shape[1]), np.log(1), self.ncut, base=np.e)
        ncuti = (np.round(ncuti)).astype(int)
        for i in range(self.ncut):
            vcModel = VC(self.x[:, varidx], self.y, self.ncomp, nrep=self.nrep)
            vcModel.fit(isSmooth = False)
            self.criteria[i, varidx] = vcModel.criteria
            var_ranked = np.argsort(vcModel.criteria)
            if i < self.ncut - 1:
                varidx = varidx[var_ranked[:ncuti[i+1]]]

    def evalCriteria(self, cv=3):
        for i in range(self.ncut):
            varSeli = ~np.isnan(self.criteria[i, :])
            xi = self.x[:,varSeli]
            if sum(varSeli) < self.ncomp:
                regModel = LinearRegression()
            else:
                regModel = PLSRegression(min([self.ncomp, rank(xi)]))
            cvScore = cross_val_score(regModel, xi, self.y, cv=cv)
            self.featureR2[i] = np.mean(cvScore)

    def cutFeature(self, *args):
        cuti = np.argmax(self.featureR2)
        self.selFeature = ~np.isnan(self.criteria[cuti, :])
        if len(args) != 0:
            returnx = list(args)
            i = 0
            for argi in args:
                if argi.shape[1] == self.x.shape[1]:
                    returnx[i] = argi[:, self.selFeature]
                i += 1
        return tuple(returnx)


def plotFeatureSelection(wv, X, selFeatures, methodNames = None,
                         ylabel = "Intensity (A.U.)", 
                         xlabel = "Wavelength (nm)",
                         title = "Feature selection results",
                         ax = None):
    """
    Plot the results of feature selection.

    Parameters
    ----------
    wv : numpy.ndarray
        The wavelengths of the spectra.
    X : numpy.ndarray
        The spectra to plot.
    selFeatures : list of int or list of list of int
        The selected features to highlight in the plot.
    methodNames : str or list of str, optional
        The names of the feature selection methods used.
    ylabel : str, optional
        The label for the y-axis.
    xlabel : str, optional
        The label for the x-axis.
    title : str, optional
        The title of the plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plotted axes.
    """
    scale = 0.4
    ofset = 1
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(wv,X.transpose())
    if isinstance(selFeatures[0], int) or \
        isinstance(selFeatures[0], np.int64):
        ax.eventplot(wv[selFeatures],
                     linelengths = scale*np.std(X),
                     lineoffsets = np.min(X)-(scale+ofset)*np.std(X))
        if methodNames is not None:
            ax.text(wv[0]-0.05*(wv[-1]-wv[0]),np.min(X)-(scale+ofset)*np.std(X),methodNames)
    else:
        for i in range(len(selFeatures)):
            ax.eventplot(wv[selFeatures[i]],
                         linelengths = scale*np.std(X),
                         lineoffsets = np.min(X)-(scale*2*i+ofset)*np.std(X))
            if methodNames is not None:
                ax.text(wv[0]-0.05*(wv[-1]-wv[0]), np.min(X)-(scale*2*i+ofset)*np.std(X),
                        methodNames[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax