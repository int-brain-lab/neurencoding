# Standard library
import logging

# Third party libraries
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.linear_model import LinearRegression

# Neurencoding repo imports
from ._models import EphysModel

_logger = logging.getLogger("neurencoding")


class LinearGLM(EphysModel):

    def __init__(self,
                 design_matrix,
                 spk_times,
                 spk_clu,
                 binwidth=0.02,
                 metric='rsq',
                 estimator=None,
                 mintrials=100):
        """
        Fit a linear model using a DesignMatrix object and spike data. Can use ridge regression
        or pure linear regression

        Parameters
        ----------
        design_matrix : neurencoding.design.DesignMatrix
            Design matrix specification that includes information about groups of regressors
        spk_times : np.array
            1-D Array of spike times
        spk_clu : np.array
            1-D Array, same shape as spk_times, assigning cluster identities to each spike time
        binwidth : float, optional
            Length of the bins to be used to count spikes, by default 0.02
        metric : str, optional
            Scoring metric which to use for the models .score() method. Can be rsq, dsq, msepike,
            nllspike, by default 'rsq'
        estimator : sklearn.linear_model estimator, optional
            Estimator to use for model fitting. If None will default to pure linear regression.
            Must have a .fit(X, y) method and after fitting contain .coef_ and .intercept_
            attributes. By default None.
        train : float, optional
            Proportion of data to use as training set, by default 0.8
        blocktrain : bool, optional
            Whether to use contiguous blocks of trials for cross-validation, by default False
        mintrials : int, optional
            Minimum number of trials in which a neuron must fire >0 spikes to be considered for
            fitting, by default 100
        """
        super().__init__(design_matrix, spk_times, spk_clu, binwidth, mintrials)
        if estimator is None:
            estimator = LinearRegression()
        if not isinstance(estimator, BaseEstimator):
            _logger.warn("Estimator is not a sklearn estimator, may not work as expected."
                         " Trying anyway.")
        self.metric = metric
        self.estimator = estimator
        self.link = lambda x: x
        self.invlink = self.link

    def _fit(self, dm, binned, cells=None, bestparams=False):
        """
        Fitting primitive that brainbox.EphysModel.fit method will call

        Parameters
        ----------
        dm : np.ndarray
            Design matrix to use for fitting
        binned : np.ndarray
            Array of binned spike times. Must share first dimension with dm
        cells : iterable with .shape attribute, optional
            List of cells which are being fit. Use to generate index for output
            coefficients and intercepts, must share shape with second dimension
            of binned. When None will default to a list of all cells in the model object,
            by default None

        Returns
        -------
        coefs, pd.Series
            Series containing fit coefficients for cells
        intercepts, pd.Series
            Series containing intercepts for fits.
        """
        if cells is None:
            cells = self.clu_ids.flatten()
        if cells.shape[0] != binned.shape[1]:
            raise ValueError('Length of cells does not match shape of binned')

        coefs = pd.Series(index=cells, name='coefficients', dtype=object)
        intercepts = pd.Series(index=cells, name='intercepts', dtype=np.float64)
        if isinstance(self.estimator, MetaEstimatorMixin):
            pars = pd.Series(index=cells, name='intercepts', dtype=np.float64)

        for cell in cells:
            cell_idx = np.argwhere(cells == cell)[0, 0]
            lm = self.estimator.fit(dm, binned[:, cell_idx])
            if isinstance(lm, MetaEstimatorMixin):
                est = lm.best_estimator_
                weight, intercept = est.coef_.copy(), est.intercept_
                pars.loc[cell] = lm.best_params_["alpha"]
                del est
            else:
                weight, intercept = lm.coef_.copy(), lm.intercept_
            coefs.at[cell] = weight
            intercepts.at[cell] = intercept
        if bestparams:
            return coefs, intercepts, pars
        return coefs, intercepts
