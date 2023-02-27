# Standard library
import logging
from warnings import catch_warnings

# Third party libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

# Neurencoding repo imports
from ._models import EphysModel

_logger = logging.getLogger("neurencoding")


class PoissonGLM(EphysModel):
    def __init__(
        self,
        design_matrix,
        spk_times,
        spk_clu,
        binwidth=0.02,
        metric="dsq",
        fit_intercept=True,
        alpha=0,
        mintrials=100,
    ):
        """
        Fit a poisson model using a DesignMatrix and spiking rate.
        Uses the sklearn.linear_model.PoissonRegressor to perform fitting.

        Parameters
        ----------
        design_matrix : neurencoding.DesignMatrix
            Pre-constructed design matrix with the regressors you want for per-neuron fits.
            Must be compiled.
        spk_times : numpy.ndarray
            n_spikes x 1 vector array of times at which spikes were detected
        spk_clu : numpy.ndarray
            n_spikes x 1 vector array of cluster identities corresponding to each spike in
            spk_times
        binwidth : float, optional
            Spikes in input will be binned into non-overlapping bins, this is the width of those
            bins, by default 0.02
        metric : str, optional
            Choice of metric for use by PoissonGLM.score, by default 'dsq'
        fit_intercept : bool, optional
            Whether or not to fit a bias term in the poisson model, by default True
        alpha : float or array, optional
            Regularization strength for the poisson regression, determines the strength of the
            L2 penalty in the objective for fitting. If an array of values is passed,
            sklearn's GridSearchCV will be used to test all values and choose the best via
            cross-validation, by default 0
        mintrials : int, optional
            Minimum number of trials in which a unit must fire at least one spike in order to be
            included in the fitting, by default 100
        """
        super().__init__(design_matrix, spk_times, spk_clu, binwidth, mintrials)
        self.metric = metric
        self.fit_intercept = fit_intercept
        if hasattr(alpha, "shape"):
            self._alpha_grid = alpha
            self.estimator = GridSearchCV(
                PoissonRegressor(fit_intercept=fit_intercept), {"alpha": alpha}, max_iter=300
            )
        else:
            self.alpha = alpha
            self.estimator = PoissonRegressor(
                fit_intercept=fit_intercept, alpha=alpha, max_iter=300
            )

        self.link = np.exp
        self.invlink = np.log

    def _fit(self, dm, binned, cells=None, noncovwarn=True, bestparams=False):
        """
        Fit a GLM using scikit-learn implementation of PoissonRegressor. Uses a regularization
        strength parameter alpha, which is the strength of ridge regularization term.

        Parameters
        ----------
        dm : numpy.ndarray
            Design matrix, in which rows are observations and columns are regressor values. Should
            NOT contain a bias column for the intercept. Scikit-learn handles that.
        binned : numpy.ndarray
            Vector of observed spike counts which we seek to predict. Must be of the same length
            as dm.shape[0]
        alpha : float
            Regularization strength, applied as multiplicative constant on ridge regularization.
        cells : list
            List of cells labels for columns in binned. Will default to all cells in model if None
            is passed. Must be of the same length as columns in binned. By default None.
        bestparams: bool
            Whether or not to
        """
        if cells is None:
            cells = self.clu_ids.flatten()
        if cells.shape[0] != binned.shape[1]:
            raise ValueError("Length of cells does not match shape of binned")

        coefs = pd.Series(index=cells, name="coefficients", dtype=object)
        intercepts = pd.Series(index=cells, name="intercepts")
        alphas = pd.Series(index=cells, name="alphas")
        nonconverged = []
        for cell in tqdm(cells, "Fitting units:", leave=False):
            cell_idx = np.argwhere(cells == cell)[0, 0]
            cellbinned = binned[:, cell_idx]
            with catch_warnings(record=True) as w:
                fitobj = self.estimator.fit(dm, cellbinned)
            if len(w) != 0:
                nonconverged.append(cell)
            if isinstance(self.estimator, GridSearchCV):
                alphas.at[cell] = fitobj.best_params_["alpha"]
                fitobj = fitobj.best_estimator_
            coefs.at[cell] = fitobj.coef_
            if self.fit_intercept:
                intercepts.at[cell] = fitobj.intercept_
            else:
                intercepts.at[cell] = 0
        if noncovwarn:
            if len(nonconverged) != 0:
                _logger.warn("Non-converged cells: {}".format(nonconverged))
        if bestparams:
            return coefs, intercepts, alphas
        return coefs, intercepts
