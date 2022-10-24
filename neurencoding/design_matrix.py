# Standard library
import logging
from typing import Optional, Union

# Third party libraries
import numba as nb
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

_logger = logging.getLogger('neurencoding')


class DesignMatrix:
    """
    Design matrix constructor that will take in information about the temporal structure of a trial
    and allow the generation of a design matrix with specified regressors
    """

    def __init__(self, trialsdf: pd.DataFrame, vartypes: dict[str, str], binwidth: float = 0.02):
        """
        Class for generating design matrices to model neural data. Provides handy routines for
        describing neural spiking activity using basis functions and other primitives.

        Based on work by Memming Park in:
        Il Memming Park, Miriam LR Meister, Alex C Huk, & Jonathan W Pillow Nature Neuroscience 17,
        1395-1403. (2014)
        and the accompanying code in MATLAB.

        Parameters
        ----------
        trialsdf : pandas.DataFrame
            Dataframe in which each row is a trial, and each column is a trial-by-trial covariate.
            This includes, optionally, continuous covariates like eye position and wheel movement
            per trial, which can be used with object-datatype dataframes. The length of vectors
            stored in continuous-variable columns must match the length of the trial.

            Obligatory columns for the dataframe are "trial_start" and "trial_end", which tell the
            constructor which time points to associate with that trial.
        vartypes : dict
            Dictionary of types for each of the columns in trialsdf. Columns must be of the types:
            -- timing: timing events, in which the column values are times since the start of the
                session of an event within that trial, e.g. stimulus onset.
            -- value: scalars which describe a whole-trial value, such as stimulus contrast or
                probability block
            -- continuous: Columns which contain 1-D vectors per row that describe a covariate that
                changes within the trial. e.g. pupil diameter.
            Dictionary keys should be columns in trialsdf, values should be strings that are equal
            to one of the above.
        binwidth : float, optional
            Length of time bins which will be used for design matrix, by default 0.02
        """
        # Data checks #
        validtypes = ('timing', 'continuous', 'value')
        if not all([name in vartypes for name in trialsdf.columns]):
            raise KeyError("Some columns were not described in vartypes")
        if not all([value in validtypes for value in vartypes.values()]):
            raise ValueError("Invalid values were passed in vartypes")

        # Filter out cells which don't meet the criteria for minimum spiking, while doing trial
        # assignment
        vartypes['duration'] = 'value'
        base_df = trialsdf.copy()
        trialsdf = trialsdf.copy()  # Make sure we don't modify the original dataframe
        trbounds = trialsdf[['trial_start', 'trial_end']]  # Get the start/end of trials
        # Empty trial duration value to use later
        trialsdf['duration'] = np.nan
        # Figure out which columns are timing variables if vartypes was passed
        timingvars = [col for col in trialsdf.columns if vartypes[col] == 'timing']

        for i, (start, end) in trbounds.iterrows():
            if any(np.isnan((start, end))):
                _logger.warning(f"NaN values found in trial start or end at trial number {i}. "
                                "Discarding trial.")
                trialsdf.drop(i, inplace=True)
                continue
            for col in timingvars:
                # Round values for the timing variables to the 5th decimal place and subtract
                # trial start time.
                trialsdf.at[i, col] = np.round(trialsdf.at[i, col] - start, decimals=5)
            trialsdf.at[i, 'duration'] = end - start

        # Set model parameters to begin with
        self.binwidth = binwidth
        self.covar = {}
        self.trialsdf = trialsdf
        self.base_df = base_df
        self.vartypes = vartypes
        self.compiled = False
        return

    def binf(self, t: float):
        """
        Function to bin time t into binwidth of DM

        Parameters
        ----------
        t : float
            time, in seconds

        Returns
        -------
        int
            number of time bins
        """
        return np.ceil(t / self.binwidth).astype(int)

    def add_covariate_timing(self,
                             covlabel: str,
                             eventname: str,
                             bases: np.ndarray,
                             offset: float = 0,
                             deltaval: Optional[Union[str, pd.Series]] = None,
                             cond: Optional[Union[npt.ArrayLike, callable]] = None,
                             desc: str = ''):
        """
        Convenience wrapper for adding timing event regressors to the design matrix.

        Timing events are regressed against using basis functions, such as those generated by the
        functions in this module, which are convolved with a kronecker delta at the time where the
        event occurred. This operation is effectively a copy/paste of the basis functions, each of
        which have their own column in the design matrix. This means that when we fit weights, a
        single weight can govern the prediction of the model over a longer time period.

        Can be offset, such that the bases functions are applied before or after the timing.

        The height of the bases can be modified by a column in the design matrix, which will then
        multiply the delta function which is convolved with the bases.

        Parameters
        ----------
        covlabel : str
            Label which the covariate will use. Can be accessed via dot syntax of the instance
            usually.
        eventname : str
            Label of the column in trialsdf which has the event timing for each trial.
        bases : numpy.array
            nTB x nB array, i.e. number of time bins for the bases functions by number of bases.
            Each column in the array is used together to describe the response of a unit to that
            timing event.
        offset : float, seconds
            Offset of bases functions relative to timing event. Negative values mean bases will
            be applied *before* the timing event by that amount.
        deltaval : None, str, or pandas series, optional
            Values of the kronecker delta function peak used to encode the event. If a string, the
            column in trialsdf with that label will be used. If a pandas series with indexes
            matching trialsdf, corresponding elements of the series will be the delta funtion val.
            If None (default) height is 1.
        cond : None, list, or fun, optional
            Condition which to apply this covariate. Can either be a list of trial indices, or a
            function which takes in rows of the trialsdf and returns booleans.
        desc : str, optional
            Additional information about the covariate, if desired. by default ''
        """
        if covlabel in self.covar:
            raise AttributeError(f'Covariate {covlabel} already exists in model.')
        self._compile_check()
        if deltaval is None:
            gainmod = False
        elif isinstance(deltaval, pd.Series):
            gainmod = True
        elif isinstance(deltaval, str) and deltaval in self.trialsdf.columns:
            gainmod = True
            deltaval = self.trialsdf[deltaval]
        else:
            raise TypeError('deltaval must be None, pandas serizes, or string reference'
                            f' to trialsdf column. {type(deltaval)} was passed instead.')
        if eventname not in self.vartypes:
            raise ValueError(f'{eventname} not in columns of trialsdf.')
        if self.vartypes[eventname] != 'timing':
            raise TypeError(f'Column {eventname} in trialsdf is not registered as a timing')

        vecsizes = self.trialsdf['duration'].apply(self.binf)
        stiminds = self.trialsdf[eventname].apply(self.binf)
        stimvecs = []
        for i in self.trialsdf.index:
            vec = np.zeros(vecsizes[i])
            if gainmod:
                vec[stiminds[i]] = deltaval[i]
            else:
                vec[stiminds[i]] = 1
            stimvecs.append(vec.reshape(-1, 1))
        regressor = pd.Series(stimvecs, index=self.trialsdf.index)
        self.add_covariate(covlabel, regressor, bases, offset, cond, desc)
        return

    def add_covariate_boxcar(self,
                             covlabel: str,
                             boxstart: str,
                             boxend: str,
                             cond: Optional[Union[npt.ArrayLike, callable]] = None,
                             height: Optional[Union[str, pd.Series]] = None,
                             desc: str = ''):
        """
        Convenience wrapped on add_covariate to add a boxcar covariate on the given start and end
        variables, such that the covariate is a step function with non-zero value between those
        values.

        Parameters
        ----------
        covlabel : str
            Name of the covariate for accessing later. Can be accessed via dot syntax of the
            instance usually.
        boxstart : str
            Column name in trialsdf which will be used to define the start of the boxcar
        boxend : str
            Column name in trialsdf which defines the end of boxcar variable
        cond : None, list, or func, optional
            Condition in which to apply this covariate. Can either be a list of trial indices, or
            a function which takes in a row of the trialsdf and returns a boolen on inclusion,
            by default None
        height : None, str, or pandas series, optional
            Values for the height of the boxcar during the period defined per trial. Can be a
            reference to a column in trialsdf or a separate series, by default None
        desc : str, optional
            Additional information about the covariate to store as a string, by default ''

        """
        if covlabel in self.covar:
            raise AttributeError(f'Covariate {covlabel} already exists in model.')
        self._compile_check()
        if boxstart not in self.trialsdf.columns or boxend not in self.trialsdf.columns:
            raise KeyError('boxstart or boxend not found in trialsdf columns.')
        if self.vartypes[boxstart] != 'timing':
            raise TypeError(f'Column {boxstart} in trialsdf is not registered as a timing. '
                            'boxstart and boxend need to refer to timing events in trialsdf.')
        if self.vartypes[boxend] != 'timing':
            raise TypeError(f'Column {boxend} in trialsdf is not registered as a timing. '
                            'boxstart and boxend need to refer to timing events in trialsdf.')

        if isinstance(height, str):
            if height in self.trialsdf.columns:
                if self.vartypes[height]=='continuous': 
                    continuous=1;
                else: 
                    continuous=0;
                height = self.trialsdf[height]
            else:
                raise KeyError(f'{height} is str not in columns of trialsdf')
        elif isinstance(height, pd.Series):
            if len(height) != len(self.trialsdf) or not all(height.index == self.trialsdf.index):
                raise IndexError('Indices of height series does not match trialsdf.')
        elif height is None:
            height = pd.Series(np.ones(len(self.trialsdf.index)), index=self.trialsdf.index)
        vecsizes = self.trialsdf['duration'].apply(self.binf)
        stind = self.trialsdf[boxstart].apply(self.binf)
        endind = self.trialsdf[boxend].apply(self.binf)
        stimvecs = []
        for i in self.trialsdf.index:
            bxcar = np.zeros(vecsizes[i])
            if continuous:
                bxcar[stind[i]:endind[i] + 1] = height[i][stind[i]:endind[i] + 1].T
            else: 
                bxcar[stind[i]:endind[i] + 1] = height[i]
            stimvecs.append(bxcar)
        regressor = pd.Series(stimvecs, index=self.trialsdf.index)
        self.add_covariate(covlabel, regressor, None, offset=0, cond=cond, desc=desc)
        return

    def add_covariate_raw(self,
                          covlabel: str,
                          raw: Union[str, callable, pd.Series],
                          cond: Optional[Union[npt.ArrayLike, callable]] = None,
                          desc: str = ''):
        """
        Convenience wrapper to add a 'raw' covariate, that is to say a covariate which is a
        continuous value that changes with time during the course of a trial.

        Note: This has not been tested and is not guaranteed to work or to work correctly.

        Parameters
        ----------
        covlabel : str
            String used to reference covariate, can usually be accessed by instance's dot syntax
        raw : str, func, or pandas series
            The covariate to add to the design matrix. Can be a str reference to a column in
            trialsdf, a function which takes in rows of trialsdf and produces a vector for each
            row of the appropriate size given binwidth and trial duration, or a pandas series
            of vectors of said appropriate type.
        cond : None, list, or func, optional
            Trials in which to apply the given covariate. Can be a list of trial numbers,
            or a function which accepts rows of the trialsdf and returns a boolean, by default None
        desc : str, optional
            Additional information about the covariate for access later, by default ''
        """
        stimlens = self.trialsdf.duration.apply(self.binf)
        if isinstance(raw, str):
            if raw not in self.trialsdf.columns:
                raise KeyError(f'String {raw} not found in columns of trialsdf. Strings must'
                               'refer to valid column names.')
            covseries = self.trialsdf[raw]
            if np.any(covseries.apply(len) != stimlens):
                raise IndexError(f'Some array shapes in {raw} do not match binned duration.')
            self.add_covariate(covlabel, covseries, None, cond=cond, desc=desc)

        if callable(raw):
            try:
                covseries = self.trialsdf.apply(raw, axis=1)
            except Exception:
                raise TypeError('Function for raw covariate generation did not run properly.'
                                'Make sure that the function passed takes in rows of trialsdf.')
            if np.any(covseries.apply(len) != stimlens):
                raise IndexError(f'Some array shapes in {raw} do not match binned duration.')
            self.add_covariate(covlabel, covseries, None, cond=cond, desc=desc)

        if isinstance(raw, pd.Series):
            if np.any(raw.index != self.trialsdf.index):
                raise IndexError('Indices of raw do not match indices of trialsdf.')
            if np.any(raw.apply(len) != stimlens):
                raise IndexError(f'Some array shapes in {raw} do not match binned duration.')
            self.add_covariate(covlabel, raw, None, cond=cond, desc=desc)

    def add_covariate(self,
                      covlabel: str,
                      regressor: pd.Series,
                      bases: Optional[npt.ArrayLike],
                      offset: float = 0,
                      cond: Optional[Union[npt.ArrayLike, callable]] = None,
                      desc: str = ''):
        """
        Parent function to add covariates to model object. Takes a regressor in the form of a
        pandas Series object, a T x M array of M bases, and stores them for use in the design
        matrix generation.

        Parameters
        ----------
        covlabel : str
            Label for the covariate being added. Will be exposed, if possible, through
            (instance).(covlabel) attribute.
        regressor : pandas.Series
            Series in which each element is the value(s) of a regressor for a trial at that index.
            These will be convolved with the bases functions (if provided) to produce the
            components of the design matrix. *Regressor must be (T / dt) x 1 array for each trial*
        bases : numpy.array or None
            T x M array of M basis functions over T timesteps. Columns will be convolved with the
            elements of `regressor` to produce elements of the design matrix. If None, it is
            assumed a raw regressor is being used.
        offset : int, optional
            Offset of the regressor from the bases during convolution. Negative values indicate
            that the firing of the unit will be , by default 0
        cond : list or func, optional
            Condition for which to apply covariate. Either a list of trials which the covariate
            applies to, or a function of the form f(dataframerow) which returns a boolean,
            by default None
        desc : str, optional
            Description of the covariate for reference purposes, by default '' (empty)
        """
        if covlabel in self.covar:
            raise AttributeError(f'Covariate {covlabel} already exists in model.')
        self._compile_check()
        # Test for mismatch in length of regressor vs trials
        mismatch = np.zeros(len(self.trialsdf.index), dtype=bool)
        for i in self.trialsdf.index:
            currtr = self.trialsdf.loc[i]
            nT = self.binf(currtr.duration)
            if regressor.loc[i].shape[0] != nT:
                mismatch[i] = True

        if np.any(mismatch):
            raise ValueError('Length mismatch between regressor and trial on trials'
                             f'{np.argwhere(mismatch)}.')

        # Initialize containers for the covariate dicts
        if not hasattr(self, 'currcol'):
            self.currcol = 0
        if callable(cond):
            try:
                cond = self.trialsdf.index[self.trialsdf.apply(cond, axis=1)].to_numpy()
            except AttributeError:
                cond = self.trialsdf.index[self.trialsdf.apply(cond, axis=1)]
        if not all(regressor.index == self.trialsdf.index):
            raise IndexError('Indices of regressor and trials dataframes do not match.')

        cov = {
            'description':
                desc,
            'bases':
                bases,
            'valid_trials':
                cond if cond is not None else self.trialsdf.index,
            'offset':
                offset,
            'regressor':
                regressor,
            'dmcol_idx':
                np.arange(self.currcol, self.currcol +
                          bases.shape[1]) if bases is not None else self.currcol
        }
        if bases is None:
            self.currcol += 1
        else:
            self.currcol += bases.shape[1]

        self.covar[covlabel] = cov
        return

    def compile_design_matrix(self, dense: Optional[bool] = True):
        """
        Compiles design matrix for the current experiment based on the covariates which were added
        with the various NeuralGLM.add_covariate methods available. Can optionally compile a sparse
        design matrix using the scipy.sparse package, however that method may take longer depending
        on the degree of sparseness.

        Parameters
        ----------
        dense : bool, optional
            Whether or not to compute a dense design matrix or a sparse one, by default True
        """
        covars = self.covar
        # Go trial by trial and compose smaller design matrices
        miniDMs = []
        rowtrials = []
        for i, trial in self.trialsdf.iterrows():
            nT = self.binf(trial.duration)
            miniX = np.zeros((nT, self.currcol))
            rowlabs = np.ones((nT, 1), dtype=int) * i
            for cov in covars.values():
                sidx = cov['dmcol_idx']
                # Optionally use cond to filter out which trials to apply certain regressors,
                if i not in cov['valid_trials']:
                    continue
                stim = cov['regressor'][i]
                # Convolve Kernel or basis function with stimulus or regressor
                if cov['bases'] is None:
                    miniX[:, sidx] = stim
                else:
                    if len(stim.shape) == 1:
                        stim = stim.reshape(-1, 1)
                    miniX[:, sidx] = convbasis(stim, cov['bases'], self.binf(cov['offset']))
            # Sparsify convolved result and store in miniDMs
            if dense:
                miniDMs.append(miniX)
            else:
                miniDMs.append(sp.lil_matrix(miniX))
            rowtrials.append(rowlabs)
        if dense:
            dm = np.vstack(miniDMs)

        else:
            dm = sp.vstack(miniDMs).to_csc()
        trlabels = np.vstack(rowtrials)
        self.dm = dm
        self.trlabels = trlabels
        self.compiled = True
        return

    def __getitem__(self, key):
        if not self.compiled:
            raise AttributeError('Cannot index uncompiled design matrix')
        return self.dm[key]

    def _compile_check(self):
        if self.compiled:
            _logger.warning(
                'Design matrix was already compiled once. Be sure to compile again if adding'
                ' additional covariates.')
        return


# Precompilation for speed
@nb.njit
def denseconv(X, bases):
    T, dx = X.shape
    TB, M = bases.shape
    indices = np.ones((dx, M))
    sI = np.sum(indices, axis=1)
    BX = np.zeros((T, int(np.sum(sI))))
    sI = np.cumsum(sI)
    k = 0
    for kCov in range(dx):
        A = np.zeros((T + TB - 1, int(np.sum(indices[kCov, :]))))
        for i, j in enumerate(np.argwhere(indices[kCov, :]).flat):
            A[:, i] = np.convolve(X[:, kCov], bases[:, j])
        BX[:, k:sI[kCov]] = A[:T, :]
        k = sI[kCov]
    return BX


def convbasis(stim, bases, offset=0):
    if offset < 0:
        stim = np.pad(stim, ((0, -offset), (0, 0)), 'constant')
    elif offset > 0:
        stim = np.pad(stim, ((offset, 0), (0, 0)), 'constant')

    X = denseconv(stim, bases)

    if offset < 0:
        X = X[-offset:, :]
    elif offset > 0:
        X = X[:-offset, :]
    return X
