# Third party libraries
import logging
import numpy as np
import pandas as pd
from numpy.matlib import repmat
from tqdm import tqdm
from brainbox.core import TimeSeries
from brainbox.processing import sync
from one.api import ONE
from brainbox.io.one import SessionLoader


def raised_cosine(duration, nbases, binfun):
    """
    Create raised cosine basis functions of durection `duration` with `nbases` bases.
    
    The `binfun` argument is a function that takes a duration in seconds and returns the number of
    bins in that given duration. Often implemented as e.g. `np.ceil(t / dt).astype(int)`. Needs
    to return an integer.
    """
    nbins = binfun(duration)
    ttb = repmat(np.arange(1, nbins + 1).reshape(-1, 1), 1, nbases)
    dbcenter = nbins / nbases
    cwidth = 4 * dbcenter
    bcenters = 0.5 * dbcenter + dbcenter * np.arange(0, nbases)
    x = ttb - repmat(bcenters.reshape(1, -1), nbins, 1)
    bases = (np.abs(x / cwidth) < 0.5) * (np.cos(x * np.pi * 2 / cwidth) * 0.5 + 0.5)
    return bases


def full_rcos(duration, nbases, binfun, n_before=1):
    """
    Create full raised cosine basis functions of durection `duration` with `nbases` bases.

    This differs from `raised_cosine` in that the basis functions are not restricted to have a
    center within the duration of the window, and `n_before` bases are added before the beginning
    of the basis window.
    
    The `binfun` argument is a function that takes a duration in seconds and returns the number of
    bins in that given duration. Often implemented as e.g. `np.ceil(t / dt).astype(int)`. Needs
    to return an integer.
    """
    if not isinstance(n_before, int):
        n_before = int(n_before)
    nbins = binfun(duration)
    ttb = repmat(np.arange(1, nbins + 1).reshape(-1, 1), 1, nbases)
    dbcenter = nbins / (nbases - 2)
    cwidth = 4 * dbcenter
    bcenters = 0.5 * dbcenter + dbcenter * np.arange(-n_before, nbases - n_before)
    x = ttb - repmat(bcenters.reshape(1, -1), nbins, 1)
    bases = (np.abs(x / cwidth) < 0.5) * (np.cos(x * np.pi * 2 / cwidth) * 0.5 + 0.5)
    return bases



def nonlinear_rcos(duration, nbases, nloffset, binfun):
    if nloffset <= 0:
        raise ValueError('nloffset must be positive and nonzero')
    def basisfun(x, c, dc):
        inner = np.fmax(-np.pi, np.fmin(np.pi, np.pi * (x - c) / (2 * dc)))
        return (np.cos(inner) + 1) / 2
    nbins = binfun(duration)
    nlin = lambda x: np.log(x + 1e-20)
    nlininv = lambda x: np.exp(x) - 1e-20
    yrange = nlin(np.array([0, binfun(duration)]) + binfun(nloffset))
    dbcenter = (yrange[1] - yrange[0]) / (nbases - 1)
    centers = np.arange(yrange[0], yrange[1] + dbcenter, dbcenter)
    maxt = nlininv(yrange[1] + 2 * dbcenter) - binfun(nloffset)
    samplet = nlin(np.arange(nbins) + binfun(nloffset))
    bases = basisfun(samplet.reshape(-1, 1), centers.reshape(1, -1), dbcenter)
    return bases


def neglog(weights, x, y):
    xproj = x @ weights
    f = np.exp(xproj)
    nzidx = f != 0
    if np.any(y[~nzidx] != 0):
        return np.inf
    return -y[nzidx].reshape(1, -1) @ xproj[nzidx] + np.sum(f)


def bincount2D(x, y, xbin=0, ybin=0, xlim=None, ylim=None, weights=None):
    """
    Computes a 2D histogram by aggregating values in a 2D array.

    :param x: values to bin along the 2nd dimension (c-contiguous)
    :param y: values to bin along the 1st dimension
    :param xbin:
        scalar: bin size along 2nd dimension
        0: aggregate according to unique values
        array: aggregate according to exact values (count reduce operation)
    :param ybin:
        scalar: bin size along 1st dimension
        0: aggregate according to unique values
        array: aggregate according to exact values (count reduce operation)
    :param xlim: (optional) 2 values (array or list) that restrict range along 2nd dimension
    :param ylim: (optional) 2 values (array or list) that restrict range along 1st dimension
    :param weights: (optional) defaults to None, weights to apply to each value for aggregation
    :return: 3 numpy arrays MAP [ny,nx] image, xscale [nx], yscale [ny]
    """
    # if no bounds provided, use min/max of vectors
    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
        ylim = [np.min(y), np.max(y)]

    def _get_scale_and_indices(v, bin, lim):
        # if bin is a nonzero scalar, this is a bin size: create scale and indices
        if np.isscalar(bin) and bin != 0:
            scale = np.arange(lim[0], lim[1] + bin / 2, bin)
            ind = (np.floor((v - lim[0]) / bin)).astype(np.int64)
        # if bin == 0, aggregate over unique values
        else:
            scale, ind = np.unique(v, return_inverse=True)
        return scale, ind

    xscale, xind = _get_scale_and_indices(x, xbin, xlim)
    yscale, yind = _get_scale_and_indices(y, ybin, ylim)
    # aggregate by using bincount on absolute indices for a 2d array
    nx, ny = [xscale.size, yscale.size]
    ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
    r = np.bincount(ind2d, minlength=nx * ny, weights=weights).reshape(ny, nx)

    # if a set of specific values is requested output an array matching the scale dimensions
    if not np.isscalar(xbin) and xbin.size > 1:
        _, iout, ir = np.intersect1d(xbin, xscale, return_indices=True)
        _r = r.copy()
        r = np.zeros((ny, xbin.size))
        r[:, iout] = _r[:, ir]
        xscale = xbin

    if not np.isscalar(ybin) and ybin.size > 1:
        _, iout, ir = np.intersect1d(ybin, yscale, return_indices=True)
        _r = r.copy()
        r = np.zeros((ybin.size, r.shape[1]))
        r[iout, :] = _r[ir, :]
        yscale = ybin

    return r, xscale, yscale


def load_trials_df(
    eid,
    one=None,
    t_before=0.0,
    t_after=0.2,
    ret_wheel=False,
    ret_abswheel=False,
    wheel_binsize=0.02,
    addtl_types=[],
    trials_mask=None,
):
    """
    Generate a pandas dataframe of per-trial timing information about a given session.
    Each row in the frame will correspond to a single trial, with timing values indicating timing
    session-wide (i.e. time in seconds since session start). Can optionally return a resampled
    wheel velocity trace of either the signed or absolute wheel velocity.
    The resulting dataframe will have a new set of columns, trial_start and trial_end, which define
    via t_before and t_after the span of time assigned to a given trial.
    (useful for bb.modeling.glm)
    Parameters
    ----------
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    one : one.api.OneAlyx, optional
        one object to use for loading. Will generate internal one if not used, by default None
    t_before : float, optional
        Time before stimulus onset to include for a given trial, as defined by the trial_start
        column of the dataframe. If zero, trial_start will be identical to stimOn, by default 0.
    t_after : float, optional
        Time after feedback to include in the trail, as defined by the trial_end
        column of the dataframe. If zero, trial_end will be identical to feedback, by default 0.
    ret_wheel : bool, optional
        Whether to return the time-resampled wheel velocity trace, by default False
    ret_abswheel : bool, optional
        Whether to return the time-resampled absolute wheel velocity trace, by default False
    wheel_binsize : float, optional
        Time bins to resample wheel velocity to, by default 0.02
    addtl_types : list, optional
        List of additional types from an ONE trials object to include in the dataframe. Must be
        valid keys to the dict produced by one.load_object(eid, 'trials'), by default empty.
    trials_mask : list, optional
        List of trial indices to include in the dataframe. If None, all trials will be included.
    Returns
    -------
    pandas.DataFrame
        Dataframe with trial-wise information. Indices are the actual trial order in the original
        data, preserved even if some trials do not meet the maxlen criterion. As a result will not
        have a monotonic index. Has special columns trial_start and trial_end which define start
        and end times via t_before and t_after
    """
    if not one:
        one = ONE()

    if ret_wheel and ret_abswheel:
        raise ValueError("ret_wheel and ret_abswheel cannot both be true.")

    # Define which datatypes we want to pull out
    trialstypes = [
        "choice",
        "probabilityLeft",
        "feedbackType",
        "feedback_times",
        "contrastLeft",
        "contrastRight",
        "goCue_times",
        "stimOn_times",
    ]
    trialstypes.extend(addtl_types)

    loader = SessionLoader(one=one, eid=eid)
    loader.load_session_data(pose=False, motion_energy=False, pupil=False, wheel=False)
    if ret_wheel or ret_abswheel:
        loader.load_wheel(smooth_size=0.001)

    trials = loader.trials
    starttimes = trials.stimOn_times
    endtimes = trials.feedback_times
    trialsdf = trials[trialstypes]
    if trials_mask is not None:
        trialsdf = trialsdf.loc[trials_mask]
    trialsdf["trial_start"] = trialsdf["stimOn_times"] - t_before
    trialsdf["trial_end"] = trialsdf["feedback_times"] + t_after
    tdiffs = trialsdf["trial_end"] - np.roll(trialsdf["trial_start"], -1)
    if np.any(tdiffs[:-1] > 0):
        logging.warning(
            f"{sum(tdiffs[:-1] > 0)} trials overlapping due to t_before and t_after "
            "values. Try reducing one or both!"
        )
    if not ret_wheel and not ret_abswheel:
        return trialsdf

    wheel = loader.wheel
    whlt, whlv = wheel.times, wheel.velocity
    starttimes = trialsdf["trial_start"]
    endtimes = trialsdf["trial_end"]
    wh_endlast = 0
    trials = []
    for (start, end) in np.vstack((starttimes, endtimes)).T:
        wh_startind = np.searchsorted(whlt[wh_endlast:], start) + wh_endlast
        wh_endind = np.searchsorted(whlt[wh_endlast:], end, side="right") + wh_endlast + 4
        wh_endlast = wh_endind
        tr_whlvel = whlv[wh_startind:wh_endind]
        tr_whlt = whlt[wh_startind:wh_endind] - start
        whlseries = TimeSeries(tr_whlt, tr_whlvel, columns=["whlvel"])
        whlsync = sync(wheel_binsize, timeseries=whlseries, interp="previous")
        trialstartind = np.searchsorted(whlsync.times, 0)
        trialendind = np.ceil((end - start) / wheel_binsize).astype(int)
        trvel = whlsync.values[trialstartind : trialendind + trialstartind]
        if np.abs((trialendind - len(trvel))) > 0:
            raise IndexError("Mismatch between expected length of wheel data and actual.")
        if ret_wheel:
            trials.append(trvel)
        elif ret_abswheel:
            trials.append(np.abs(trvel))
    trialsdf["wheel_velocity"] = trials
    return trialsdf


class SequentialSelector:
    def __init__(self, model, n_features_to_select=None, direction="forward", scoring=None):
        """
        Sequential feature selection for neural models

        Parameters
        ----------
        model : neurencoding.neural_model.EphysModel
            Any class which inherits EphysModel and has already been instantiated.
        n_features_to_select : int, optional
            Number of covariates to select. When None, will sequentially fit all parameters and
            store the associated scores. By default None
        direction : str, optional
            Direction of sequential selection. 'forward' indicates model will be built from 1
            regressor up, while 'backward' indicates regrssors will be removed one at a time until
            n_features_to_select is reached or 1 regressor remains. By default 'forward'
        scoring : str, optional
            Scoring function to use. Must be a valid argument to the subclass of EphysModel passed
            to SequentialSelector. By default None
        """
        self.model = model
        self.design = model.design
        if n_features_to_select:
            self.n_features_to_select = int(n_features_to_select)
        else:
            self.n_features_to_select = len(self.design.covar)
        if direction not in ["forward", "backward"]:
            raise ValueError("direction must be 'forward' or 'backward'")
        self.direction = direction
        self.scoring = scoring
        self.trlabels = self.design.trlabels
        if hasattr(self.model, "traininds"):
            self.train = np.isin(self.trlabels, self.model.traininds).flatten()
            self.test = ~self.train
        else:
            self.train = None
            self.test = None
        self.features = np.array(list(self.design.covar.keys()))

    def fit(self, train_idx=None, full_scores=False, progress=False):
        """
        Fit the sequential feature selection

        Parameters
        ----------
        train_idx : array-like
            indices of trials to use in the training set. If the model passed to the SFS instance
            did not already have training indices, this must be specified. If it did have indices,
            then this will override those.
        full_scores : bool, optional
            Whether to store the full set of submodel scores at each step. Produces additional
            attributes .full_scores_train_ and .full_scores_test_
        progress : bool, optional
            Whether to show a progress bar, by default False
        """
        if train_idx is None and self.train is None:
            raise ValueError(
                "train_idx cannot be None if model used to create SFS did not have "
                "any training indices"
            )
        if train_idx is not None:
            self.train = np.isin(self.trlabels, train_idx).flatten()
            self.test = ~self.train
            test_idx = np.unique(self.trlabels[self.test])
        else:
            train_idx = np.unique(self.trlabels[self.train])
            test_idx = np.unique(self.trlabels[self.test])
        n_features = len(self.features)
        maskdf = pd.DataFrame(index=self.model.clu_ids, columns=self.features, dtype=bool)
        maskdf.loc[:, :] = False
        ncols = (
            self.n_features_to_select
            if self.direction == "forward"
            else n_features - self.n_features_to_select
        )
        seqdf = pd.DataFrame(index=self.model.clu_ids, columns=range(ncols))
        trainscoredf = pd.DataFrame(index=self.model.clu_ids, columns=range(ncols))
        testscoredf = pd.DataFrame(index=self.model.clu_ids, columns=range(ncols))

        if not 0 < self.n_features_to_select <= n_features:
            raise ValueError(
                "n_features_to_select is not a valid number in the context" " of the model."
            )

        n_iterations = (
            self.n_features_to_select
            if self.direction == "forward"
            else n_features - self.n_features_to_select
        )
        if self.direction == "backward":
            self.model.fit(train_idx=train_idx, printcond=False)
            self.basescores_test_ = self.model.score(testinds=test_idx)
            self.basescores_train_ = self.model.score(testinds=train_idx)
        if full_scores:
            fullindex = pd.MultiIndex.from_product(
                [self.model.clu_ids, np.arange(n_iterations)], names=["clu_id", "feature_iter"]
            )
            fulltrain = pd.DataFrame(index=fullindex, columns=range(n_features))
            fulltest = pd.DataFrame(index=fullindex, columns=range(n_features))

        for i in tqdm(
            range(n_iterations), desc="step", leave=False, disable=not progress
        ):  # loop in rpogress
            masks_set = maskdf.groupby(self.features.tolist()).groups
            for current_mask in tqdm(masks_set, desc="feature subset", leave=False):
                cells = masks_set[current_mask]
                outputs = self._get_best_new_feature(current_mask, cells, full_scores)
                if full_scores:
                    new_feature_idx, nf_train, nf_test, nf_fulltrain, nf_fulltest = outputs
                else:
                    new_feature_idx, nf_train, nf_test = outputs
                for cell in cells:
                    maskdf.at[cell, self.features[new_feature_idx.loc[cell]]] = True
                    seqdf.loc[cell, i] = self.features[new_feature_idx.loc[cell]]
                    trainscoredf.loc[cell, i] = nf_train.loc[cell]
                    testscoredf.loc[cell, i] = nf_test.loc[cell]
                    if full_scores:
                        fulltest.loc[cell, i] = nf_fulltest.loc[cell]
                        fulltrain.loc[cell, i] = nf_fulltrain.loc[cell]
        self.support_ = maskdf
        self.sequences_ = seqdf
        self.scores_test_ = testscoredf
        self.scores_train_ = trainscoredf
        if full_scores:
            colnames = {i: k for i, k in enumerate(self.features)}
            self.full_scores_train_ = fulltrain.rename(columns=colnames)
            self.full_scores_test_ = fulltest.rename(columns=colnames)
        if not self.direction == "backward":
            self.deltas_train_ = self._compute_deltas(self.scores_train_, self.sequences_)
            self.deltas_test_ = self._compute_deltas(self.scores_test_, self.sequences_)
        # TODO: Add line here that actually computes Rsq of mean model based on train set when scored on train set,
        # then subtract that from the first column of self.deltas_test

    def _get_best_new_feature(self, mask, cells, full_scores=False):
        """
        Returns
        -------
        maxind, trainmax, testmax, trainscores, testscores
        """
        mask = np.array(mask)
        candidate_features = np.flatnonzero(~mask)
        cell_idxs = np.argwhere(np.isin(self.model.clu_ids, cells)).flatten()
        my = self.model.binnedspikes[np.ix_(self.train, cell_idxs)]
        my_test = self.model.binnedspikes[np.ix_(self.test, cell_idxs)]
        trainscores = pd.DataFrame(index=cells, columns=candidate_features, dtype=float)
        testscores = pd.DataFrame(index=cells, columns=candidate_features, dtype=float)
        for feature_idx in candidate_features:
            candidate_mask = mask.copy()
            candidate_mask[feature_idx] = True
            if self.direction == "backward":
                candidate_mask = ~candidate_mask
            fitfeatures = self.features[candidate_mask]
            feat_idx = np.hstack([self.design.covar[feat]["dmcol_idx"] for feat in fitfeatures])
            mdm = self.design[np.ix_(self.train, feat_idx)]
            mdm_test = self.design[
                np.ix_(self.test, feat_idx)
            ]  # select the rows self.test and columns feat_idx

            coefs, intercepts = self.model._fit(mdm, my, cells=cells)
            for i, cell in enumerate(cells):
                trainscores.at[cell, feature_idx] = self.model._scorer(
                    coefs.loc[cell], intercepts.loc[cell], mdm, my[:, i]
                )
                testscores.at[cell, feature_idx] = self.model._scorer(
                    coefs.loc[cell], intercepts.loc[cell], mdm_test, my_test[:, i]
                )

        maxind = trainscores.idxmax(axis=1)
        trainmax = trainscores.max(axis=1)
        # Ugly kludge to compensate for DataFrame.lookup being deprecated
        midx, cols = pd.factorize(maxind)
        testmax = pd.Series(
            testscores.reindex(cols, axis=1).to_numpy()[np.arange(len(testscores)), midx],
            index=testscores.index,
        )
        if full_scores:
            return maxind, trainmax, testmax, trainscores, testscores
        else:
            return maxind, trainmax, testmax

    @staticmethod
    def _compute_deltas(scores, sequences, direction='forward'):
        if direction == "backwards":
            raise NotImplementedError("No support for backwards delta computation for now")
        n_cov = scores.shape[1]
        diffs = pd.DataFrame(index=scores.index, columns=range(n_cov))
        for i in range(n_cov):
            if i == 0:
                diffs[i] = scores[i]
            else:
                diffs[i] = scores[i] - scores[i - 1]
        diffmelt = pd.melt(
            diffs, ignore_index=False, var_name="position", value_name="diff"
        ).set_index("position", append=True)
        posmelt = pd.melt(
            sequences, ignore_index=False, var_name="position", value_name="covname"
        ).set_index("position", append=True)
        joindf = diffmelt.join(posmelt, how="inner")
        assert len(joindf) == len(diffmelt)
        deltadf = joindf.droplevel("position").pivot(columns="covname", values="diff")
        return deltadf
