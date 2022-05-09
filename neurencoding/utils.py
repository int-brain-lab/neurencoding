# Third party libraries
import numpy as np
import pandas as pd
from numpy.matlib import repmat
from tqdm import tqdm


def raised_cosine(duration, nbases, binfun):
    nbins = binfun(duration)
    ttb = repmat(np.arange(1, nbins + 1).reshape(-1, 1), 1, nbases)
    dbcenter = nbins / nbases
    cwidth = 4 * dbcenter
    bcenters = 0.5 * dbcenter + dbcenter * np.arange(0, nbases)
    x = ttb - repmat(bcenters.reshape(1, -1), nbins, 1)
    bases = (np.abs(x / cwidth) < 0.5) * (np.cos(x * np.pi * 2 / cwidth) * 0.5 + 0.5)
    return bases


def full_rcos(duration, nbases, binfun, n_before=1):
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


class SequentialSelector:

    def __init__(self, model, n_features_to_select=None, direction='forward', scoring=None):
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
        self.direction = direction
        self.scoring = scoring
        self.trlabels = self.design.trlabels
        if hasattr(self.model, 'traininds'):
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
            raise ValueError('train_idx cannot be None if model used to create SFS did not have '
                             'any training indices')
        if train_idx is not None:
            self.train = np.isin(self.trlabels, train_idx).flatten()
            self.test = ~self.train
        n_features = len(self.features)
        maskdf = pd.DataFrame(index=self.model.clu_ids, columns=self.features, dtype=bool)
        maskdf.loc[:, :] = False
        seqdf = pd.DataFrame(index=self.model.clu_ids, columns=range(self.n_features_to_select))
        trainscoredf = pd.DataFrame(index=self.model.clu_ids,
                                    columns=range(self.n_features_to_select))
        testscoredf = pd.DataFrame(index=self.model.clu_ids,
                                   columns=range(self.n_features_to_select))

        if not 0 < self.n_features_to_select <= n_features:
            raise ValueError('n_features_to_select is not a valid number in the context'
                             ' of the model.')

        n_iterations = (self.n_features_to_select if self.direction == 'forward' else n_features -
                        self.n_features_to_select)
        if full_scores:
            fullindex = pd.MultiIndex.from_product([self.model.clu_ids, np.arange(n_iterations)],
                                                   names=['clu_id', 'feature_iter'])
            fulltrain = pd.DataFrame(index=fullindex, columns=range(self.n_features_to_select))
            fulltest = pd.DataFrame(index=fullindex, columns=range(self.n_features_to_select))

        for i in tqdm(range(n_iterations), desc='step', leave=False, disable=not progress):
            masks_set = maskdf.groupby(self.features.tolist()).groups
            for current_mask in tqdm(masks_set, desc='feature subset', leave=False):
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
                        fulltest.loc[cell, i] = nf_fulltest
                        fulltrain.loc[cell, i] = nf_fulltrain
        self.support_ = maskdf
        self.sequences_ = seqdf
        self.scores_test_ = testscoredf
        self.scores_train_ = trainscoredf
        if full_scores:
            self.full_scores_train_ = fulltrain
            self.full_scores_test_ = fulltest
        self.deltas_train_ = self._compute_deltas(self.scores_train_, self.sequences_)
        self.deltas_test_ = self._compute_deltas(self.scores_test_, self.sequences_)

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
            if self.direction == 'backward':
                candidate_mask = ~candidate_mask
            fitfeatures = self.features[candidate_mask]
            feat_idx = np.hstack([self.design.covar[feat]['dmcol_idx'] for feat in fitfeatures])
            mdm = self.design[np.ix_(self.train, feat_idx)]
            mdm_test = self.design[np.ix_(self.test, feat_idx)]

            coefs, intercepts = self.model._fit(mdm, my, cells=cells)
            for i, cell in enumerate(cells):
                trainscores.at[cell,
                               feature_idx] = self.model._scorer(coefs.loc[cell],
                                                                 intercepts.loc[cell], mdm, my[:,
                                                                                               i])
                testscores.at[cell,
                              feature_idx] = self.model._scorer(coefs.loc[cell],
                                                                intercepts.loc[cell], mdm_test,
                                                                my_test[:, i])

        maxind = trainscores.idxmax(axis=1)
        trainmax = trainscores.max(axis=1)
        # Ugly kludge to compensate for DataFrame.lookup being deprecated
        midx, cols = pd.factorize(maxind)
        testmax = pd.Series(testscores.reindex(cols, axis=1).to_numpy()[np.arange(len(testscores)),
                                                                        midx],
                            index=testscores.index)
        if full_scores:
            return maxind, trainmax, testmax, trainscores, testscores
        else:
            return maxind, trainmax, testmax

    @staticmethod
    def _compute_deltas(scores, sequences):
        n_cov = scores.shape[1]
        diffs = pd.DataFrame(index=scores.index, columns=range(n_cov))
        for i in range(n_cov):
            if i == 0:
                diffs[i] = scores[i]
            else:
                diffs[i] = scores[i] - scores[i - 1]
        diffmelt = pd.melt(diffs, ignore_index=False, var_name='position',
                           value_name='diff').set_index('position', append=True)
        posmelt = pd.melt(sequences, ignore_index=False, var_name='position',
                          value_name='covname').set_index('position', append=True)
        joindf = diffmelt.join(posmelt, how='inner')
        assert len(joindf) == len(diffmelt)
        deltadf = joindf.droplevel("position").pivot(columns="covname", values="diff")
        return deltadf
