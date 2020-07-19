from numpy import ma
import sys
import torch
import numpy as np
import torch_models
from functools import partial

from nonconformist.cp import IcpRegressor
from nonconformist.base import RegressorAdapter
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.tree.tree import BaseDecisionTree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.forest import ForestRegressor
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y

from .utils import weighted_percentile

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
from sklearn.model_selection import train_test_split
def qd_objective(y_true, y_l, y_u, alpha__):
    lambda_ = 0.01  # lambda in loss fn

    soften_ = 160.
    n_ = len(y_true)  # batch size
    '''Loss_QD-soft, from algorithm 1'''

    K_HU = np.maximum(0., np.sign(y_u - y_true))
    K_HL = np.maximum(0., np.sign(y_true - y_l))
    K_H = K_HU * K_HL



    MPIW_c = np.sum(((y_u - y_l) * K_H)) / np.sum(K_H)
    PICP_H = np.mean(K_H)



    Loss_SN = MPIW_c / np.max(((y_u - y_l) * K_H)) + lambda_ * n_ / (alpha__ * (1 - alpha__)) * np.maximum(0., (
            1 - alpha__) - PICP_H)

    return Loss_SN

def CV_quntiles_rf(params,
                   X,
                   y,
                   target_coverage,
                   grid_q,
                   test_ratio,
                   random_state,
                   coverage_factor=0.9):
    """ Tune the low and high quantile level parameters of quantile random
        forests method, using cross-validation

    Parameters
    ----------
    params : dictionary of parameters
            params["random_state"] : integer, seed for splitting the data
                                     in cross-validation. Also used as the
                                     seed in quantile random forest (QRF)
            params["min_samples_leaf"] : integer, parameter of QRF
            params["n_estimators"] : integer, parameter of QRF
            params["max_features"] : integer, parameter of QRF
    X : numpy array, containing the training features (nXp)
    y : numpy array, containing the training labels (n)
    target_coverage : desired coverage of prediction band. The output coverage
                      may be smaller if coverage_factor <= 1, in this case the
                      target will be modified to target_coverage*coverage_factor
    grid_q : numpy array, of low and high quantile levels to test
    test_ratio : float, test size of the held-out data
    random_state : integer, seed for splitting the data in cross-validation.
                   Also used as the seed in QRF.
    coverage_factor : float, when tuning the two QRF quantile levels one may
                      ask for prediction band with smaller average coverage,
                      equal to coverage_factor*(q_high - q_low) to avoid too
                      conservative estimation of the prediction band

    Returns
    -------
    best_q : numpy array of low and high quantile levels (length 2)

    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.

    """
    target_coverage = coverage_factor * target_coverage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
    best_avg_length = 1e10
    best_q = grid_q[0]

    rf = RandomForestQuantileRegressor(random_state=params["random_state"],
                                       min_samples_leaf=params["min_samples_leaf"],
                                       n_estimators=params["n_estimators"],
                                       max_features=params["max_features"])
    rf.fit(X_train, y_train)
    best_loss=np.inf
    for q in grid_q:
        y_lower = rf.predict(X_test, quantile=q[0])
        y_upper = rf.predict(X_test, quantile=q[1])

        #coverage, avg_length = compute_coverage_len(y_test, y_lower, y_upper)
        #if (coverage >= target_coverage) and (avg_length < best_avg_length):
        #    best_avg_length = avg_length
        #    best_q = q
        #else:
        #    break
        avg_length=qd_objective(y_test, y_lower, y_upper, q[0]*2)
        if (avg_length < best_loss):
            best_loss = avg_length
            best_q = q
        else:
            break

    return best_q

def generate_sample_indices(random_state, n_samples):
    """
    Generates bootstrap indices for each tree fit.
    Parameters
    ----------
    random_state: int, RandomState instance or None
        If int, random_state is the seed used by the random number generator.
        If RandomState instance, random_state is the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    n_samples: int
        Number of samples to generate from each tree.
    Returns
    -------
    sample_indices: array-like, shape=(n_samples), dtype=np.int32
        Sample indices.
    """
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    return sample_indices


class BaseForestQuantileRegressor(ForestRegressor):
    def fit(self, X, y):
        """
        Build a forest from the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.
        Returns
        -------
        self : object
            Returns self.
        """
        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=False)
        super(BaseForestQuantileRegressor, self).fit(X, y)

        self.y_train_ = y
        self.y_train_leaves_ = -np.ones((self.n_estimators, len(y)), dtype=np.int32)
        self.y_weights_ = np.zeros_like((self.y_train_leaves_), dtype=np.float32)

        for i, est in enumerate(self.estimators_):
            if self.bootstrap:
                bootstrap_indices = generate_sample_indices(
                    est.random_state, len(y))
            else:
                bootstrap_indices = np.arange(len(y))

            est_weights = np.bincount(bootstrap_indices, minlength=len(y))
            y_train_leaves = est.y_train_leaves_
            for curr_leaf in np.unique(y_train_leaves):
                y_ind = y_train_leaves == curr_leaf
                self.y_weights_[i, y_ind] = (
                    est_weights[y_ind] / np.sum(est_weights[y_ind]))

            self.y_train_leaves_[i, bootstrap_indices] = y_train_leaves[bootstrap_indices]
        return self

    def predict(self, X, quantile=None):
        """
        Predict regression value for X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        quantile : int, optional
            Value ranging from 0 to 100. By default, the mean is returned.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array of shape = [n_samples]
            If quantile is set to None, then return E(Y | X). Else return
            y such that F(Y=y | x) = quantile.
        """
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if quantile is None:
            return super(BaseForestQuantileRegressor, self).predict(X)

        sorter = np.argsort(self.y_train_)
        X_leaves = self.apply(X)
        weights = np.zeros((X.shape[0], len(self.y_train_)))
        quantiles = np.zeros((X.shape[0]))
        for i, x_leaf in enumerate(X_leaves):
            mask = self.y_train_leaves_ != np.expand_dims(x_leaf, 1)
            x_weights = ma.masked_array(self.y_weights_, mask)
            weights = x_weights.sum(axis=0)
            quantiles[i] = weighted_percentile(
                self.y_train_, quantile, weights, sorter)
        return quantiles


class RandomForestQuantileRegressor(BaseForestQuantileRegressor):
    """
    A random forest regressor that provides quantile estimates.
    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).
    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.
    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.
    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    Attributes
    ----------
    estimators_ : list of DecisionTreeQuantileRegressor
        The collection of fitted sub-estimators.
    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.
    y_train_ : array-like, shape=(n_samples,)
        Cache the target values at fit time.
    y_weights_ : array-like, shape=(n_estimators, n_samples)
        y_weights_[i, j] is the weight given to sample ``j` while
        estimator ``i`` is fit. If bootstrap is set to True, this
        reduces to a 2-D array of ones.
    y_train_leaves_ : array-like, shape=(n_estimators, n_samples)
        y_train_leaves_[i, j] provides the leaf node that y_train_[i]
        ends up when estimator j is fit. If y_train_[i] is given
        a weight of zero when estimator j is fit, then the value is -1.
    References
    ----------
    .. [1] Nicolai Meinshausen, Quantile Regression Forests
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    """
    def __init__(self,
                 n_estimators=10,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomForestQuantileRegressor, self).__init__(
            base_estimator=DecisionTreeQuantileRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes


class ExtraTreesQuantileRegressor(BaseForestQuantileRegressor):
    """
    An extra-trees regressor that provides quantile estimates.
    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and use averaging to improve the predictive accuracy
    and control over-fitting.
    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.
    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.
    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate the R^2 on unseen data.
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    Attributes
    ----------
    estimators_ : list of ExtraTreeQuantileRegressor
        The collection of fitted sub-estimators.
    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.
    y_train_ : array-like, shape=(n_samples,)
        Cache the target values at fit time.
    y_weights_ : array-like, shape=(n_estimators, n_samples)
        y_weights_[i, j] is the weight given to sample ``j` while
        estimator ``i`` is fit. If bootstrap is set to True, this
        reduces to a 2-D array of ones.
    y_train_leaves_ : array-like, shape=(n_estimators, n_samples)
        y_train_leaves_[i, j] provides the leaf node that y_train_[i]
        ends up when estimator j is fit. If y_train_[i] is given
        a weight of zero when estimator j is fit, then the value is -1.
    References
    ----------
    .. [1] Nicolai Meinshausen, Quantile Regression Forests
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    """
    def __init__(self,
                 n_estimators=10,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(ExtraTreesQuantileRegressor, self).__init__(
            base_estimator=ExtraTreeQuantileRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes


class BaseTreeQuantileRegressor(BaseDecisionTree):
    def predict(self, X, quantile=None, check_input=False):
        """
        Predict regression value for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        quantile : int, optional
            Value ranging from 0 to 100. By default, the mean is returned.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples]
            If quantile is set to None, then return E(Y | X). Else return
            y such that F(Y=y | x) = quantile.
        """
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if quantile is None:
            return super(BaseTreeQuantileRegressor, self).predict(X, check_input=check_input)

        quantiles = np.zeros(X.shape[0])
        X_leaves = self.apply(X)
        unique_leaves = np.unique(X_leaves)
        for leaf in unique_leaves:
            quantiles[X_leaves == leaf] = weighted_percentile(
                self.y_train_[self.y_train_leaves_ == leaf], quantile)
        return quantiles

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.

        Returns
        -------
        self : object
            Returns self.
        """
        # y passed from a forest is 2-D. This is to silence the
        # annoying data-conversion warnings.
        y = np.asarray(y)
        if np.ndim(y) == 2 and y.shape[1] == 1:
            y = np.ravel(y)

        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=False)
        super(BaseTreeQuantileRegressor, self).fit(
            X, y, sample_weight=sample_weight, check_input=check_input,
            X_idx_sorted=X_idx_sorted)
        self.y_train_ = y

        # Stores the leaf nodes that the samples lie in.
        self.y_train_leaves_ = self.tree_.apply(X)
        return self


class DecisionTreeQuantileRegressor(DecisionTreeRegressor, BaseTreeQuantileRegressor):
    """A decision tree regressor that provides quantile estimates.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    y_train_ : array-like
        Train target values.

    y_train_leaves_ : array-like.
        Cache the leaf nodes that each training sample falls into.
        y_train_leaves_[i] is the leaf that y_train[i] ends up at.
    """
    def __init__(self,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 presort=False):
        super(DecisionTreeQuantileRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            presort=presort)


class ExtraTreeQuantileRegressor(ExtraTreeRegressor, BaseTreeQuantileRegressor):
    def __init__(self,
                 criterion='mse',
                 splitter='random',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 random_state=None,
                 max_leaf_nodes=None):
        super(ExtraTreeQuantileRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state)


def compute_coverage_len(y_test, y_lower, y_upper):
    """ Compute average coverage and length of prediction intervals

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    avg_length = np.mean(abs(y_upper - y_lower))
    return coverage, avg_length

def run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance, condition=None):
    """ Run split conformal method

    Parameters
    ----------

    nc : class of nonconformist object
    X_train : numpy array, training features (n1Xp)
    y_train : numpy array, training labels (n1)
    X_test : numpy array, testing features (n2Xp)
    idx_train : numpy array, indices of proper training set examples
    idx_cal : numpy array, indices of calibration set examples
    significance : float, significance level (e.g. 0.1)
    condition : function, mapping feature vector to group id

    Returns
    -------

    y_lower : numpy array, estimated lower bound for the labels (n2)
    y_upper : numpy array, estimated upper bound for the labels (n2)

    """
    icp = IcpRegressor(nc,condition=condition)

    # Fit the ICP using the proper training set
    icp.fit(X_train[idx_train,:], y_train[idx_train])

    # Calibrate the ICP using the calibration set
    icp.calibrate(X_train[idx_cal,:], y_train[idx_cal])

    # Produce predictions for the test set, with confidence 90%
    predictions = icp.predict(X_test, significance=significance)

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    return y_lower, y_upper


def run_icp_sep(nc, X_train, y_train, X_test, idx_train, idx_cal, significance, condition):
    """ Run split conformal method, train a seperate regressor for each group

    Parameters
    ----------

    nc : class of nonconformist object
    X_train : numpy array, training features (n1Xp)
    y_train : numpy array, training labels (n1)
    X_test : numpy array, testing features (n2Xp)
    idx_train : numpy array, indices of proper training set examples
    idx_cal : numpy array, indices of calibration set examples
    significance : float, significance level (e.g. 0.1)
    condition : function, mapping a feature vector to group id

    Returns
    -------

    y_lower : numpy array, estimated lower bound for the labels (n2)
    y_upper : numpy array, estimated upper bound for the labels (n2)

    """
    
    X_proper_train = X_train[idx_train,:]
    y_proper_train = y_train[idx_train]
    X_calibration = X_train[idx_cal,:]
    y_calibration = y_train[idx_cal]
    
    category_map_proper_train = np.array([condition((X_proper_train[i, :], y_proper_train[i])) for i in range(y_proper_train.size)])
    category_map_calibration = np.array([condition((X_calibration[i, :], y_calibration[i])) for i in range(y_calibration.size)])
    category_map_test = np.array([condition((X_test[i, :], None)) for i in range(X_test.shape[0])])
    
    categories = np.unique(category_map_proper_train)

    y_lower = np.zeros(X_test.shape[0])
    y_upper = np.zeros(X_test.shape[0])
    
    cnt = 0

    for cond in categories:
        
        icp = IcpRegressor(nc[cnt])
        
        idx_proper_train_group = category_map_proper_train == cond
        # Fit the ICP using the proper training set
        icp.fit(X_proper_train[idx_proper_train_group,:], y_proper_train[idx_proper_train_group])
    
        idx_calibration_group = category_map_calibration == cond
        # Calibrate the ICP using the calibration set
        icp.calibrate(X_calibration[idx_calibration_group,:], y_calibration[idx_calibration_group])
    
        idx_test_group = category_map_test == cond
        # Produce predictions for the test set, with confidence 90%
        predictions = icp.predict(X_test[idx_test_group,:], significance=significance)
    
        y_lower[idx_test_group] = predictions[:,0]
        y_upper[idx_test_group] = predictions[:,1]
        
        cnt = cnt + 1

    return y_lower, y_upper

def compute_coverage(y_test,y_lower,y_upper,significance,name=""):
    """ Compute average coverage and length, and print results

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    significance : float, desired significance level
    name : string, optional output string (e.g. the method name)

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    print("%s: Percentage in the range (expecting %.2f): %f" % (name, 100 - significance*100, coverage))
    sys.stdout.flush()

    avg_length = abs(np.mean(y_lower - y_upper))
    print("%s: Average length: %f" % (name, avg_length))
    sys.stdout.flush()
    return coverage, avg_length

def compute_coverage_per_sample(y_test,y_lower,y_upper,significance,name="",x_test=None,condition=None):
    """ Compute average coverage and length, and print results

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    significance : float, desired significance level
    name : string, optional output string (e.g. the method name)
    x_test : numpy array, test features
    condition : function, mapping a feature vector to group id

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """
    
    if condition is not None:
        
        category_map = np.array([condition((x_test[i, :], y_test[i])) for i in range(y_test.size)])
        categories = np.unique(category_map)
        
        coverage = np.empty(len(categories), dtype=np.object)
        length = np.empty(len(categories), dtype=np.object)
        
        cnt = 0
        
        for cond in categories:
                        
            idx = category_map == cond
            
            coverage[cnt] = (y_test[idx] >= y_lower[idx]) & (y_test[idx] <= y_upper[idx])

            coverage_avg = np.sum( coverage[cnt] ) / len(y_test[idx]) * 100
            print("%s: Group %d : Percentage in the range (expecting %.2f): %f" % (name, cond, 100 - significance*100, coverage_avg))
            sys.stdout.flush()
        
            length[cnt] = abs(y_upper[idx] - y_lower[idx])
            print("%s: Group %d : Average length: %f" % (name, cond, np.mean(length[cnt])))
            sys.stdout.flush()
            cnt = cnt + 1
    
    else:        
        
        coverage = (y_test >= y_lower) & (y_test <= y_upper)
        coverage_avg = np.sum(coverage) / len(y_test) * 100
        print("%s: Percentage in the range (expecting %.2f): %f" % (name, 100 - significance*100, coverage_avg))
        sys.stdout.flush()
    
        length = abs(y_upper - y_lower)
        print("%s: Average length: %f" % (name, np.mean(length)))
        sys.stdout.flush()
    
    return coverage, length


def plot_func_data(y_test,y_lower,y_upper,name=""):
    """ Plot the test labels along with the constructed prediction band

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    name : string, optional output string (e.g. the method name)

    """

    # allowed to import graphics
    import matplotlib.pyplot as plt

    interval = y_upper - y_lower
    sort_ind = np.argsort(interval)
    y_test_sorted = y_test[sort_ind]
    upper_sorted = y_upper[sort_ind]
    lower_sorted = y_lower[sort_ind]
    mean = (upper_sorted + lower_sorted) / 2

    # Center such that the mean of the prediction interval is at 0.0
    y_test_sorted -= mean
    upper_sorted -= mean
    lower_sorted -= mean

    plt.plot(y_test_sorted, "ro")
    plt.fill_between(
        np.arange(len(upper_sorted)), lower_sorted, upper_sorted, alpha=0.2, color="r",
        label="Pred. interval")
    plt.xlabel("Ordered samples")
    plt.ylabel("Values and prediction intervals")

    plt.title(name)
    plt.show()

    interval = y_upper - y_lower
    sort_ind = np.argsort(y_test)
    y_test_sorted = y_test[sort_ind]
    upper_sorted = y_upper[sort_ind]
    lower_sorted = y_lower[sort_ind]

    plt.plot(y_test_sorted, "ro")
    plt.fill_between(
        np.arange(len(upper_sorted)), lower_sorted, upper_sorted, alpha=0.2, color="r",
        label="Pred. interval")
    plt.xlabel("Ordered samples by response")
    plt.ylabel("Values and prediction intervals")

    plt.title(name)
    plt.show()

###############################################################################
# Deep conditional mean regression
# Minimizing MSE loss
###############################################################################

class MSENet_RegressorAdapter(RegressorAdapter):
    """ Conditional mean estimator, formulated as neural net
    """
    def __init__(self,
                 model,
                 fit_params=None,
                 in_shape=1,
                 hidden_size=1,
                 learn_func=torch.optim.Adam,
                 epochs=1000,
                 batch_size=10,
                 dropout=0.1,
                 lr=0.01,
                 wd=1e-6,
                 test_ratio=0.2,
                 random_state=0):

        """ Initialization

        Parameters
        ----------
        model : unused parameter (for compatibility with nc class)
        fit_params : unused parameter (for compatibility with nc class)
        in_shape : integer, input signal dimension
        hidden_size : integer, hidden layer dimension
        learn_func : class of Pytorch's SGD optimizer
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD
        dropout : float, dropout rate
        lr : float, learning rate for SGD
        wd : float, weight decay
        test_ratio : float, ratio of held-out data, used in cross-validation
        random_state : integer, seed for splitting the data in cross-validation

        """
        super(MSENet_RegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.model = torch_models.mse_model(in_shape=in_shape, hidden_size=hidden_size, dropout=dropout)
        self.loss_func = torch.nn.MSELoss()
        self.learner = torch_models.LearnerOptimized(self.model,
                                                     partial(learn_func, lr=lr, weight_decay=wd),
                                                     self.loss_func,
                                                     device=device,
                                                     test_ratio=self.test_ratio,
                                                     random_state=self.random_state)

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        self.learner.fit(x, y, self.epochs, batch_size=self.batch_size)

    def predict(self, x):
        """ Estimate the label given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of predicted labels (n)

        """
        return self.learner.predict(x)

###############################################################################
# Deep neural network for conditional quantile regression
# Minimizing pinball loss
###############################################################################

class AllQNet_RegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self,
                 model,
                 fit_params=None,
                 in_shape=1,
                 hidden_size=1,
                 quantiles=[.05, .95],
                 learn_func=torch.optim.Adam,
                 epochs=1000,
                 batch_size=10,
                 dropout=0.1,
                 lr=0.01,
                 wd=1e-6,
                 test_ratio=0.2,
                 random_state=0,
                 use_rearrangement=False):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        in_shape : integer, input signal dimension
        hidden_size : integer, hidden layer dimension
        quantiles : numpy array, low and high quantile levels in range (0,1)
        learn_func : class of Pytorch's SGD optimizer
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD
        dropout : float, dropout rate
        lr : float, learning rate for SGD
        wd : float, weight decay
        test_ratio : float, ratio of held-out data, used in cross-validation
        random_state : integer, seed for splitting the data in cross-validation
        use_rearrangement : boolean, use the rearrangement algorithm (True)
                            of not (False). See reference [1].

        References
        ----------
        .. [1]  Chernozhukov, Victor, Iván Fernández‐Val, and Alfred Galichon.
                "Quantile and probability curves without crossing."
                Econometrica 78.3 (2010): 1093-1125.

        """
        super(AllQNet_RegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = quantiles
        if use_rearrangement:
            self.all_quantiles = torch.from_numpy(np.linspace(0.01,0.99,99)).float()
        else:
            self.all_quantiles = self.quantiles
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.model = torch_models.all_q_model(quantiles=self.all_quantiles,
                                              in_shape=in_shape,
                                              hidden_size=hidden_size,
                                              dropout=dropout)
        self.loss_func = torch_models.AllQuantileLoss(self.all_quantiles)
        self.learner = torch_models.LearnerOptimizedCrossing(self.model,
                                                             partial(learn_func, lr=lr, weight_decay=wd),
                                                             self.loss_func,
                                                             device=device,
                                                             test_ratio=self.test_ratio,
                                                             random_state=self.random_state,
                                                             qlow=self.quantiles[0],
                                                             qhigh=self.quantiles[1],
                                                             use_rearrangement=use_rearrangement)

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        self.learner.fit(x, y, self.epochs, self.batch_size)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        return self.learner.predict(x)







###########################new quantile forest


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin


# approach inspired by:
# http://blog.datadive.net/prediction-intervals-for-random-forests/

class QBag(BaseEstimator, RegressorMixin):
    def __init__(self, alpha, base_estimator=None):
        self.alpha = alpha
        if base_estimator is None:
            base_estimator = RandomForestRegressor(100,min_samples_leaf= 5)
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        return self

    def predict(self, X):
        yp = np.zeros(len(X))
        ms = self.base_estimator.estimators_
        for i, x in enumerate(X):
            yps = [m.predict([x])[0] for m in ms]
            yp[i] = np.percentile(yps, self.alpha)
        return yp


class QuantileForestRegressorAdapterNew(RegressorAdapter):
    """ Conditional quantile estimator, defined as quantile random forests (QRF)

    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.

    """

    def __init__(self,
                 model,
                 fit_params=None,
                 quantiles=None,
                 params=None):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        quantiles : numpy array, low and high quantile levels in range (0,100)
        params : dictionary of parameters
                params["random_state"] : integer, seed for splitting the data
                                         in cross-validation. Also used as the
                                         seed in quantile random forests (QRF)
                params["min_samples_leaf"] : integer, parameter of QRF
                params["n_estimators"] : integer, parameter of QRF
                params["max_features"] : integer, parameter of QRF
                params["CV"] : boolean, use cross-validation (True) or
                               not (False) to tune the two QRF quantile levels
                               to obtain the desired coverage
                params["test_ratio"] : float, ratio of held-out data, used
                                       in cross-validation
                params["coverage_factor"] : float, to avoid too conservative
                                            estimation of the prediction band,
                                            when tuning the two QRF quantile
                                            levels in cross-validation one may
                                            ask for prediction intervals with
                                            reduced average coverage, equal to
                                            coverage_factor*(q_high - q_low).
                params["range_vals"] : float, determines the lowest and highest
                                       quantile level parameters when tuning
                                       the quanitle levels bt cross-validation.
                                       The smallest value is equal to
                                       quantiles[0] - range_vals.
                                       Similarly, the largest is equal to
                                       quantiles[1] + range_vals.
                params["num_vals"] : integer, when tuning QRF's quantile
                                     parameters, sweep over a grid of length
                                     num_vals.

        """
        super(QuantileForestRegressorAdapterNew, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = [quantiles[0] / 100, quantiles[1] / 100]
        self.cv_quantiles = self.quantiles
        print(self.quantiles)
        self.params = params
        self.rfqrl = QBag(quantiles[0])
        self.rfqru = QBag(quantiles[1])

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)


        if self.params["CV"]:
            target_coverage = self.quantiles[1] - self.quantiles[0]
            coverage_factor = self.params["coverage_factor"]
            range_vals = self.params["range_vals"]
            num_vals = self.params["num_vals"]
            grid_q_low = np.linspace(self.quantiles[0],self.quantiles[0]+range_vals,num_vals).reshape(-1,1)
            grid_q_high = np.linspace(self.quantiles[1],self.quantiles[1]-range_vals,num_vals).reshape(-1,1)
            grid_q = np.concatenate((grid_q_low,grid_q_high),1)

            self.cv_quantiles =  CV_quntiles_rf(self.params,
                                                              x,
                                                              y,
                                                              target_coverage,
                                                              grid_q,
                                                              self.params["test_ratio"],
                                                              self.params["random_state"],
                                                              coverage_factor)
        """
        self.rfqrl.fit(x, y)
        self.rfqru.fit(x, y)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        lower = self.rfqrl.predict(x)
        upper = self.rfqru.predict(x)

        ret_val = np.zeros((len(lower),2))
        ret_val[:,0] = lower
        ret_val[:,1] = upper
        return ret_val





###############################################################################
# Quantile random forests model
###############################################################################

class QuantileForestRegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator, defined as quantile random forests (QRF)

    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.

    """

    def __init__(self,
                 model,
                 fit_params=None,
                 quantiles=[5, 95],
                 params=None):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        quantiles : numpy array, low and high quantile levels in range (0,100)
        params : dictionary of parameters
                params["random_state"] : integer, seed for splitting the data
                                         in cross-validation. Also used as the
                                         seed in quantile random forests (QRF)
                params["min_samples_leaf"] : integer, parameter of QRF
                params["n_estimators"] : integer, parameter of QRF
                params["max_features"] : integer, parameter of QRF
                params["CV"] : boolean, use cross-validation (True) or
                               not (False) to tune the two QRF quantile levels
                               to obtain the desired coverage
                params["test_ratio"] : float, ratio of held-out data, used
                                       in cross-validation
                params["coverage_factor"] : float, to avoid too conservative
                                            estimation of the prediction band,
                                            when tuning the two QRF quantile
                                            levels in cross-validation one may
                                            ask for prediction intervals with
                                            reduced average coverage, equal to
                                            coverage_factor*(q_high - q_low).
                params["range_vals"] : float, determines the lowest and highest
                                       quantile level parameters when tuning
                                       the quanitle levels bt cross-validation.
                                       The smallest value is equal to
                                       quantiles[0] - range_vals.
                                       Similarly, the largest is equal to
                                       quantiles[1] + range_vals.
                params["num_vals"] : integer, when tuning QRF's quantile
                                     parameters, sweep over a grid of length
                                     num_vals.

        """
        super(QuantileForestRegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = quantiles
        self.cv_quantiles = self.quantiles
        self.params = params
        self.rfqr = RandomForestQuantileRegressor(**params)

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)


        if self.params["CV"]:
            target_coverage = self.quantiles[1] - self.quantiles[0]
            coverage_factor = self.params["coverage_factor"]
            range_vals = self.params["range_vals"]
            num_vals = self.params["num_vals"]
            grid_q_low = np.linspace(self.quantiles[0],self.quantiles[0]+range_vals,num_vals).reshape(-1,1)
            grid_q_high = np.linspace(self.quantiles[1],self.quantiles[1]-range_vals,num_vals).reshape(-1,1)
            grid_q = np.concatenate((grid_q_low,grid_q_high),1)

            self.cv_quantiles =  CV_quntiles_rf(self.params,
                                                              x,
                                                              y,
                                                              target_coverage,
                                                              grid_q,
                                                              self.params["test_ratio"],
                                                              self.params["random_state"],
                                                              coverage_factor)
        """
        self.rfqr.fit(x, y)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        lower = self.rfqr.predict(x, quantile=self.cv_quantiles[0])
        upper = self.rfqr.predict(x, quantile=self.cv_quantiles[1])

        ret_val = np.zeros((len(lower),2))
        ret_val[:,0] = lower
        ret_val[:,1] = upper
        return ret_val



#quantileregresion


import numpy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


class QuantileLinearRegression(LinearRegression):
    """
    Quantile Linear Regression or linear regression
    trained with norm :epkg:`L1`. This class inherits from
    :epkg:`sklearn:linear_models:LinearRegression`.
    See notebook :ref:`quantileregressionrst`.
    Norm :epkg:`L1` is chosen if ``quantile=0.5``, otherwise,
    for *quantile=*:math:`\\rho`,
    the following error is optimized:
    .. math::
        \\sum_i \\rho |f(X_i) - Y_i|^- + (1-\\rho) |f(X_i) - Y_i|^+
    where :math:`|f(X_i) - Y_i|^-= \\max(Y_i - f(X_i), 0)` and
    :math:`|f(X_i) - Y_i|^+= \\max(f(X_i) - Y_i, 0)`.
    :math:`f(i)` is the prediction, :math:`Y_i` the expected
    value.
    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=-1, delta=0.0001, max_iter=10, quantile=0.5,
                 verbose=False):
        """
        Parameters
        ----------
        fit_intercept: boolean, optional, default True
            whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations
            (e.g. data is expected to be already centered).
        normalize: boolean, optional, default False
            This parameter is ignored when ``fit_intercept`` is set to False.
            If True, the regressors X will be normalized before regression by
            subtracting the mean and dividing by the l2-norm.
            If you wish to standardize, please use
            :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
            an estimator with ``normalize=False``.
        copy_X: boolean, optional, default True
            If True, X will be copied; else, it may be overwritten.
        n_jobs: int, optional, default 1
            The number of jobs to use for the computation.
            If -1 all CPUs are used. This will only provide speedup for
            n_targets > 1 and sufficient large problems.
        max_iter: int, optional, default 1
            The number of iteration to do at training time.
            This parameter is specific to the quantile regression.
        delta: float, optional, default 0.0001
            Used to ensure matrices has an inverse
            (*M + delta*I*).
        quantile: float, by default 0.5,
            determines which quantile to use
            to estimate the regression.
        verbose: bool, optional, default False
            Prints error at each iteration of the optimisation.
        """
        LinearRegression.__init__(self, fit_intercept=fit_intercept, normalize=normalize,
                                  copy_X=copy_X, n_jobs=n_jobs)
        self.max_iter = max_iter
        self.verbose = verbose
        self.delta = delta
        self.quantile = quantile

    def fit(self, X, y, sample_weight=None):

        if len(y.shape) > 1 and y.shape[1] != 1:
            raise ValueError("QuantileLinearRegression only works for Y real")

        def compute_z(Xm, beta, Y, W, delta=0.0001):
            "compute z"
            deltas = numpy.ones(X.shape[0]) * delta
            epsilon, mult = QuantileLinearRegression._epsilon(Y, Xm @ beta, self.quantile)
            r = numpy.reciprocal(numpy.maximum(  # pylint: disable=E1111
                epsilon, deltas))  # pylint: disable=E1111
            if mult is not None:
                epsilon *= 1 - mult
                r *= 1 - mult
            return r, epsilon

        if not isinstance(X, numpy.ndarray):
            if hasattr(X, 'values'):
                X = X.values
            else:
                raise TypeError("X must be an array or a dataframe.")

        if self.fit_intercept:
            Xm = numpy.hstack([X, numpy.ones((X.shape[0], 1))])
        else:
            Xm = X

        clr = LinearRegression(fit_intercept=False, copy_X=self.copy_X,
                               n_jobs=self.n_jobs, normalize=self.normalize)

        W = numpy.ones(X.shape[0]) if sample_weight is None else sample_weight
        self.n_iter_ = 0
        lastE = None
        for i in range(0, self.max_iter):
            clr.fit(Xm, y, W)
            beta = clr.coef_
            W, epsilon = compute_z(Xm, beta, y, W, delta=self.delta)
            if sample_weight is not None:
                W *= sample_weight
                epsilon *= sample_weight
            E = epsilon.sum()
            self.n_iter_ = i
            if self.verbose:
                print(
                    '[QuantileLinearRegression.fit] iter={0} error={1}'.format(i + 1, E))
            if lastE is not None and lastE == E:
                break
            lastE = E

        if self.fit_intercept:
            self.coef_ = beta[:-1]
            self.intercept_ = beta[-1]
        else:
            self.coef_ = beta
            self.intercept_ = 0

        return self

    @staticmethod
    def _epsilon(y_true, y_pred, quantile, sample_weight=None):
        diff = y_pred - y_true
        epsilon = numpy.abs(diff)
        if quantile != 0.5:
            sign = numpy.sign(diff)  # pylint: disable=E1111
            mult = numpy.ones(y_true.shape[0])
            mult[sign > 0] *= quantile  # pylint: disable=W0143
            mult[sign < 0] *= (1 - quantile)  # pylint: disable=W0143
        else:
            mult = None
        if sample_weight is not None:
            epsilon *= sample_weight
        return epsilon, mult

    def score(self, X, y, sample_weight=None):
        """
        Returns Mean absolute error regression loss.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            mean absolute error regression loss
        """
        pred = self.predict(X)

        if self.quantile != 0.5:
            epsilon, mult = QuantileLinearRegression._epsilon(
                y, pred, self.quantile, sample_weight)
            if mult is not None:
                epsilon *= mult * 2
            return epsilon.sum() / X.shape[0]
        else:
            return mean_absolute_error(y, pred, sample_weight=sample_weight)


class QuantileRegression(RegressorAdapter):
    """ Conditional quantile estimator, defined as quantile random forests (QRF)

    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.

    """

    def __init__(self,model,quantiles=None,


                 params=None):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        quantiles : numpy array, low and high quantile levels in range (0,100)
        params : dictionary of parameters
                params["random_state"] : integer, seed for splitting the data
                                         in cross-validation. Also used as the
                                         seed in quantile random forests (QRF)
                params["min_samples_leaf"] : integer, parameter of QRF
                params["n_estimators"] : integer, parameter of QRF
                params["max_features"] : integer, parameter of QRF
                params["CV"] : boolean, use cross-validation (True) or
                               not (False) to tune the two QRF quantile levels
                               to obtain the desired coverage
                params["test_ratio"] : float, ratio of held-out data, used
                                       in cross-validation
                params["coverage_factor"] : float, to avoid too conservative
                                            estimation of the prediction band,
                                            when tuning the two QRF quantile
                                            levels in cross-validation one may
                                            ask for prediction intervals with
                                            reduced average coverage, equal to
                                            coverage_factor*(q_high - q_low).
                params["range_vals"] : float, determines the lowest and highest
                                       quantile level parameters when tuning
                                       the quanitle levels bt cross-validation.
                                       The smallest value is equal to
                                       quantiles[0] - range_vals.
                                       Similarly, the largest is equal to
                                       quantiles[1] + range_vals.
                params["num_vals"] : integer, when tuning QRF's quantile
                                     parameters, sweep over a grid of length
                                     num_vals.

        """
        super(QuantileRegression, self).__init__(model  )
        # Instantiate model
        self.quantiles = [quantiles[0]/100,quantiles[1]/100]
        self.cv_quantiles = self.quantiles
        print(self.quantiles)
        self.params = params


    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)


        if self.params["CV"]:
            target_coverage = self.quantiles[1] - self.quantiles[0]
            coverage_factor = self.params["coverage_factor"]
            range_vals = self.params["range_vals"]
            num_vals = self.params["num_vals"]
            grid_q_low = np.linspace(self.quantiles[0],self.quantiles[0]+range_vals,num_vals).reshape(-1,1)
            grid_q_high = np.linspace(self.quantiles[1],self.quantiles[1]-range_vals,num_vals).reshape(-1,1)
            grid_q = np.concatenate((grid_q_low,grid_q_high),1)

            self.cv_quantiles =  CV_quntiles_rf(self.params,
                                                              x,
                                                              y,
                                                              target_coverage,
                                                              grid_q,
                                                              self.params["test_ratio"],
                                                              self.params["random_state"],
                                                              coverage_factor)
        """
        mod1 = QuantileLinearRegression(quantile=self.cv_quantiles[0])
        self.lowerQR = mod1.fit(x, y)
        mod2 = QuantileLinearRegression(quantile=self.cv_quantiles[1])
        self.upperQR = mod2.fit(x, y)



    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        lower = self.lowerQR.predict(x  )
        upper = self.upperQR.predict(x )

        ret_val = np.zeros((len(lower),2))
        ret_val[:,0] = lower
        ret_val[:,1] = upper
        return ret_val







import lightgbm as lgb
class QuantileLightGBM(RegressorAdapter):
    """ Conditional quantile estimator, defined as quantile random forests (QRF)

    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.

    """

    def __init__(self,model,

                 quantiles=None,
                 params=None):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        quantiles : numpy array, low and high quantile levels in range (0,100)
        params : dictionary of parameters
                params["random_state"] : integer, seed for splitting the data
                                         in cross-validation. Also used as the
                                         seed in quantile random forests (QRF)
                params["min_samples_leaf"] : integer, parameter of QRF
                params["n_estimators"] : integer, parameter of QRF
                params["max_features"] : integer, parameter of QRF
                params["CV"] : boolean, use cross-validation (True) or
                               not (False) to tune the two QRF quantile levels
                               to obtain the desired coverage
                params["test_ratio"] : float, ratio of held-out data, used
                                       in cross-validation
                params["coverage_factor"] : float, to avoid too conservative
                                            estimation of the prediction band,
                                            when tuning the two QRF quantile
                                            levels in cross-validation one may
                                            ask for prediction intervals with
                                            reduced average coverage, equal to
                                            coverage_factor*(q_high - q_low).
                params["range_vals"] : float, determines the lowest and highest
                                       quantile level parameters when tuning
                                       the quanitle levels bt cross-validation.
                                       The smallest value is equal to
                                       quantiles[0] - range_vals.
                                       Similarly, the largest is equal to
                                       quantiles[1] + range_vals.
                params["num_vals"] : integer, when tuning QRF's quantile
                                     parameters, sweep over a grid of length
                                     num_vals.

        """
        super(QuantileLightGBM, self).__init__(model  )
        # Instantiate model
        self.quantiles = [quantiles[0]/100,quantiles[1]/100]
        self.cv_quantiles = self.quantiles
        self.params = params
        self.learner_paramsl = params.copy()
        self.learner_paramsu = params.copy()
        self.learner_paramsl['loss'] = 'quantile'
        self.learner_paramsu['loss'] = 'quantile'
        self.learner_paramsl['alpha'] = self.quantiles[0]
        self.learner_paramsu['alpha'] = self.quantiles[1]
        print(self.quantiles)



    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)


        if self.params["CV"]:
            target_coverage = self.quantiles[1] - self.quantiles[0]
            coverage_factor = self.params["coverage_factor"]
            range_vals = self.params["range_vals"]
            num_vals = self.params["num_vals"]
            grid_q_low = np.linspace(self.quantiles[0],self.quantiles[0]+range_vals,num_vals).reshape(-1,1)
            grid_q_high = np.linspace(self.quantiles[1],self.quantiles[1]-range_vals,num_vals).reshape(-1,1)
            grid_q = np.concatenate((grid_q_low,grid_q_high),1)

            self.cv_quantiles =  CV_quntiles_rf(self.params,
                                                              x,
                                                              y,
                                                              target_coverage,
                                                              grid_q,
                                                              self.params["test_ratio"],
                                                              self.params["random_state"],
                                                              coverage_factor)
        """


        self.clfl = lgb.LGBMRegressor(**self.learner_paramsl)
        self.clfl.fit(x, y)

        self.clfu = lgb.LGBMRegressor(**self.learner_paramsu)
        self.clfu.fit(x, y)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        lower = self.clfl.predict(x  )
        upper = self.clfu.predict(x )

        ret_val = np.zeros((len(lower),2))
        ret_val[:,0] = lower
        ret_val[:,1] = upper
        return ret_val

class QuantileGradientBoosting(RegressorAdapter):
    """ Conditional quantile estimator, defined as quantile random forests (QRF)

    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.

    """

    def __init__(self,model,

                 quantiles=None,
                 params=None):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        quantiles : numpy array, low and high quantile levels in range (0,100)
        params : dictionary of parameters
                params["random_state"] : integer, seed for splitting the data
                                         in cross-validation. Also used as the
                                         seed in quantile random forests (QRF)
                params["min_samples_leaf"] : integer, parameter of QRF
                params["n_estimators"] : integer, parameter of QRF
                params["max_features"] : integer, parameter of QRF
                params["CV"] : boolean, use cross-validation (True) or
                               not (False) to tune the two QRF quantile levels
                               to obtain the desired coverage
                params["test_ratio"] : float, ratio of held-out data, used
                                       in cross-validation
                params["coverage_factor"] : float, to avoid too conservative
                                            estimation of the prediction band,
                                            when tuning the two QRF quantile
                                            levels in cross-validation one may
                                            ask for prediction intervals with
                                            reduced average coverage, equal to
                                            coverage_factor*(q_high - q_low).
                params["range_vals"] : float, determines the lowest and highest
                                       quantile level parameters when tuning
                                       the quanitle levels bt cross-validation.
                                       The smallest value is equal to
                                       quantiles[0] - range_vals.
                                       Similarly, the largest is equal to
                                       quantiles[1] + range_vals.
                params["num_vals"] : integer, when tuning QRF's quantile
                                     parameters, sweep over a grid of length
                                     num_vals.

        """
        super(QuantileGradientBoosting, self).__init__(model  )
        # Instantiate model
        self.quantiles = [quantiles[0]/100,quantiles[1]/100]
        self.cv_quantiles = self.quantiles
        self.params = params
        self.learner_paramsl = params.copy()
        self.learner_paramsu = params.copy()
        self.learner_paramsl['loss'] = 'quantile'
        self.learner_paramsu['loss'] = 'quantile'
        self.learner_paramsl['alpha'] = self.quantiles[0]
        self.learner_paramsu['alpha'] = self.quantiles[1]
        print(self.quantiles)



    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)


        if self.params["CV"]:
            target_coverage = self.quantiles[1] - self.quantiles[0]
            coverage_factor = self.params["coverage_factor"]
            range_vals = self.params["range_vals"]
            num_vals = self.params["num_vals"]
            grid_q_low = np.linspace(self.quantiles[0],self.quantiles[0]+range_vals,num_vals).reshape(-1,1)
            grid_q_high = np.linspace(self.quantiles[1],self.quantiles[1]-range_vals,num_vals).reshape(-1,1)
            grid_q = np.concatenate((grid_q_low,grid_q_high),1)

            self.cv_quantiles =  CV_quntiles_rf(self.params,
                                                              x,
                                                              y,
                                                              target_coverage,
                                                              grid_q,
                                                              self.params["test_ratio"],
                                                              self.params["random_state"],
                                                              coverage_factor)
        """


        self.clfl = GradientBoostingRegressor(**self.learner_paramsl)
        self.clfl.fit(x, y)

        self.clfu = GradientBoostingRegressor(**self.learner_paramsu)
        self.clfu.fit(x, y)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        lower = self.clfl.predict(x  )
        upper = self.clfu.predict(x )

        ret_val = np.zeros((len(lower),2))
        ret_val[:,0] = lower
        ret_val[:,1] = upper
        return ret_val


from sklearn.neighbors.regression import KNeighborsRegressor, check_array, _get_weights


class QuantileKNNRegressor(KNeighborsRegressor):
    def __init__(self, alpha_):
        super(QuantileKNNRegressor, self).__init__(n_neighbors=120)
        self.alpha_ = alpha_

    def predict(self, X):
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        ######## Begin modification
        if weights is None:
            y_pred = np.percentile(_y[neigh_ind], self.alpha_ * 100, axis=1)
        else:
            # y_pred = weighted_median(_y[neigh_ind], weights, axis=1)
            raise NotImplementedError("weighted median")
        ######### End modification

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred


class QuantileKNN(RegressorAdapter):
    """ Conditional quantile estimator, defined as quantile random forests (QRF)

    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.

    """

    def __init__(self,model,

                 quantiles=None,
                 params=None):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        quantiles : numpy array, low and high quantile levels in range (0,100)
        params : dictionary of parameters
                params["random_state"] : integer, seed for splitting the data
                                         in cross-validation. Also used as the
                                         seed in quantile random forests (QRF)
                params["min_samples_leaf"] : integer, parameter of QRF
                params["n_estimators"] : integer, parameter of QRF
                params["max_features"] : integer, parameter of QRF
                params["CV"] : boolean, use cross-validation (True) or
                               not (False) to tune the two QRF quantile levels
                               to obtain the desired coverage
                params["test_ratio"] : float, ratio of held-out data, used
                                       in cross-validation
                params["coverage_factor"] : float, to avoid too conservative
                                            estimation of the prediction band,
                                            when tuning the two QRF quantile
                                            levels in cross-validation one may
                                            ask for prediction intervals with
                                            reduced average coverage, equal to
                                            coverage_factor*(q_high - q_low).
                params["range_vals"] : float, determines the lowest and highest
                                       quantile level parameters when tuning
                                       the quanitle levels bt cross-validation.
                                       The smallest value is equal to
                                       quantiles[0] - range_vals.
                                       Similarly, the largest is equal to
                                       quantiles[1] + range_vals.
                params["num_vals"] : integer, when tuning QRF's quantile
                                     parameters, sweep over a grid of length
                                     num_vals.

        """
        super(QuantileKNN, self).__init__(model  )
        # Instantiate model
        self.quantiles = [quantiles[0]/100,quantiles[1]/100]
        self.cv_quantiles = self.quantiles
        self.params = params
        self.learner_paramsl = params.copy()
        self.learner_paramsu = params.copy()

        self.learner_paramsl['alpha_'] = self.quantiles[0]
        self.learner_paramsu['alpha_'] = self.quantiles[1]
        print(self.quantiles)



    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)


        if self.params["CV"]:
            target_coverage = self.quantiles[1] - self.quantiles[0]
            coverage_factor = self.params["coverage_factor"]
            range_vals = self.params["range_vals"]
            num_vals = self.params["num_vals"]
            grid_q_low = np.linspace(self.quantiles[0],self.quantiles[0]+range_vals,num_vals).reshape(-1,1)
            grid_q_high = np.linspace(self.quantiles[1],self.quantiles[1]-range_vals,num_vals).reshape(-1,1)
            grid_q = np.concatenate((grid_q_low,grid_q_high),1)

            self.cv_quantiles =  CV_quntiles_rf(self.params,
                                                              x,
                                                              y,
                                                              target_coverage,
                                                              grid_q,
                                                              self.params["test_ratio"],
                                                              self.params["random_state"],
                                                              coverage_factor)
        """


        self.clfl = QuantileKNNRegressor(**self.learner_paramsl)
        self.clfl.fit(x, y)

        self.clfu = QuantileKNNRegressor(**self.learner_paramsu)
        self.clfu.fit(x, y)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        lower = self.clfl.predict(x  )
        upper = self.clfu.predict(x )

        ret_val = np.zeros((len(lower),2))
        ret_val[:,0] = lower
        ret_val[:,1] = upper
        return ret_val
