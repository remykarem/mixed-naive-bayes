# -*- coding: utf-8 -*-

"""
The `mixed_naive_bayes` module implements Categorical and Gaussian
Naive Bayes algorithms. These are supervised learning methods based on
applying Bayes' theorem with strong (naive) feature independence assumptions.

The API's design is similar to scikit-learn's.

Look at the example in `mixed_naive_bayes.mixed_naive_bayes.MixedNB`.
"""

import warnings
import numpy as np

_ALPHA_MIN = 1e-10


class MixedNB():
    """
    Naive Bayes classifier for Categorical and Gaussian models.

    Note: When using categorical_features, MixedNB expects that
    for each feature, all possible classes are captured in the
    trining data X in the `mixed_naive_bayes.mixed_naive_bayes.MixedNB.fit` method.
    This is to ensure numerical stability.

    Parameters
    ----------
    categorical_features : array-like shape (num_categorical_classes,) or 
    'all' (default=None)
        Columns which have categorical feature_distributions
    max_categories : array-like, shape (num_categorical_classes,) (default=None)
        The maximum number of categories that can be found for each 
        categorical feature. If none specified, they will be generated
        automatically.
    alpha : non-negative float, optional (default=0)
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
        This is for features with categorical distribution.
    priors : array-like, size (num_classes,), optional (default=None)
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.
    var_smoothing : float, optional (default=1e-9)
        Portion of the largest variance of all features that is added to
        variances for calculation stability.

    Attributes
    ----------
    priors : array, shape (num_classes,)
        probability of each class.
    epsilon : float
        absolute additive value to variances
    num_samples : int
        number of training samples
    categorical_features : int
        number of classes (number of layes of y)
    gaussian_features : array, shape (num_classes,)
        the distribution for every feature and class
    categorical_posteriors : array
        the distribution of the categorical features
    theta : array
        the mean of the gaussian features
    sigma : array
        the variance of the gaussian features

    References
    ----------
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes

    Example
    -------
    >>> from mixed_naive_bayes import MixedNB
    >>> X = [[0, 0, 180, 75],
             [1, 1, 165, 61],
             [2, 1, 166, 60],
             [1, 1, 173, 68],
             [0, 2, 178, 71]]
    >>> y = [0, 0, 1, 1, 0]
    >>> clf = MixedNB(categorical_features=[0,1])
    >>> clf.fit(X,y)
    >>> clf.predict(X)
    """

    def __init__(self, categorical_features=None, max_categories=None,
                 alpha=0.5, priors=None, var_smoothing=1e-9):
        self.alpha = alpha
        self.var_smoothing = var_smoothing
        self.num_features = 0
        self.epsilon = 1e-9
        self._is_fitted = False
        self.max_categories = max_categories
        self.categorical_features = categorical_features
        self.initial_priors = priors

        self.gaussian_features = []
        self.priors = self.initial_priors
        self.theta = []
        self.sigma = []
        self.categorical_posteriors = []

    def __repr__(self):
        return str(f"{self.__class__.__name__}(alpha={self.alpha}, " +
                   f"var_smoothing={self.var_smoothing})")

    def fit(self, X, y):
        """Fit Mixed Naive Bayes according to X, y

        This method also prepares a `self.models` object. Note that the reason
        why some variables are cast to list() is to make the models object
        JSON serializable.

        Parameters
        ----------
        X : array-like, shape (num_samples, n_features)
            Training vectors, where num_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (num_samples,)
            Target values.

        Returns
        -------
        self : object
        """
        if self._is_fitted:
            self.gaussian_features = []
            self.priors = self.initial_priors
            self.theta = []
            self.sigma = []
            self.categorical_posteriors = []

        # Validate inputs
        self.alpha = _validate_inits(self.alpha)
        X, y = _validate_training_data(
            X, y, self.categorical_features, self.max_categories)

        # From https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/naive_bayes.py#L344
        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon = self.var_smoothing * np.var(X, ddof=1, axis=0).max()

        # Get whatever that is needed
        uniques = np.unique(y)
        num_classes = uniques.size
        (num_samples, self.num_features) = X.shape

        # Correct the inputs
        if self.priors is None:
            self.priors = np.bincount(y)/num_samples
        else:
            self.priors = np.asarray(self.priors)
            if len(self.priors) != num_classes:
                raise ValueError(
                    'Number of priors must match number of classes.')
            if not np.isclose(self.priors.sum(), 1.0):
                raise ValueError("The sum of priors should be 1.")
            if (self.priors < 0).any():
                raise ValueError('Priors must be non-negative.')

        if self.categorical_features is None:
            self.categorical_features = []
        elif self.categorical_features is 'all':
            self.categorical_features = np.arange(0, self.num_features)

        # Get the index columns of the discrete data and continuous data
        self.categorical_features = np.array(
            self.categorical_features).astype(int)
        self.gaussian_features = np.delete(
            np.arange(self.num_features), self.categorical_features)

        # How many categories are there in each categorical_feature
        # Add 1 due to zero-indexing
        if self.max_categories is None:
            self.max_categories = np.max(
                X[:, self.categorical_features], axis=0) + 1
            self.max_categories = self.max_categories.astype(int)
        else:
            self.max_categories = np.array(self.max_categories).astype(int)

        # Prepare empty arrays
        if self.gaussian_features.size != 0:
            self.theta = np.zeros((num_classes, len(self.gaussian_features)))
            self.sigma = np.zeros((num_classes, len(self.gaussian_features)))
        if self.categorical_features.size != 0:
            self.categorical_posteriors = [
                np.zeros((num_classes, num_categories))
                for num_categories in self.max_categories]

        # TODO optimise below!
        for y_i in uniques:

            if self.gaussian_features.size != 0:
                x = X[y == y_i, :][:, self.gaussian_features]
                self.theta[y_i, :] = np.mean(x, axis=0)
                # note: it's really sigma squared
                self.sigma[y_i, :] = np.var(x, axis=0)

            if self.categorical_features.size != 0:
                for i, categorical_feature in enumerate(self.categorical_features):
                    dist = np.bincount(X[y == y_i, :][:, categorical_feature].astype(int),
                                       minlength=self.max_categories[i]) + self.alpha
                    self.categorical_posteriors[i][y_i, :] = dist/np.sum(dist)

        self._is_fitted = True

        return self

    def predict_proba(self, X_test, verbose=False):
        """
        Return probability estimates for the test vector X_test.

        Parameters
        ----------
        X_test : array-like, shape = [num_samples, num_features]

        Returns
        -------
        C : array-like, shape = [num_samples, num_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        if not self._is_fitted:
            raise NotFittedError
        _validate_test_data(X_test, self.num_features)

        X_test = np.array(X_test)

        if self.gaussian_features.size != 0:
            # TODO optimisation: Below is a copy. Can consider masking
            x_gaussian = X_test[:, self.gaussian_features]
            mu = self.theta[:, np.newaxis]
            s = self.sigma[:, np.newaxis] + self.epsilon

            # For every y_class and feature,
            # take values of x's from the samples
            # to get its likelihood
            # (num_classes, num_samples, num_features)
            something = 1./np.sqrt(2.*np.pi*s) * \
                np.exp(-((x_gaussian-mu)**2.)/(2.*s))

            # For every y_class and sample,
            # multiply all the features
            # (num_samples, num_classes)
            t = np.prod(something, axis=2)[:, :, np.newaxis]
            t = np.squeeze(t.T)

        if self.categorical_features.size != 0:

            # Cast tensor to int
            X = X_test[:, self.categorical_features].astype(int)

            # A list of length=num_features.
            # Each item in the list contains the distributions for the y_classes
            # Shape of each item is (num_classes,1,num_samples)
            probas = [categorical_posterior[:, X[:, i][:, np.newaxis]]
                      for i, categorical_posterior
                      in enumerate(self.categorical_posteriors)]

            r = np.concatenate([probas], axis=0)
            r = np.squeeze(r, axis=-1)
            r = np.moveaxis(r, [0, 1, 2], [2, 0, 1])

            # (num_samples, num_classes)
            p = np.prod(r, axis=2).T

        if self.gaussian_features.size != 0 and self.categorical_features.size != 0:
            finals = t * p * self.priors
        elif self.gaussian_features.size != 0:
            finals = t * self.priors
        elif self.categorical_features.size != 0:
            finals = p * self.priors

        normalised = finals.T/(np.sum(finals, axis=1) + 1e-6)
        normalised = np.moveaxis(normalised, [0, 1], [1, 0])

        return normalised

    def predict(self, X, verbose=False):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [num_samples, n_features]

        Returns
        -------
        C : array, shape = [num_samples]
            Predicted target values for X
        """
        probs = self.predict_proba(X, verbose)
        return np.argmax(probs, axis=1)

    def get_params(self, deep=False):
        """Get parameters for this model.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return {
            'categorical_features': self.categorical_features,
            'max_categories': self.max_categories,
            'alpha': self.alpha,
            'priors': self.priors,
            'var_smoothing': self.var_smoothing
        }

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (num_samples, n_features)
            Test samples.
        y : array-like, shape = (num_samples) 
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_true = np.array(y)
        y_predicted = np.array(self.predict(X))
        bool_comparison = y_true == y_predicted

        return np.sum(bool_comparison) / bool_comparison.size


class NotFittedError(Exception):
    """
    Exception class for cases when the predict API is called before
    model is fitted.
    """

    def __str__(self):
        return "This MixedNB instance is not fitted yet. Call 'fit' \
            with appropriate arguments before using this method."


def _validate_test_data(X, num_features):
    X = np.array(X)

    if X.ndim is not 2:
        raise ValueError("Bad input shape of X_test. " +
                         f"Expected an array of dim 2 but got dim {X.ndim} instead.")

    if X.shape[1] != num_features:
        raise ValueError("Bad input shape of X_test. " +
                         f"Expected (,{num_features}) but got (,{X.shape[1]}) instead")


def _validate_inits(alpha):

    if not isinstance(alpha, (int, float)):
        raise TypeError(
            'Expected smoothing parameter alpha to be int or float.')

    if alpha < 0:
        raise ValueError('Expected smoothing parameter alpha > 0. '
                         f'Got {alpha}.')

    if alpha < _ALPHA_MIN:
        warnings.warn('alpha too small will result in numeric errors, '
                      f'setting alpha = {_ALPHA_MIN}')
        alpha = _ALPHA_MIN

    return alpha


def _validate_training_data(X_raw, y_raw, categorical_features, max_categories):
    """Verifying user inputs

    The following will be checked:

    - dimensions 
    - number of samples
    - data type (numbers only)
    - data type for categorical distributions (integers only, starting from 0 onwards)
    - number of categories
    """
    ACCEPTABLE_TYPES = ['float64', 'int64', 'float32', 'int32']
    X = np.array(X_raw)
    y = np.array(y_raw)

    if X.ndim is not 2:
        raise ValueError("Bad input shape of X. " +
                         f"Expected 2D array, but got {X.ndim}D instead. " +
                         "Reshape your data accordingly.")
    if y.ndim is not 1:
        raise ValueError("Bad input shape of y. " +
                         f"Expected 1D array, but got {y.ndim}D instead. " +
                         "Reshape your data accordingly.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "No. of samples in X does not match no. of samples in y")

    if X.dtype not in ACCEPTABLE_TYPES:
        raise TypeError("Expected X to contain only numerics, " +
                         f"but got type {X.dtype} instead. For categorical variables, " +
                         "encode your data using sklearn's LabelEncoder.")

    if y.dtype not in ACCEPTABLE_TYPES:
        raise TypeError("Expected y to contain only numerics, " +
                         f"but got type {y.dtype} instead. For categorical variables, " +
                         "encode your data using sklearn's LabelEncoder.")

    y_classes = np.unique(y)
    if y_classes.size is 1:
        raise ValueError(
            "Found only 1 class in y. There's nothing to classify here!")

    if not np.array_equal(y_classes, np.arange(0, y_classes.size)):
        raise ValueError(f"Expected y to have classes {np.arange(0, y_classes.size)} " +
                         f"but got {y_classes} instead. " +
                         "Encode your data using sklearn's LabelEncoder.")

    if categorical_features is not None:
        if categorical_features is 'all':
            categorical_features = np.arange(0, X.shape[1])
        for feature_no in categorical_features:
            if not np.array_equal(X[:, feature_no], X[:, feature_no].astype(int)):
                warnings.warn(f"Feature no. {feature_no} is continuous data. " +
                              "Casting data to integer.")
            if max_categories is None:
                uniques = np.unique(X[:, feature_no]).astype(int)
                if not np.array_equal(uniques, np.arange(0, np.max(uniques)+1)):
                    raise ValueError(f"Expected feature no. {feature_no} to have " +
                                     f"{np.arange(0,np.max(uniques)+1)} as " +
                                     f"unique values, but got {uniques} instead. " +
                                     "Encode your data using sklearn's LabelEncoder, " + 
                                     "or specify the maximum no. of categories this " + 
                                     "feature can take.")

    return X, y


def load_example():
    """Load an example dataset"""
    X = [[0, 0, 180.1, 75],
         [1, 1, 165, 61],
         [1, 0, 167, 62],
         [0, 1, 178, 63],
         [1, 1, 174, 69],
         [2, 1, 166, 60],
         [0, 2, 167, 59],
         [2, 2, 165, 60],
         [1, 1, 173, 68],
         [0, 2, 178, 71]]
    y = [0, 0, 1, 0, 0, 0, 1, 1, 0, 0]

    return X, y
