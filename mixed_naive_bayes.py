# -*- coding: utf-8 -*-

"""
The `mixed_naive_bayes` module implements Categorical and Gaussian
Naive Bayes algorithms. These are supervised learning methods based on
applying Bayes' theorem with strong (naive) feature independence assumptions.

The API's design is similar to scikit-learn's.

Look at the example in `mixed_naive_bayes.MixedNB`.
"""

# import logging
import numpy as np

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG, format='%(funcName)s: %(message)s')
# logging.basicConfig(level=logging.INFO, format='%(message)s')


_EPSILON = 1e-8


class NotFittedError(Exception):
    """
    Exception class for cases when the predict API is called before
    model is fitted.
    """

    def __str__(self):
        return "This MixedNB instance is not fitted yet. Call 'fit' \
            with appropriate arguments before using this method."


def load_example():
    """Load an example dataset"""
    # Assume all data flushed to 0
    X0 = [[0, 0], [1, 1], [1, 0], [0, 1], [1, 1],
          [2, 1], [0, 2], [2, 2], [1, 1], [0, 2]]
    X1 = [[180, 75], [165, 61], [167, 62],
          [178, 63], [174, 69], [166, 60],
          [167, 59], [165, 60], [173, 68],
          [178, 71]]
    X = [[0, 0, 180, 75],
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
    # X = [[1, 0], [1, 0], [0, 0], [0, 1], [1, 1], [1, 1],
    #      [0, 1], [0, 1], [0, 1], [1, 1], [1, 1], [0, 0]]
    # y = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0]
    X = np.array(X)
    y = np.array(y)

    return X, y


class MixedNB():
    """
    Naive Bayes classifier for categorical and Gaussian models.

    Note: MixedNB expects that for each feature, all possible classes
    are in the dataset or encoded.

    Parameters
    ----------
    alpha : non-negative float, optional (default=0)
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
        This is for features with categorical distribution.
    class_prior : array-like, size (n_classes,), optional (default=None)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    var_smoothing : float, optional (default=1e-9)
        Portion of the largest variance of all features that is added to
        variances for calculation stability.

    Attributes
    ----------
    class_prior : array, shape (n_classes,)
        probability of each class.
    epsilon : float
        absolute additive value to variances
    num_samples : int
        number of training samples
    num_features : int
        number of features of X
    num_classes : int
        number of classes (number of layes of y)
    models : array, shape (n_classes,)
        the distribution for every feature and class

    References
    ----------
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes

    Example
    -------
    >>> import numpy as np
    >>> X = [[1, 0], [1, 0], [0, 0], [0, 1], [1, 1], [1, 1],
             [0, 1], [0, 1], [0, 1], [1, 1], [1, 1], [0, 0]]
    >>> y = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0]
    >>> X = np.array(X)
    >>> y = np.array(y)
    >>> clf = MixedNB()
    >>> clf.fit(X, y, categorical_features=[0, 1])
    >>> print(clf.predict([[0, 0]]))
    """

    def __init__(self, alpha=0.0, class_prior=None, var_smoothing=1e-9):
        self.alpha = alpha
        self.class_prior = class_prior
        self.var_smoothing = var_smoothing
        self.num_classes = 0
        self.num_samples = 0
        self.num_features = 0
        self.epsilon = 1e-9
        self._is_fitted = False

        self.prior = []
        self.theta = []
        self.sigma = []
        self.categorical_posteriors = []
        self.gaussian_features = []
        self.categorical_features = []

    def fit(self, X, y, categorical_features=[], verbose=False):
        """Fit Mixed Naive Bayes according to X, y

        This method also prepares a `self.models` object. Note that the reason
        why some variables are cast to list() is to make the models object
        JSON serializable.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        categorical_features : array
            Columns which have categorical feature_distributions

        Returns
        -------
        self : object
        """
        validate_inits(self.alpha, self.class_prior)
        validate_training_data(X, y, categorical_features)

        X = 

        self.categorical_features = np.array(categorical_features)
        print(self.categorical_features.dtype)

        self.num_classes = np.unique(y).size
        self.num_samples = X.shape[0]
        self.num_features = X.shape[-1]

        # From https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/naive_bayes.py#L344
        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon = self.var_smoothing * np.var(X, axis=0).max()

        self.gaussian_features = np.delete(
            np.arange(self.num_features), self.categorical_features)

        # How many categories are there in each categorical_feature
        # Add 1 due to zero-indexing
        max_categories = np.max(X[:, self.categorical_features], axis=0).astype(np.int64) + 1

        # Prepare empty array
        self.prior = np.zeros((self.num_classes))
        self.theta = np.zeros((self.num_classes, len(self.gaussian_features)))
        self.sigma = np.zeros((self.num_classes, len(self.gaussian_features)))
        self.categorical_posteriors = [
            np.zeros((self.num_classes, num_categories))
            for num_categories in max_categories]

        # Compute prior probabilities
        if self.prior is None:
            self.prior = np.bincount(y)/self.num_samples

        for y_i in np.unique(y):

            if self.gaussian_features.size != 0:
                # TODO experiment the assignment below
                x = X[y == y_i, :][:, self.gaussian_features]
                self.theta[y_i, :] = np.mean(x, axis=0)
                # Bessel's correction; n-1
                self.sigma[y_i, :] = np.std(x, ddof=1, axis=0)

            if self.categorical_features.size != 0:
                for categorical_feature in self.categorical_features:
                    dist = np.bincount(X[y == y_i, :][:, categorical_feature],
                                    minlength=max_categories[categorical_feature])
                    self.categorical_posteriors[categorical_feature][y_i,
                                                                    :] = dist/np.sum(dist)

        self._is_fitted = True
        print("Model fitted")

        return self

    def predict_proba(self, X_test, verbose=False):
        """
        Return probability estimates for the test vector X_test.

        Parameters
        ----------
        X_test : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        if not self._is_fitted:
            raise NotFittedError

        validate_test_data(X_test, self.num_features)

        X_test = np.array(X_test)


        if self.gaussian_features.size != 0:
            x_gaussian = X_test[:, self.gaussian_features]
            mu = self.theta[:, np.newaxis]
            s = self.sigma[:, np.newaxis]

            # For every y_class and feature,
            # take values of x's from the samples 
            # to get its likelihood
            # (num_classes, num_samples, num_features)
            something = 1/np.sqrt(2*np.pi*s**2) * \
                np.exp(-(x_gaussian-mu)**2/(2*s**2))

            # For every y_class and sample, 
            # multiply all the features
            # (num_samples, num_classes)
            t = np.prod(something, axis=2)[:, :, np.newaxis]
            t = np.squeeze(t.T)


        if self.categorical_features.size != 0:

            # Cast tensor to int
            X = X_test[:, self.categorical_features].astype(np.int64)

            # A list of length=num_features. 
            # Each item in the list contains the distributions for the y_classes
            # Shape of each item is (num_classes,1,num_samples)
            probas = [categorical_posterior[:, X_test[:, i][:,np.newaxis]]
                    for i, categorical_posterior in enumerate(self.categorical_posteriors)]

            r = np.concatenate([probas], axis=0)
            r = np.squeeze(r)
            r = np.moveaxis(r, [0,1,2], [2,0,1])

            # (num_samples, num_classes)
            p = np.prod(r, axis=2).T

        if self.gaussian_features.size != 0 and self.categorical_features.size != 0:
            finals = t * p *  self.prior
        elif self.gaussian_features.size != 0:
            finals = t *  self.prior
        elif self.categorical_features.size != 0:
            finals = p *  self.prior

        normalised = finals.T/np.sum(finals, axis=1)
        normalised = np.moveaxis(normalised, [0,1], [1,0])

        return normalised

    def predict(self, X_test, verbose=False):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        probs = self.predict_proba(X_test, verbose)
        return np.argmax(probs, axis=1)

    def get_params(self):
        print(self.models)

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_true = np.array(y)
        y_predicted = np.array(self.predict(X))
        bool_comparison = y_true == y_predicted

        return np.sum(bool_comparison) / bool_comparison.size


def validate_test_data(X, num_features):
    X = np.array(X)

    if X.ndim is not 2:
        raise ValueError("Bad input shape of X_test. " +
                         f"Expected an array of dim 2 but got dim {X.ndim} instead.")

    if X.shape[1] != num_features:
        raise ValueError("Bad input shape of X_test. " +
                         f"Expected (,{num_features}) but got (,{X.shape[1]}) instead")


def validate_inits(alpha, priors):
    if alpha < 0:
        raise ValueError("alpha must be nonnegative.")

    if priors is not None and np.sum(priors) != 1:
        raise ValueError("The sum of the priors should be 1.")


def validate_training_data(X_raw, y_raw, categorical_features):
    """Verifying user inputs

    The following will be checked:

    - dimensions 
    - number of samples
    - data type (numbers only)
    - data type for categorical distributions (integers only, starting from 0 onwards)
    """
    ACCEPTABLE_TYPES = ['float64', 'int64', 'float32', 'int32']
    X = np.array(X_raw)
    y = np.array(y_raw)

    if X.ndim is not 2:
        raise ValueError("Bad input shape of X. " +
                         f"Expected 2D array, but got dim {X.ndim} instead. " +
                         "Reshape your data accordingly.")
    if y.ndim is not 1:
        raise ValueError("Bad input shape of y. " +
                         f"Expected 2D array, but got dim {y.ndim} instead. " +
                         "Reshape your data accordingly.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "No. of samples in X does not match no. of samples in y")

    if X.dtype not in ACCEPTABLE_TYPES:
        raise ValueError("Expected X to contain only numerics, " +
                         f"but got type {X.dtype} instead. For categorical variables, " +
                         "Encode your data using sklearn's LabelEncoder.")

    if y.dtype not in ACCEPTABLE_TYPES:
        raise ValueError("Expected X to contain only numerics, " +
                         f"but got type {y.dtype} instead. For categorical variables, " +
                         "Encode your data using sklearn's LabelEncoder.")

    for feature_no in categorical_features:
        uniques = np.unique(X[:, feature_no]).astype(np.int64)
        if not np.array_equal(uniques, list(range(np.max(uniques)+1))):
            raise ValueError(f"Expected feature no. {feature_no} to have " +
                             f"{list(range(np.max(uniques)))} " +
                             f"unique values, but got {uniques} instead.")
