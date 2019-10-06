# -*- coding: utf-8 -*-

"""
The `mixed_naive_bayes` module implements Categorical and Gaussian
Naive Bayes algorithms. These are supervised learning methods based on
applying Bayes' theorem with strong (naive) feature independence assumptions.

The API's design is similar to scikit-learn's.

Look at the example in `mixed_naive_bayes.MixedNB`.
"""

from enum import Enum
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG, format='%(funcName)s: %(message)s')
logging.basicConfig(level=logging.INFO, format='%(message)s')


_EPSILON = 1e-8


class Distribution(Enum):
    """Enum class"""
    CATEGORICAL = 1
    GAUSSIAN = 2


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
    class_prior_ : array, shape (n_classes,)
        probability of each class.
    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.
    theta_ : array, shape (n_classes, n_features)
        mean of each feature per class
    sigma_ : array, shape (n_classes, n_features)
        variance of each feature per class
    epsilon_ : float
        absolute additive value to variances

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

    def __init__(self, alpha=0.0, class_prior=None):
        self.alpha = alpha
        self.class_prior = class_prior
        self.num_samples = 0
        self.num_features = 0
        self.num_classes = 0
        self.models = {}
        self._is_fitted = False

    def fit(self, X, y, categorical_features, verbose=False):
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
        verify_training_data(X, y, categorical_features)

        self.num_samples = X.shape[0]
        self.num_features = X.shape[-1]
        self.num_classes = np.unique(y).size
        logger.debug(f"No. of samples:  {self.num_samples}")
        logger.debug(f"No. of features: {self.num_features}")
        logger.debug(f"No. of classes:  {self.num_classes}")

        feature_distributions = [Distribution.CATEGORICAL.value
                                 if i in categorical_features
                                 else Distribution.GAUSSIAN.value
                                 for i in range(self.num_features)]
        logger.debug(f"Distribution of each feature: \
{[Distribution(i).name for i in feature_distributions]}")

        row_indices_for_classes = [
            np.where(y == category)[0] for category in np.unique(y)]
        logger.debug(
            f"Row indices for the different classes: {row_indices_for_classes}")

        # Compute prior probabilities
        if self.class_prior is None:
            self.models["y"] = np.bincount(y)
        else:
            self.models["y"] = np.squeeze(self.class_prior)

        # Compute statistics for x's
        for col in range(self.num_features):

            model_distribution = {}
            model_parameters = {}

            # Count the number of unique labels for this column
            num_uniques = np.unique(X[:, col]).size

            logger.debug(
                f"Distribution for feature {col}: {Distribution(feature_distributions[col]).name}")

            for category in range(self.num_classes):

                arr = np.take(X[:, col], row_indices_for_classes[category])
                logger.debug(f"Data for y={category}: {arr}")

                if feature_distributions[col] == Distribution.CATEGORICAL.value:
                    # The following line accounts for labels that might not exist in
                    # this class but exists in the other class
                    arr = np.bincount(arr, minlength=num_uniques)
                    model_distribution["distribution"] = Distribution.CATEGORICAL.name
                    model_parameters[f"y={category}"] = arr
                    model_distribution["parameters"] = model_parameters
                else:
                    mu = np.mean(arr)
                    sigma = np.std(arr, ddof=1)  # (n-1); Bessel's corrections
                    model_distribution["distribution"] = Distribution.GAUSSIAN.name
                    model_parameters[f"y={category}"] = [mu, sigma]
                    model_distribution["parameters"] = model_parameters

                self.models[f"x{col}"] = model_parameters

            self.models[f"x{col}"] = model_distribution

        logger.debug(self.models)

        self._is_fitted = True
        print("Model fitted")

        if verbose:
            print(self.models)

        return self

    def _prior(self, y_class):
        """
        Compute the prior probability

        p(y = y_class)

        Parameters
        ----------
        y_class : value to condition on

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        prob = self.models["y"][y_class]/self.num_samples
        logger.info(f"p(y={y_class}) \t\t= {prob}")
        return prob

    def _posterior(self, i, x, y_class):
        """
        Compute the posterior probability

        p(x_i = x | y = y_class)

        Parameters
        ----------
        i : column index (zero-indexing)
        x : realised value of feature x_col
        y_class : value to condition on

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        distribution = self.models[f"x{i}"]["distribution"]

        if distribution == Distribution.CATEGORICAL.name:
            freq = self.models[f"x{i}"]["parameters"][f"y={y_class}"]
            prob = (freq[x] + self.alpha) / \
                (sum(freq) + self.alpha*self.num_classes)
        else:
            params = self.models[f"x{i}"]["parameters"][f"y={y_class}"]
            prob = normal_pdf(x, params[0], params[1])

        logger.info(f"p(x_{i}={x}|y={y_class}) \t= {prob}")

        return prob

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

        verify_test_data(X_test, self.num_features)

        # The next 23 lines (from `probabilities_samples_all = []` to
        # `probabilities_samples_all = np.array(probabilities_samples_all)`
        # can be understood by reading the innermost part
        # (1. Get the prior probability) until the outermost (6. Repeat for all samples)
        # Repeating for all samples goes through a for-loop, while the inner parts
        # are calculated using nested list comprehensions.
        probabilities_samples_all = []

        # 6. Repeat for all samples
        for x_test in X_test:
            probabilities_samples_one = [
                # 4. Multiply all the posterior probabilities and prior probability
                np.prod(
                    # 3. Concatenate the posterior probabilities and prior probability
                    np.concatenate([
                        # 2. Get the posterior probability for each feature
                        [self._posterior(feature_no, x_feature, y_class)
                         for feature_no, x_feature in enumerate(x_test)],
                        # 1. Get the prior probability
                        [self._prior(y_class)]
                    ],
                        axis=0))
                # 5. Repeat for all classes
                for y_class in range(self.num_classes)]
            probabilities_samples_all.append(probabilities_samples_one)
        probabilities_samples_all = np.array(probabilities_samples_all)

        logger.info(
            f"Unnormalised probabilities:\n{probabilities_samples_all}")

        # Normalise the class probabilities for each sample. For example,
        # [[0.9,0.6],[0.1,0.2]]
        # becomes
        # [[0.9/1.5,0.6/1.5],[0.1/0.3,0.2/0.3]]
        # Epsilon is added to prevent any division by zero, if any
        normalising_constant = np.sum(
            probabilities_samples_all, axis=1) + _EPSILON
        normalised_probabilities_samples_all = \
            probabilities_samples_all / normalising_constant[:, np.newaxis]

        logger.info(
            f"Normalised probabilities:\n{normalised_probabilities_samples_all}")

        return normalised_probabilities_samples_all

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


def normal_pdf(x, mu, sigma):
    """The probability density function of the normal distribution"""
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))


def verify_test_data(X, num_features):
    X = np.array(X)

    if X.ndim is not 2:
        raise ValueError("Bad input shape of X_test. " +
                         f"Expected an array of dim 2 but got dim {X.ndim} instead.")

    if X.shape[1] != num_features:
        raise ValueError("Bad input shape of X_test. " +
                         f"Expected (,{num_features}) but got (,{X.shape[1]}) instead")


def verify_training_data(X_raw, y_raw, categorical_features):
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
        uniques = np.unique(X[:, feature_no])
        if not np.array_equal(uniques, list(range(np.max(uniques)+1))):
            raise ValueError(f"Expected feature no. {feature_no} to have " +
                             f"{list(range(np.max(uniques)))} " +
                             f"unique values, but got {uniques} instead.")


X, y = load_example()
clf = MixedNB(alpha=0)
clf.fit(X, y, categorical_features=[0, 1])
print(clf.score(X,y))
# print(clf.predict_proba([[0, 0], [1, 1]]))
