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
logging.basicConfig(level=logging.INFO, format='%(funcName)s: %(message)s')


_EPSILON = 1e-8


class Distribution(Enum):
    """Enum class"""
    CATEGORICAL = 1
    GAUSSIAN = 2

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
    y =  [0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
    X = np.array(X)
    y = np.array(y)

    return X, y


class MixedNB():
    """
    Naive Bayes classifier for categorical and Gaussian models.

    Parameters
    ----------
    alpha : float, optional (default=0)
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
        This is for features with categorical distribution.
    priors : array-like, shape (n_classes,)
        Prior probabilities of the classes. If specified, the priors are not
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

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB(priors=None, var_smoothing=1e-09)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    >>> clf_pf = GaussianNB()
    >>> clf_pf.partial_fit(X, Y, np.unique(Y))
    GaussianNB(priors=None, var_smoothing=1e-09)
    >>> print(clf_pf.predict([[-0.8, -1]]))
    [1]
    """

    def __init__(self):
        self.alpha = 0.0
        self.num_classes = 1
        self.is_fitted = False
        self.models = {}
        self.num_samples = 0


    def fit(self, X, y, distributions=[]):
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
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        distributions = [Distribution.CATEGORICAL, Distribution.CATEGORICAL, 
            Distribution.GAUSSIAN, Distribution.GAUSSIAN]
        verify_user_inputs(X, y, distributions)

        indices_for_categories = [
            np.where(y == category)[0] for category in np.unique(y)]
        
        num_features = X.shape[-1]
        self.num_samples = X.shape[0]
        self.num_classes = np.unique(y).size

        # Compute frequency counts for y
        self.models["y"] = np.bincount(y)

        # Compute statistics for x
        for col in range(num_features):

            model_distribution = {}
            model_parameters = {}

            for category in range(self.num_classes):

                arr = np.take(X[:, col], indices_for_categories[category])
                logger.debug(arr)

                if distributions[col] == Distribution.CATEGORICAL:
                    model_distribution["distribution"] = Distribution.CATEGORICAL.name
                    model_parameters[f"y={category}"] = np.bincount(arr)
                    model_distribution["parameters"] = model_parameters
                else:
                    mu = np.mean(arr)
                    sigma = np.std(arr, ddof=1) # (n-1); Bessel's corrections
                    model_distribution["distribution"] = Distribution.GAUSSIAN.name
                    model_parameters[f"y={category}"] = [mu, sigma]
                    model_distribution["parameters"] = model_parameters

                self.models[f"x{col}"] = model_parameters

            self.models[f"x{col}"] = model_distribution

        logger.debug(self.models)

        self.is_fitted = True

        return self


    def _posterior_proba(self, col, val, y):
        """
        Compute the posterior probability

        p(x_col = val | y=y)

        Parameters
        ----------
        col : column index (zero-indexing)
        val : realised value of feature x_col
        y : value to condition on

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        if self.models[f"x{col}"]["distribution"] == Distribution.CATEGORICAL.name:
            freq = self.models[f"x{col}"]["parameters"][f"y={y}"]
            return freq[val]/sum(freq)
        else:
            params = self.models[f"x{col}"]["parameters"][f"y={y}"]
            return normal_pdf(val, params[0], params[1])

    def predict_proba(self, X_test=[[1, 1, 177, 72]], verbose=False):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        # TODO only works for binary
        y_class = 0
        prod = 1
        for i, value in enumerate(X_test[0]):
            prob = self._posterior_proba(col=i, val=value, y=y_class)
            logger.info(f"p(x_{i}={value}|y={y_class}) \t= {prob}")
            prod *= prob

        p_y = self.models["y"][y_class]/self.num_samples
        logger.info(f"p(y={y_class}) \t\t= {p_y}")
        numerator = prod * p_y + self.alpha

        # TODO reuse numerator's value
        denominator = 0
        for y_class, p_y in enumerate(self.models["y"]/self.num_samples):
            prod = 1
            for i, value in enumerate(X_test[0]):
                prob = self._posterior_proba(col=i, val=value, y=y_class)
                logger.info(f"p(x_{i}={value}|y={y_class}) \t= {prob}")
                prod *= prob

            logger.info(f"p(y={y_class}) \t\t= {p_y}")
            denominator += prod * p_y
        
        # To prevent division by 0
        denominator += _EPSILON

        # To prevent division by 0
        denominator += self.alpha * self.num_classes

        return np.array([numerator/denominator, 1-numerator/denominator])

    def predict(self, X_test=[[1, 1, 177, 72]], verbose=False):
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
        if not self.is_fitted:
            print("You shall not pass")
        THRESHOLD = 0.5
        probs = self.predict_proba(X_test=[[1, 1, 177, 72]], verbose=verbose)
        return (np.array(probs) > THRESHOLD).astype(np.int)

    def get_params(self):
        print(json.dumps(self.models))

    def save(self, path):
        """Persist model to disk as JSON?"""
        raise NotImplementedError("Work in progress")

    def load(self, path):
        """Load model from disk"""
        self.is_fitted = True
        raise NotImplementedError("Work in progress")


def normal_pdf(x, mu, sigma):
    """The probability density function of the normal distribution"""
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))

def verify_user_inputs(x, y, dist):
    """Verifying user inputs"""
    pass

X, y = load_example()
clf = MixedNB()
clf.fit(X,y)
# clf.get_params()
print(clf.predict_proba([[1,0,175,65]]))

