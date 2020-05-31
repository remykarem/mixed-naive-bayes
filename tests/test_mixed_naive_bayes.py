# -*- coding: utf-8 -*-

"""
Test cases for Mixed Naive Bayes.
- User inputs
- Correctness
    - discrete data
    - continuous data
"""

import pytest

import numpy as np
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.naive_bayes import GaussianNB

from mixed_naive_bayes import MixedNB, load_example


def test_input_param():
    clf = MixedNB(alpha='l')
    with pytest.raises(TypeError):
        clf.fit([0, 1, 2], [0, 1, 0])


def test_input_string_x():
    clf = MixedNB()
    with pytest.raises(TypeError):
        clf.fit([['X'], ['y']], [0, 1])


def test_input_string_y():
    clf = MixedNB()
    with pytest.raises(TypeError):
        clf.fit([[2], [1]], [0, '1'])


def test_input_wrong_dims():
    clf = MixedNB()
    with pytest.raises(ValueError):
        clf.fit([[0, 1, 2]], [0, 1])


def test_input_wrong_dims_2():
    clf = MixedNB()
    with pytest.raises(ValueError):
        clf.fit([[0, 1, 2]], [[0, 1]])


def test_input_y_not_encoded():
    clf = MixedNB()
    with pytest.raises(ValueError):
        clf.fit([[1, 2], [2, 2], [3, 3]], [0, 8, 0])


def test_continuous_data_iris():
    iris = load_iris()
    X = iris['data']
    y = iris['target']

    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X, y)
    gaussian_nb_pred = gaussian_nb.predict(X)

    mixed_nb = MixedNB()
    mixed_nb.fit(X, y)
    mixed_nb_pred = mixed_nb.predict(X)

    assert (mixed_nb_pred == gaussian_nb_pred).all()


def test_continuous_data_wine():
    wine = load_wine()
    X = wine['data']
    y = wine['target']

    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X, y)
    gaussian_nb_score = gaussian_nb.score(X, y)

    mixed_nb = MixedNB()
    mixed_nb.fit(X, y)
    mixed_nb_score = mixed_nb.score(X, y)

    assert np.isclose(gaussian_nb_score, mixed_nb_score)


def test_continuous_data_digits():
    digits = load_digits()
    X = digits['data']
    y = digits['target']

    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X, y)
    gaussian_nb_score = gaussian_nb.score(X, y)

    mixed_nb = MixedNB()
    mixed_nb.fit(X, y)
    mixed_nb_score = mixed_nb.score(X, y)

    assert np.isclose(gaussian_nb_score, mixed_nb_score)


def test_continuous_data_breast_cancer():
    breast_cancer = load_breast_cancer()
    X = breast_cancer['data']
    y = breast_cancer['target']

    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X, y)
    gaussian_nb_score = gaussian_nb.score(X, y)

    mixed_nb = MixedNB()
    mixed_nb.fit(X, y)
    mixed_nb_score = mixed_nb.score(X, y)

    assert np.isclose(gaussian_nb_score, mixed_nb_score)


def test_categorical_data_digits_all_negative():
    digits = load_digits()
    X = digits['data']
    y = digits['target']

    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X, y)

    mixed_nb = MixedNB(categorical_features='all')
    with pytest.raises(ValueError):
        mixed_nb.fit(X, y)


def test_categorical_data_digits_all():
    digits = load_digits()
    X = digits['data']
    y = digits['target']

    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X, y)
    gaussian_nb.score(X, y)

    mixed_nb = MixedNB(categorical_features='all',
                       max_categories=np.repeat(17, 64))
    mixed_nb.fit(X, y)
    mixed_nb.score(X, y)


def test_categorical_data_digits():
    digits = load_digits()
    X = digits['data']
    y = digits['target']

    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X, y)
    gaussian_nb.score(X, y)

    mixed_nb = MixedNB(categorical_features='all',
                       max_categories=np.repeat(17, 64))
    mixed_nb.fit(X[:1440], y[:1440])
    mixed_nb.score(X[:1440], y[:1440])


def test_categorical_data_simple():
    X, y = load_example()

    mixed_nb = MixedNB([0, 1])
    mixed_nb.fit(X, y)
    mixed_nb.score(X, y)
