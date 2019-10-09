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
        clf.fit([0,1,2], [0,1,0])

def test_input_string():
    clf = MixedNB()
    with pytest.raises(ValueError):
        clf.fit([['X'],['y']], [0,1], [0])

def test_input_wrong_shape():
    clf = MixedNB()
    with pytest.raises(ValueError):
        clf.fit([0,1,2], [0,1], [0])

def test_continuous_data_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X,y)
    gaussian_nb_pred = gaussian_nb.predict(X)
    
    mixed_nb = MixedNB()
    mixed_nb.fit(X,y)
    mixed_nb_pred = mixed_nb.predict(X)
    
    assert (mixed_nb_pred == gaussian_nb_pred).all()

def test_continuous_data_wine():
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X,y)
    gaussian_nb_score = gaussian_nb.score(X,y)
    
    mixed_nb = MixedNB()
    mixed_nb.fit(X,y)
    mixed_nb_score = mixed_nb.score(X,y)
    
    print(mixed_nb.score(X,y))
    print(gaussian_nb.score(X,y))
    
    assert np.abs(mixed_nb_score-gaussian_nb_score) < 0.01
    
    
def test_continuous_data_digits():
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X,y)
    gaussian_nb_score = gaussian_nb.score(X,y)
    
    mixed_nb = MixedNB()
    mixed_nb.fit(X,y)
    mixed_nb_score = mixed_nb.score(X,y)
    
    print(mixed_nb.score(X,y))
    print(gaussian_nb.score(X,y))
    
    assert np.abs(mixed_nb_score-gaussian_nb_score) < 0.01
    
def test_continuous_data_breast_cancer():
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X,y)
    gaussian_nb_score = gaussian_nb.score(X,y)
    
    mixed_nb = MixedNB()
    mixed_nb.fit(X,y)
    mixed_nb_score = mixed_nb.score(X,y)
    
    print(gaussian_nb_score)
    print(mixed_nb_score)
    
    assert np.abs(mixed_nb_score-gaussian_nb_score) < 0.02

    
def test_categorical_data_digits():
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    mixed_nb = MixedNB()
    mixed_nb.fit(X[:1440],y[:1440],list(range(64)))
    mixed_nb_score = mixed_nb.score(X[:1440],y[:1440])
    
    print(mixed_nb_score)
    
def test_categorical_data_simple():
    X, y = load_example()
    
    mixed_nb = MixedNB()
    mixed_nb.fit(X,y,[0,1])
    mixed_nb_score = mixed_nb.score(X,y)
    
    print(mixed_nb_score)
    