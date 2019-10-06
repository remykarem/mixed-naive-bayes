# -*- coding: utf-8 -*-

"""
Test cases for Mixed Naive Bayes.
"""

import pytest
from mixed_naive_bayes import normal_pdf, MixedNB

def test_normal_pdf():
    assert int(normal_pdf(0,0,1) * 100) / 100 == 0.39

def test_input_string():
    clf = MixedNB()
    with pytest.raises(ValueError):
        clf.fit(X=[['X'],['y']], y=[0,1], categorical_features=[0])

def test_input_wrong_shape():
    clf = MixedNB()
    with pytest.raises(ValueError):
        clf.fit(X=[0,1,2], y=[0,1], categorical_features=[0])
