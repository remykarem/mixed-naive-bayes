.. pandas-lightning documentation master file, created by
   sphinx-quickstart on Thu Jul 23 23:45:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting started
===============

Install using ``pip``:

.. code-block:: bash

   pip install mixed-naive-bayes

Example 1: Discrete and continuous data
---------------------------------------

Below is an example of a dataset with discrete (first 2 columns) and continuous data (last 2). We assume that the
discrete features follow a categorical distribution and the features with the continuous data follow a Gaussian
distribution. Specify ``categorical_features=[0,1]`` then fit and predict as per usual.

>>> from mixed_naive_bayes import MixedNB

>>> X = [[0, 0, 180.9, 75.0],
...      [1, 1, 165.2, 61.5],
...      [2, 1, 166.3, 60.3],
...      [1, 1, 173.0, 68.2],
...      [0, 2, 178.4, 71.0]]
>>> y = [0, 0, 1, 1, 0]

>>> clf = MixedNB(categorical_features=[0,1])
>>> clf.fit(X,y)
>>> clf.predict(X)

.. note:: The module expects that the categorical data be label-encoded accordingly. See the following example to see how.

Example 2: Discrete and continuous data
---------------------------------------

Below is a similar dataset. However, for this dataset we assume a categorical distribution on the first 3 features, and
a Gaussian distribution on the last feature. Feature 3 however has not been label-encoded. We can use sklearn's
[`LabelEncoder()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
preprocessing module to fix this.

>>> import numpy as np
>>> from sklearn.preprocessing import LabelEncoder
>>> X = [[0, 0, 180, 75.0],
...      [1, 1, 165, 61.5],
...      [2, 1, 166, 60.3],
...      [1, 1, 173, 68.2],
...      [0, 2, 178, 71.0]]
>>> y = [0, 0, 1, 1, 0]
>>> X = np.array(X)
>>> y = np.array(y)
>>> label_encoder = LabelEncoder()
>>> X[:,2] = label_encoder.fit_transform(X[:,2])
>>> print(X)
array([[ 0,  0,  4, 75],
[ 1,  1,  0, 61],
[ 2,  1,  1, 60],
[ 1,  1,  2, 68],
[ 0,  2,  3, 71]])

Then fit and predict as usual, specifying ``categorical_features=[0,1,2]`` as the indices that we assume categorical distribution.

>>> from mixed_naive_bayes import MixedNB
>>> clf = MixedNB(categorical_features=[0,1,2])
>>> clf.fit(X,y)
>>> clf.predict(X)

Example 3: Discrete data only
-----------------------------

If all columns are to be treated as discrete, specify ``categorical_features='all'``.

>>> from mixed_naive_bayes import MixedNB
>>> X = [[0, 0],
...      [1, 1],
...      [1, 0],
...      [0, 1],
...      [1, 1]]
>>> y = [0, 0, 1, 0, 1]
>>> clf = MixedNB(categorical_features='all')
>>> clf.fit(X,y)
>>> clf.predict(X)

.. note:: The module expects that the categorical data be label-encoded accordingly. See the previous example to see how.

Example 4: Continuous data only
-------------------------------

If all features are assumed to follow Gaussian distribution, then leave the constructor blank.

>>> from mixed_naive_bayes import MixedNB
>>> X = [[0, 0],
...      [1, 1],
...      [1, 0],
...      [0, 1],
...      [1, 1]]
>>> y = [0, 0, 1, 0, 1]
>>> clf = MixedNB()
>>> clf.fit(X,y)
>>> clf.predict(X)
