# Mixed Naive Bayes (WIP)

Naive Bayes classifiers are a set of supervised learning algorithms based on applying Bayes' theorem, but with strong independence assumptions between the features given the value of the class variable (hence naive).

This module implements **Categorical** (Multinoulli) and **Gaussian** Naive Bayes algorithms. This means that we not only assume that features (given their respective *y*'s) follow the Gaussian distribution, but also the categorical distribution. Hence it is natural that the continous data be attributed to the Gaussian and the categorical data (nominal or ordinal) be attributed the the categorical distribution.

The motivation for writing this library is that scikit-learn does not have an implementation for mixed type of naive bayes. They have one for `CategoricalNB` [here](https://github.com/scikit-learn/scikit-learn/blob/86aea9915/sklearn/naive_bayes.py#L1021) but it's still pending.

I like `scikit-learn`'s APIs (`.fit()`, `.predict()` 😍) so if you use it a lot, you'll find that it's easy to get started started with this library.

I've written a tutorial [here](https://remykarem.github.io/blog/naive-bayes) for naive bayes if you need to understand a bit more on the math.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Performance (Accuracy)](#performance-accuracy)
- [Performance (Speed)](#performance-speed)
- [Tests](#tests)
- [API Documentation](#api-documentation)
- [To-Dos](#to-dos)
- [References](#references)
- [Related work](#related-work)

## Installation

### via pip

```bash
pip install git+https://github.com/remykarem/mixed-naive-bayes#egg=mixed_naive_bayes
```

## Quick start

### Categorical and Gaussian

Below is an example of a dataset with discrete and continuous data. Specify the indices of the features which are to follow the categorical distribution (columns `0` and `1`). Then fit and
predict as per usual.

```python
from mixed_naive_bayes import MixedNB
import numpy as np
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
clf = MixedNB(categorical_features=[0,1])
clf.fit(X,y)
clf.predict(X)
```

**NOTE: The module expects that you treat the categorical data be label encoded accordingly.**

### Categorical only

Specify the indices of the features which are
to follow the categorical distribution.

```python
from mixed_naive_bayes import MixedNB
clf = MixedNB([0,1])
clf.fit(X,y)
clf.score(X)
```

### Gaussian only

If all columns are to be treated as Gaussian, then leave the constructor blank.

```python
from mixed_naive_bayes import MixedNB
clf = MixedNB()
clf.fit(X,y)
clf.predict(X)
```

## Requirements

- `Python>=3.6`
- `numpy>=1.16.1`

The `scikit-learn` library is used to import data as seen in the examples. Otherwise, the module itself does not require it.

The `pytest` library is not needed unless you want to perform testing.

## Performance (Accuracy)

Measures the accuracy of (1) using categorical data and (2) my Gaussian implementation.

Dataset | GaussianNB | MixedNB (G) | MixedNB (C) | MixedNB (G+C) |
------- | ---------- | ----------- | ----------- | ------------- |
Iris    | 0.960      | 0.960       | -           | - |
Digits  | 0.858      | 0.858       | **0.961**   | - |
Wine    | 0.989      | 0.989       | -           | - |
Cancer  | 0.942      | 0.942       | -           | - |
covtype | 0.616      | 0.616       |             | **0.657** |

G - Gaussian only
C - categorical only
G+C - Gaussian and categorical

## Performance (Speed)

The library is written in [NumPy](https://numpy.org/), so many operations are vectorised and faster than their for-loop counterparts.

(Still measuring)

## Tests

I'm still writing more test cases, but in the meantime, you can run the following:

```bash
pytest tests.py
```

## API Documentation

For more information on usage of the API, visit [here](https://remykarem.github.io/docs/mixed_naive_bayes.html). This was generated using pdoc3.

## To-Dos

- [ ] Write more test cases
- [ ] Performance (Speed)
- [X] Support refitting
- [X] Regulariser for categorical distribution
- [X] Variance smoothing for Gaussian distribution
- [X] Vectorised main operations using NumPy

Possible features:

- [ ] Masking in NumPy
- [ ] Support label encoding
- [ ] Support missing data

## References

- [scikit-learn's naive bayes](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes)

## Related Work

(Hang in there)

scikit-learn's categorical naive bayes
scikit-learn's categorical naive bayes
