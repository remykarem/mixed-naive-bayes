# Mixed Naive Bayes (WIP)

This module implements **Categorical** (Multinoulli) and **Gaussian** Naive Bayes algorithms. These are supervised learning methods based on applying Bayes' theorem with strong (naive) feature independence assumptions.

The motivation for writing this library is that scikit-learn does not have an implementation for categorical naive bayes.

I like `scikit-learn`'s APIs (`.fit()`, `.predict()` 😍) so if you use it a lot, you'll find that it's easy to get started started with this library.

I've written a tutorial [here](https://remykarem.github.io/blog/naive_bayes) for naive bayes if you need to understand a bit more on the math.

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

Specify the indices of the features which are
to follow the categorical distribution. In this dataset, the columns `0` and `1` in X are to be treated as categorical.

```python
from mixed_naive_bayes import MixedNB
clf = MixedNB([0,1])
clf.fit(X,y)
clf.predict(X)
```

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
