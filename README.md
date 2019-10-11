# Mixed Naive Bayes (WIP)

Implementation of naive bayes for categorical and gaussian 
distributions.

- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [Tests](#tests)
- [API Documentation](#api-documentation)
- [To-Dos](#to-dos)

## Installation

```bash
pip install git+https://github.com/remykarem/MixedNaiveBayes#egg=mixed_naive_bayes
```

## Usage

```python
from mixed_naive_bayes import MixedNB
clf = MixedNB()
clf.fit(X,y)
clf.predict(X)
clf.score(X,y)
```

## Performance

Accuracy on the training set.

Dataset | GaussianNB | MixedNB (G) | MixedNB (C) | MixedNB (G+C) |
------- | ---------- | ----------- | ----------- | ------------- |
Iris    | 0.960      | 0.960       | -           | - |
Digits  | 0.858      | 0.858       | **0.961**   | - |
Wine    | 0.989      | 0.989       | -         | - |
Cancer  | 0.942      | 0.942       | -         | - |
covtype | 0.616      | 0.616       |            | **0.657** |


## Tests

Unit testing:

```bash
pytest tests.py
```

Run benchmarks against `sklearn`'s APIs:

```bash
python benchmarks.py
```

## API Documentation

For more information on usage of the API, visit [here](https://remykarem.github.io/docs/mixed_naive_bayes.html).

## Performance

Comparing performance with `sklearn`:

## To-Dos

- [X] Support refitting
- [X] Regulariser for categorical distribution
- [X] Variance smoothing for Gaussian distribution
- [X] Vectorised main operations using NumPy

Possible features:

- [ ] Masking in NumPy
- [ ] Support label encoding
- [ ] Support missing data
