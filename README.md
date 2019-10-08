# Mixed Naive Bayes (WIP)

Implementation of naive bayes for categorical and gaussian 
distributions.

- [Installation](#installation)
- [Usage](#usage)
- [Tests](#tests)
- [API Documentation](#api-documentation)
- [To-Dos](#to-dos)

## Installation

```bash
git clone https://github.com/remykarem/MixedNaiveBayes.git
```

## Usage

```python
from mixed_naive_bayes import MixedNB
clf = MixedNB()
clf.fit(X,y)
clf.predict(X)
clf.score(X,y)
```

## Tests


```bash
pytest tests.py
```

## API Documentation

For more information on usage of the API, visit [here](https://remykarem.github.io/docs/mixed_naive_bayes.html).

## Performance

Comparing performance with `sklearn`:

## To-Dos

- [ ] Variance smoothing for Gaussian distribution
- [ ] Regulariser for categorical distribution
- [X] Vectorised main operations using NumPy

Possible features:

- [ ] Masking in NumPy
- [ ] Support label encoding
- [ ] Support missing data
