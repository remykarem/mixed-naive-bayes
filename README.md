# Mixed Naive Bayes (WIP)

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

## Performance

Comparing performance with `sklearn`:

## To-Dos

- [X] Vectorised main operations using NumPy
- [ ] Masking in NumPy
