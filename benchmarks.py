"""
Run benchmarks on toy datasets provided by sklearn. 
This is to ensure our implementation of Gaussian Naive 
Bayes is the same as sklearn's.
"""

from sklearn.datasets import load_iris, load_digits, \
    load_wine, load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from mixed_naive_bayes import MixedNB

for load_data in [load_iris, load_digits, load_wine,
                  load_breast_cancer]:

    print(f"--- {''.join(load_data.__name__.split('_')[1:])} ---")

    dataset = load_data()

    X = dataset['data']
    y = dataset['target']

    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X, y)
    gaussian_nb_pred = gaussian_nb.predict(X)

    mixed_nb = MixedNB()
    mixed_nb.fit(X, y)
    mixed_nb_pred = mixed_nb.predict(X)

    print(f"GaussianNB: {gaussian_nb.score(X,y)}")
    print(f"MixedNB   : {mixed_nb.score(X,y)}")
