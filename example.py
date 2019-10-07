# from mixed_naive_bayes import MixedNB, load_example

# X, y = load_example()
# clf = MixedNB(alpha=1, class_prior=[.9,.1])
# clf.fit(X, y, categorical_features=[0, 1])
# print(clf.score(X,y))

import time

from sklearn.datasets import load_digits
# from sklearn.naive_bayes import GaussianNB
from mixed_naive_bayes import MixedNB


data = load_digits()
X = data.data
y = data.target
print(len(X))

# clf = GaussianNB()
clf = MixedNB()
clf.fit(X,y, list(range(16)))
print(clf.score(X,y))
