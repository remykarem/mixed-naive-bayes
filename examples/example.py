
from sklearn.datasets import load_iris
# from sklearn.naive_bayes import GaussianNB
from mixed_naive_bayes import MixedNB


data = load_iris()
X = data['data']
y = data['target']
print(len(X))

# clf = GaussianNB()
clf = MixedNB()
clf.fit(X,y)
print(clf.score(X,y))

