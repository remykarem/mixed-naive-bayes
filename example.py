from mixed_naive_bayes import MixedNB, load_example

X, y = load_example()
clf = MixedNB(alpha=1, class_prior=[.9,.1])
clf.fit(X, y, categorical_features=[0, 1])
print(clf.score(X,y))
# print(clf.predict_proba([[0, 0], [1, 1]]))
