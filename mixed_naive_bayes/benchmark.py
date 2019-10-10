from sklearn.datasets import load_iris, load_digits, load_wine, \
    load_breast_cancer, fetch_california_housing
from sklearn.naive_bayes import GaussianNB
from mixed_naive_bayes import MixedNB, load_example

for load_data in [load_iris, load_digits, load_wine, 
    load_breast_cancer]:

    dataset = load_data()

    X = dataset['data']
    y = dataset['target']

    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X,y)
    gaussian_nb_pred = gaussian_nb.predict(X)

    mixed_nb = MixedNB()
    mixed_nb.fit(X,y)
    mixed_nb_pred = mixed_nb.predict(X)

    print(gaussian_nb.score(X,y))
    print(mixed_nb.score(X,y))

print("--------------------------------------")

dataset = load_digits()

X = dataset['data']
y = dataset['target']

gaussian_nb = GaussianNB()
gaussian_nb.fit(X[:1400],y[:1400])
# gaussian_nb_pred = gaussian_nb.predict(X)

mixed_nb = MixedNB()
mixed_nb.fit(X[:1400],y[:1400],[0,1,9,17,24,25,32,33,40,41,48,49,57])
# mixed_nb_pred = mixed_nb.predict(X)

print(gaussian_nb.score(X[:1400],y[:1400]))
print(mixed_nb.score(X[:1400],y[:1400]))

print("--------------------------------------")

dataset = fetch_california_housing()

X = dataset['data']
y = dataset['target']

gaussian_nb = GaussianNB()
mixed_nb = MixedNB()

# gaussian_nb.fit(X,y)
mixed_nb.fit(X,y)

print(mixed_nb.score(X,y))
