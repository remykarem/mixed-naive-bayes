# ## MixedNB with digits dataset
# ### using categorical naive bayes

# Load the required modules
import numpy as np
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from mixed_naive_bayes import MixedNB, load_example

# Load the digits dataset
digits = load_digits()
X = digits['data']
y = digits['target']

# Fit to `sklearn`'s GaussianNB
gaussian_nb = GaussianNB()
gaussian_nb.fit(X, y)
gaussian_nb_score = gaussian_nb.score(X, y)

# Fit to our classifier
mixed_nb = MixedNB(categorical_features='all')
mixed_nb.fit(X, y)
mixed_nb_score = mixed_nb.score(X, y)

print(gaussian_nb_score)
print(mixed_nb_score)
