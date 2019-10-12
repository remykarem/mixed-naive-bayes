# ## Mixed Naive Bayes with covertype dataset
# ### using a combination of categorical and gaussian naive bayes

# Load the required modules
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.naive_bayes import GaussianNB
from mixed_naive_bayes import MixedNB

# Fetch the dataset
dataset = fetch_covtype()
X_raw = dataset['data']
y_raw = dataset['target']

# Grab the quantitative columns
quant = X_raw[:,:10]

# Take the next 4 columns
soil = np.argwhere(X_raw[:,-44:-40]==1)[:,1]
soil = soil[:,np.newaxis]

# Take last 40 columns
wild = np.argwhere(X_raw[:,-40:]==1)[:,1]
wild = wild[:,np.newaxis]

# Concat X's and minus 1 from y
# to make categories start from 0
X = np.hstack([quant, soil, wild])
y = y_raw-1

del X_raw, y_raw
del quant, soil, wild

# Assume features `10` and `11` are multinoulli (categorical) 
# and the rest Gaussian
clf = MixedNB([10,11])
clf.fit(X,y)
print(f"MixedNB (C+G): {clf.score(X,y)}")

# Assume all features are Gaussian (using our library)
clf = MixedNB()
clf.fit(X,y)
print(f"MixedNB (G)  : {clf.score(X,y)}")

# Assume all features are Gaussian (using `sklearn`'s library)
clf = GaussianNB()
clf.fit(X,y)
print(f"GaussianNB   : {clf.score(X,y)}")
