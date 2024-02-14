import numpy as np

import model

M = model.logistic_model(12)

X_1 = np.arange(12)
X_1 = X_1.reshape(12, 1)

X_2 = np.arange(12) + 3
X_2 = X_2.reshape(12, 1)

X = np.concatenate([X_1, X_2], axis=1)
#print(X)

print(M.forward(X))