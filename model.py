import numpy
import numpy as np


class logistic_model():
    def __init__(self, inp_features):
        self.weights = np.zeros((inp_features, 1), dtype=float)
        self.b = 0.0
    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(Z))
        return A

    def forward(self, X):
        #X is a concated x's horizontally

        Z = np.dot(self.weights.T, X)
        A = self.sigmoid(Z)

        return A
