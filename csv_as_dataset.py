import random
import numpy as np
import pandas as pnd

def normalize_data(df):
    mean = df[df.columns[:-1]].mean()
    std = df[df.columns[:-1]].std()
    df[df.columns[:-1]] = (df[df.columns[:-1]] - mean) / std
    return df
def random_sample(df, count):
    return df.iloc[[random.randint(0, len(df) - 1) for i in range(count)]]
def train(df, iteration, count, M):
    for i in range(iteration):
        samp = random_sample(df, count)
        x = samp[samp.columns[:-1]]
        y = samp[samp.columns[len(samp.columns) - 1]]

        X = np.array(x.values.tolist()).T
        Y = np.array(y.values.tolist()).T
        Y = Y.reshape(1, Y.shape[0])

        M.train(X, Y)
