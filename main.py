import random
import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt

import model

def normalize_data(df):
    mean = df[df.columns[:-1]].mean()
    std = df[df.columns[:-1]].std()
    df[df.columns[:-1]] = (df[df.columns[:-1]] - mean) / std
    return df
def random_sample(df, count):
    return df.iloc[[random.randint(0, len(df) - 1) for i in range(count)]]
def train(df, iteration, count):
    for i in range(iteration):
        samp = random_sample(train_data, count)
        x = samp[samp.columns[:-1]]
        y = samp[samp.columns[len(samp.columns) - 1]]

        X = np.array(x.values.tolist()).T
        Y = np.array(y.values.tolist()).T
        Y = Y.reshape(1, Y.shape[0])

        M.train(X, Y)

def plot_losses(losses):
    plt.figure()

    for i in range(0, len(losses), 100):
        plt.plot(i, losses[i], 'ro')
    plt.show()

df = pnd.read_csv('heart.csv')
df['index'] = range(0, len(df))
df.set_index('index', inplace=True)
norm_df = normalize_data(df)
train_data = norm_df[(norm_df.index % 3 == 0) | (norm_df.index % 3 == 1)]
test_data = norm_df[(norm_df.index % 3 == 2) & (norm_df.index % 2 == 0)]
evaluation_data = norm_df[(norm_df.index % 3 == 2) & (norm_df.index % 2 == 1)]

M = model.logistic_model(len(norm_df.columns) - 1)

train(train_data, 50, 100)

test_sample = random_sample(test_data, 20)
for i in range(len(test_sample)):
    print(f'actual value{test_sample["output"].iloc[i]}')

    x = test_sample[test_sample.columns[:-1]]
    X = np.array(x.values.tolist()[i]).T
    print(f'prediction{M.forward(X)}')
    print("..............................")



plot_losses(M.losses)