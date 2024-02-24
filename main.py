import random
import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import csv_as_dataset as cad

import logistic_regression_model

def plot_losses(losses):
    plt.figure()

    for i in range(0, len(losses), 100):
        plt.plot(i, losses[i], 'ro')
    plt.show()

df = pnd.read_csv('heart.csv')
df['index'] = range(0, len(df))
df.set_index('index', inplace=True)
norm_df = cad.normalize_data(df)
train_data = norm_df[(norm_df.index % 3 == 0) | (norm_df.index % 3 == 1)]
test_data = norm_df[(norm_df.index % 3 == 2) & (norm_df.index % 2 == 0)]
evaluation_data = norm_df[(norm_df.index % 3 == 2) & (norm_df.index % 2 == 1)]

M = logistic_regression_model.logistic_model(len(norm_df.columns) - 1)

cad.train(train_data, 50, 100, M)

test_sample = cad.random_sample(test_data, 20)
for i in range(len(test_sample)):
    print(f'actual value{test_sample["output"].iloc[i]}')

    x = test_sample[test_sample.columns[:-1]]
    X = np.array(x.values.tolist()[i]).T
    print(f'prediction{M.forward(X)}')
    print("..............................")



plot_losses(M.losses)