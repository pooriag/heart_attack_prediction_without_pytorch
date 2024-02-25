import random
import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import csv_as_dataset

import logistic_regression_model
import csv_data_visualizer as cdv

def plot_losses(losses):
    plt.figure()

    for i in range(0, len(losses), 100):
        plt.plot(i, losses[i], 'ro')
    plt.show()

df = pnd.read_csv('heart.csv')
df['index'] = range(0, len(df))
df.set_index('index', inplace=True)

#cdv.visualize_binary_data_with_respect_to_each_input_features(df, df.columns[len(df.columns) - 1], df.columns[[1]])

cad = csv_as_dataset.csv_dataset("output")

norm_df = cad.normalize_data(df)
train_data = norm_df[(norm_df.index % 3 == 0) | (norm_df.index % 3 == 1)]
test_data = norm_df[(norm_df.index % 3 == 2) & (norm_df.index % 2 == 0)]
evaluation_data = norm_df[(norm_df.index % 3 == 2) & (norm_df.index % 2 == 1)]

M = logistic_regression_model.logistic_model(len(norm_df.columns) - 1)

cad.train(train_data, 50, 100, M)

test_sample = csv_as_dataset.random_sample(test_data, 20)
for i in range(len(test_sample)):
    print(f'actual value{test_sample["output"].iloc[i]}')

    x = test_sample[test_sample.columns[:-1]]
    X = np.array(x.values.tolist()[i]).T
    print(f'prediction{M.forward(X)}')
    print("..............................")



plot_losses(M.losses)