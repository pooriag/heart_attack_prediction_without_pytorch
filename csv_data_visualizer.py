import matplotlib.pyplot as plt

def visualize_binary_data_with_respect_to_each_input_features(df,output, selected_features=None):
    if selected_features == None:
        selected_features = df.columns
        selected_features.remove(output)

    if output in selected_features:
        raise Exception("your output feature is inside selected features")

    for i in range(len(selected_features)):
        plt.figure(i)

        plt.title(selected_features[i])

        for j in df.index:
            print(df[selected_features[i]])
            plt.plot(df[selected_features[i]][j], 'ro')

        plt.show()