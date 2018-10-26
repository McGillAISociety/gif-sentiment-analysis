import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rick.filepath_settings import GIF_TRAIN_METADATA


def main():
    # ----------------------------------------
    # Configuration
    # ----------------------------------------

    # Read the meta-data from file.
    meta_data_filepath = GIF_TRAIN_METADATA
    with open(meta_data_filepath, 'rb') as f:
        meta_data = pickle.load(f)

    # Categories
    categories = ['amusement', 'anger', 'contempt', 'contentment', 'disgust', 'embarrassment', 'excitement', 'fear',
                  'guilt', 'happiness', 'pleasure', 'pride', 'relief', 'sadness', 'satisfaction', 'shame', 'surprise']

    # ----------------------------------------
    # Reformat the data
    # ----------------------------------------

    # Retrieve the current values for "trueskill_rating" as a 17 x N-rows array.
    trueskill = meta_data['trueskill_rating'].tolist()
    trueskill = np.transpose(np.vstack(trueskill))

    # Split the "trueskill_rating" into 17 columns and store them in the data-frame.
    for i, category in enumerate(categories):
        meta_data[category] = trueskill[i]
    meta_data.drop(columns=['trueskill_rating'], inplace=True)

    # ----------------------------------------
    # Visualization
    # ----------------------------------------

    # Calculate and plot the correlation.
    correlation = meta_data.corr()
    sns.heatmap(correlation, square=True, annot=True, fmt='.2f')
    plt.show()

    # Find the top correlations for each category.
    print("Category vs Most Correlated Category")
    np_correlation = np.vstack(correlation.values)
    for i, category in enumerate(categories):
        # Find the indices of the top-N correlated categories.
        top_n_correlated_indices = np_correlation[i].argsort()[-5:][::-1][1:]

        # Create a list of tuples with the format [Index of Category, Category Name, Correlation].
        formatted_information = tuple(zip(top_n_correlated_indices, [categories[j] for j in top_n_correlated_indices],
                                          [np_correlation[i][j] for j in top_n_correlated_indices]))

        # Display.
        print(category, formatted_information)


if __name__ == '__main__':
    main()
