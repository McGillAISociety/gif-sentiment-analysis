import os
import pickle
import wget
import logging
import time

from rick.preprocess_data_scripts.n3_generate_train_test_split import get_train_test_split
from rick.filepath_settings import GIF_PROCESSED_METADATA_PATH, GIF_TRAIN_METADATA

""" 
Script used to generate a data-frame only containing the training data.
"""


def main():
    # ----------------------------------------
    # Configuration
    # ----------------------------------------

    # Read the meta-data from file.
    meta_data_filepath = GIF_PROCESSED_METADATA_PATH
    with open(meta_data_filepath, 'rb') as f:
        meta_data = pickle.load(f)

    # ----------------------------------------
    # Drop unnecessary data from the data-frame
    # ----------------------------------------

    # Train / Test Split
    ids_train, ids_test = get_train_test_split(test_size=0.2, random_state=123123123,
                                               processed_metadata_path=GIF_PROCESSED_METADATA_PATH)
    ids_train, ids_test = list(ids_train), list(ids_test)

    # Drop unnecessary columns
    meta_data.drop(columns=['trueskill_mu', 'trueskill_sigma', 'raw_data', 'download_link'], inplace=True)
    # Drop test rows
    meta_data.drop(ids_test, inplace=True)

    # Save to file
    filename = GIF_TRAIN_METADATA
    with open(filename, 'wb') as f:
        pickle.dump(meta_data, f)
    print('Saved processed data to: [{}]'.format(filename))
    print(meta_data)


if __name__ == '__main__':
    main()
