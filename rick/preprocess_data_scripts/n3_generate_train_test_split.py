import pickle

from sklearn.model_selection import train_test_split
from rick.filepath_settings import GIF_PROCESSED_METADATA_PATH

""" 
Script used to split the data-set into training and testing components.
Ensure that you have the GIF data and meta-data in the correct folders, as specified in the 'filepath_settings.py' file.
"""


def get_train_test_split(test_size=0.2, random_state=123123123, processed_metadata_path=GIF_PROCESSED_METADATA_PATH):
    # ----------------------------------------
    # Configuration
    # ----------------------------------------
    with open(processed_metadata_path, 'rb') as f:
        meta_data_df = pickle.load(f)

    # ----------------------------------------
    # Split the data
    # ----------------------------------------

    # Split the data-frame containing meta-data for N categories, into N data-frames. This is done to
    # ensure that each category is handled independently.

    # Finally, split the category into training and testing splits, specified by cID.
    ids_train, ids_test = train_test_split(meta_data_df.index.values,
                                           test_size=test_size,
                                           random_state=random_state)

    return ids_train, ids_test


if __name__ == '__main__':
    _ids_train, _ids_test = get_train_test_split()
    print("Split data into [{}] training samples, and [{}] testing samples.".format(len(_ids_train), len(_ids_test)))
