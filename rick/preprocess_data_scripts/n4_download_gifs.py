import os
import pickle
import wget
import logging
import time

from rick.preprocess_data_scripts.n3_generate_train_test_split import get_train_test_split
from rick.filepath_settings import GIF_PROCESSED_METADATA_PATH, GIF_TRAIN_DATA_DIR, GIF_TEST_DATA_DIR
""" 
Script used to download GIFs from the GIFGIF media lab. To use the script, first run the previous script
to extract the meta-data to file first (which contains the download links, etc.).
"""


def main():
    # ----------------------------------------
    # Configuration
    # ----------------------------------------

    # Each row contains one GIF from each category.
    number_of_rows_to_download = float('inf')
    number_of_tries_per_file = 3
    wait_time_between_tries = 1

    # Read the meta-data from file.
    meta_data_filepath = GIF_PROCESSED_METADATA_PATH
    with open(meta_data_filepath, 'rb') as f:
        meta_data = pickle.load(f)

    # Set-up the logger.
    file_name = ''.join(os.path.basename(__file__).split('.py')[:-1])
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('{}.log'.format(file_name))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # ----------------------------------------
    # Set-up for downloading
    # ----------------------------------------

    # Ensure the download locations don't already exist to avoid over-writing or merging downloaded data.
    train_download_folder = GIF_TRAIN_DATA_DIR
    if os.path.exists(train_download_folder):
        raise FileExistsError("The train download destination folder already exists!")
    else:
        os.makedirs(train_download_folder)

    test_download_folder = GIF_TEST_DATA_DIR
    if os.path.exists(test_download_folder):
        raise FileExistsError("The test download destination folder already exists!")
    else:
        os.makedirs(test_download_folder)

    # ----------------------------------------
    # Get the ID's for test/train split
    # ----------------------------------------
    ids_train, ids_test = get_train_test_split(test_size=0.2, random_state=123123123,
                                               processed_metadata_path=GIF_PROCESSED_METADATA_PATH)
    ids_train, ids_test = list(ids_train), list(ids_test)

    # ----------------------------------------
    # Download the GIF's
    # ----------------------------------------

    print("Starting to download files, output will be logged to {}.log".format(file_name))
    for i, (c_id, row) in enumerate(meta_data.iterrows()):
        if i >= number_of_rows_to_download:
            logger.info("Finished downloading {} rows.".format(number_of_rows_to_download))
            break

        # Download each file.
        for _ in range(number_of_tries_per_file):
            try:
                # Parse meta-data.
                if c_id in ids_train:
                    download_folder = train_download_folder
                elif c_id in ids_test:
                    download_folder = test_download_folder
                else:
                    raise ValueError

                gif_download_link = row['download_link']
                download_filepath = download_folder / '{cID}.gif'.format(cID=c_id)

                # Check if file already exists.
                if os.path.exists(download_filepath):
                    break

                # Download.
                logger.debug('Downloading [{}] from [{}]'.format(download_filepath, gif_download_link))
                wget.download(gif_download_link, out=str(download_filepath))
                break

            except Exception as e:
                logger.exception(e)
                time.sleep(wait_time_between_tries)


if __name__ == '__main__':
    main()
