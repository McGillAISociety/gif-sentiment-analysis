import os
import pickle
import wget
import pathlib
import logging
import time

""" 
Script used to download GIFs from the GIFGIF media lab. To use the script, first run the previous script
to extract the meta-data to file first (which contains the download links, etc.).
"""


def main():
    # ----------------------------------------
    # Configuration
    # ----------------------------------------

    # Each row contains one GIF from each category.
    number_of_rows_to_download = 500
    number_of_tries_per_file = 3
    wait_time_between_tries = 1

    # Read the meta-data from file.
    meta_data_filepath = 'gif_metadata.p'
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

    # Download destination.
    download_folder = pathlib.Path('./gif_data/')
    if os.path.exists(download_folder):
        raise FileExistsError("The download destination folder already exists!")
    else:
        os.makedirs(download_folder)

    # Create the download folders.
    for column_name in meta_data.columns:
        category_folder = download_folder / column_name
        os.makedirs(category_folder)

    # ----------------------------------------
    # Download the GIF's
    # ----------------------------------------

    print("Starting to download files, output will be logged to {}.log".format(file_name))
    for index, row in meta_data.iterrows():
        if index >= number_of_rows_to_download:
            logger.info("Finished downloading {} rows.".format(number_of_rows_to_download))
            break

        # Iterate over each category.
        for column_name in meta_data.columns:
            for _ in range(number_of_tries_per_file):
                try:
                    # Parse meta-data.
                    category_folder = download_folder / column_name
                    gif_data = row[column_name]
                    gif_content_data = gif_data['content_data']
                    gif_download_link = gif_content_data['embedLink']
                    gif_rank = gif_data['rank']
                    gif_cid = gif_data['cID']
                    gif_index = gif_data['index']

                    download_filepath = category_folder / '{rank}_{index}_{cID}.gif'.format(rank=gif_rank,
                                                                                            index=gif_index,
                                                                                            cID=gif_cid)
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
