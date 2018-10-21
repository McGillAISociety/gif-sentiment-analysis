import pickle
import numpy as np
import pandas as pd

from rick.filepath_settings import DATA_DIR, GIF_METADATA_PATH

""" 
Script used to format the raw data queried from the API.

Ensure that you have the GIF data and meta-data in the correct folders, as specified in the 'filepath_settings.py' file.
"""


def main():
    # ----------------------------------------
    # Configuration
    # ----------------------------------------
    with open(GIF_METADATA_PATH, 'rb') as f:
        meta_data_df = pickle.load(f)

    # ----------------------------------------
    # Split the data
    # ----------------------------------------

    # Reformat the data such that it more readily provides the information required. The processed format will have a
    # four columns: trueskill_mu, trueskill_sigma, download link, and raw_data (indexed by cID), for which there
    # will be a row for each unique GIF. This is done so the data can be further processed while accounting for GIF's
    # that are in multiple categories, but with different statistics within each.
    #
    # The mu and sigma provided for each of the GIF and category pairs corresponds to a TrueSkill rating on the range
    # [0, 50], with a higher score indicating higher probability. As part of preprocessing, the mu and sigma
    # are divided by 50 to normalize onto [0, 1].

    processed_data = {}
    categories = list(meta_data_df.columns)
    for row_index, row in meta_data_df.iterrows():
        # Iterate over each category.
        for column_index, column_name in enumerate(meta_data_df.columns):
            # Parse meta-data.
            gif_data = row[column_name]
            gif_mu = gif_data['parameters']['mu']
            gif_sigma = gif_data['parameters']['sigma']
            gif_cid = gif_data['cID']
            gif_download_link = gif_data['content_data']['embedLink']

            # Set-up the row if doesn't already exist.
            if not processed_data.get(gif_cid):
                processed_data[gif_cid] = {'trueskill_mu': np.zeros(len(categories)),
                                           'trueskill_sigma': np.zeros(len(categories)),
                                           'raw_data': {},
                                           'download_link': ''}

            processed_data[gif_cid]['trueskill_mu'][column_index] = gif_mu / 50
            processed_data[gif_cid]['trueskill_sigma'][column_index] = gif_sigma / 50
            processed_data[gif_cid]['raw_data'][column_name] = gif_data
            if not processed_data[gif_cid]['download_link']:
                processed_data[gif_cid]['download_link'] = gif_download_link

    # ----------------------------------------
    # Save data-frame to file.
    # ----------------------------------------
    processed_data_df = pd.DataFrame.from_dict(processed_data, orient='index')
    filename = DATA_DIR / 'processed_metadata.p'
    with open(filename, 'wb') as f:
        pickle.dump(processed_data_df, f)
    print('Saved processed data to: [{}]'.format(filename))
    print(processed_data_df)


if __name__ == '__main__':
    main()
