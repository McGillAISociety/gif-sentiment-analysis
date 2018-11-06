import pickle
import matplotlib.pyplot as plt

from rick.filepath_settings import GIF_TRAIN_METADATA, GIF_TRAIN_DATA_DIR
from rick.experiments.utilities.image_processing import convert_gif_to_frames


def main():
    # ----------------------------------------
    # Configuration
    # ----------------------------------------

    # Read the meta-data from file.
    meta_data_filepath = GIF_TRAIN_METADATA
    with open(meta_data_filepath, 'rb') as f:
        meta_data = pickle.load(f)

    # ----------------------------------------
    # Count GIF length
    # ----------------------------------------

    frame_counts = []
    for i, row in meta_data.iterrows():
        gif_filepath = str(GIF_TRAIN_DATA_DIR / '{}.gif'.format(row.name))
        frame_counts.append(len(convert_gif_to_frames(gif_filepath)))

    # ----------------------------------------
    # Visualize
    # ----------------------------------------

    n, bins, patches = plt.hist(x=frame_counts, rwidth=1, bins=350)
    plt.xlabel('GIF Frame Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of GIF Frame Counts')
    plt.show()


if __name__ == '__main__':
    main()
