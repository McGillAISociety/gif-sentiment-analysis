import torch
import cv2
import pickle
import math
import numpy as np

from sklearn.model_selection import KFold
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from rick.external_libraries.python_dynamic_images.dynamicimage import dynamicimage

from rick.filepath_settings import GIF_TRAIN_DATA_DIR, GIF_TRAIN_METADATA
from rick.experiments.utilities.image_processing import convert_gif_to_frames

""" 
Wrapper function used to initialize the data-loaders needed for training and validation.
"""


def get_training_and_validation_dataloaders(n_splits=5, batch_size=32, img_size=(224, 224, 3)):

    # ----------------------------------------
    # Load information from data-frame
    # ----------------------------------------
    # Read the meta-data from file.
    meta_data_filepath = GIF_TRAIN_METADATA
    with open(meta_data_filepath, 'rb') as f:
        data_frame = pickle.load(f)

    # ----------------------------------------
    # Form the data-set and data loaders
    # ----------------------------------------
    # Get the indices which partition the dataset into k folds.
    kfolds = KFold(n_splits=n_splits, random_state=123123123, shuffle=True)

    # Create k pairs of data-sets corresponding to each fold.
    fold_dataloaders = []
    for train_indices, val_indices in kfolds.split(data_frame.index.values):
        # Random samplers.
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # Form the training / validation set accordingly.
        train_dataset = GifDataset(data_frame.copy(), img_size)
        validation_dataset = GifDataset(data_frame.copy(), img_size)

        # Form the training / validation data loaders.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0,
                                  pin_memory=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0,
                                       pin_memory=True)

        # Append to the result.
        fold_dataloaders.append((train_loader, validation_loader))

    return fold_dataloaders


# ----------------------------------------
# Dataset Definition
# ----------------------------------------

class GifDataset(data.Dataset):

    def __init__(self, dataframe, target_img_size):
        self.dataframe = dataframe
        self.target_img_size = target_img_size
        self.loaded_gifs_cache = {}

        self.preprocess_gif = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_img_size[:2]),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, index):
        # Load the processed GIF from cache if possible.
        c_id = self.dataframe.iloc[index].name
        gif = self.loaded_gifs_cache.get(c_id)

        # If the GIF is not found in cache (i.e. first epoch), load and process the GIF.
        if gif is None:
            gif_filepath = str(GIF_TRAIN_DATA_DIR / '{}.gif'.format(c_id))
            frames = convert_gif_to_frames(gif_filepath)
            gif = self.preprocess_frames(frames)
            self.loaded_gifs_cache[c_id] = gif

        # Load TrueSkill rating.
        rating = self.dataframe.iloc[index]['trueskill_rating']

        # Pre-process and convert to tensors.
        gif, rating = self.transform(gif, rating)
        return gif, rating

    def preprocess_frames(self, frames):
        # Calculate dynamic image.
        gif = dynamicimage.get_dynamic_image(frames, normalized=True)

        # Expand the dimensions of greyscale images to be (3, H, W).
        if gif.ndim == 2:
            gif = np.stack((gif,) * 3, axis=0)

        # Resize, normalize and convert to tensor.
        gif = self.preprocess_gif(gif[..., ::-1])

        return gif

    def transform(self, gif, rating):
        # For the baseline, partition the classes into several class clusters based on their correlation.
        max_category = int(np.argmax(rating))

        # Positive = (Amusement, Contentment, Excitement, Happiness, Pleasure, Pride, Relief, Satisfaction)
        if max_category in [0, 3, 6, 9, 10, 11, 12, 14]:
            rating = np.ones(1)
        # Negative = (Anger, Contempt, Disgust, Embarrassment, Fear, Guilt, Sadness, Shame, Surprise)
        else:
            rating = np.zeros(1)
        rating = torch.from_numpy(rating).float()

        return gif, rating


def main():
    # Debug
    datasets = get_training_and_validation_dataloaders()
    for i, (t, v) in enumerate(datasets):
        for j, data in enumerate(t):
            img = data[0][0].numpy()
            img = np.transpose(img, (1, 2, 0))[..., ::-1]
            label = str(int(data[1][0].numpy()))

            cv2.imshow(label, img)
            cv2.waitKey()


if __name__ == '__main__':
    main()
