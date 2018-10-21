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

from rick.filepath_settings import GIF_TRAIN_DATA_DIR, GIF_TRAIN_METADATA

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
        self.gif_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_img_size[:2]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, index):
        # Load the middle frame from the GIF.
        c_id = self.dataframe.iloc[index].name
        gif_filepath = str(GIF_TRAIN_DATA_DIR / '{}.gif'.format(c_id))
        gif = convert_gif_to_frames(gif_filepath)
        gif = gif[math.floor(len(gif) / 2)]

        # Load TrueSkill rating.
        rating = self.dataframe.iloc[index]['trueskill_rating']

        # Pre-process and convert to tensors.
        gif, rating = self.transform(gif, rating)
        return gif, rating

    def transform(self, gif, rating):
        # Expand the dimensions of greyscale images to be (3, H, W).
        if gif.ndim == 2:
            gif = np.stack((gif,)*3, axis=0)

        # Resize, normalize and convert to tensor.
        gif = self.gif_transforms(gif[..., ::-1])

        # For this baseline, let's try to make a classifier which can detect anger, disgust, or contempt
        # which corresponds to categories 1, 2, and 4 respectively.
        max_category = np.argmax(rating)
        if int(max_category) in [1, 2, 4]:
            rating = np.ones(1)
        else:
            rating = np.zeros(1)
        rating = torch.from_numpy(rating).float()

        return gif, rating


# ----------------------------------------
# Helper Functions
# ----------------------------------------

def convert_gif_to_frames(gif_file_path, num_frames_to_read=float('inf')):
    """ Adapted from: https://github.com/asharma327/Read_Gif_OpenCV_Python/blob/master/gif_to_pic.py """

    # Initialize the frame number and create empty frame list
    gif = cv2.VideoCapture(gif_file_path)
    frame_num = 0
    frame_list = []

    # Loop until there are no frames left.
    try:
        while True:
            if len(frame_list) >= num_frames_to_read:
                break
            frames_remaining, frame = gif.read()
            frame_list.append(frame)

            if not frames_remaining:
                break
            frame_num += 1
    finally:
        gif.release()

    return frame_list


def main():
    # Debug
    datasets = get_training_and_validation_dataloaders()
    for i, (t, v) in enumerate(datasets):
        for j, data in enumerate(t):
            img = data[0][0].numpy()
            img = np.transpose(img, (1, 2, 0))[..., ::-1]

            cv2.imshow('', img)
            cv2.waitKey()


if __name__ == '__main__':
    main()
