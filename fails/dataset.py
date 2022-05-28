import logging
import os
import os.path as osp
import random
from typing import Tuple, List, Callable, Optional, Union
from consts import WORKING_DIRECTORY_PATH

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder

class MyDataset(Dataset):
    def __init__(self,
                 samples_path: Union[str, os.PathLike],
                 labels_path: Union[str, os.PathLike],
                 transforms: Optional[List[str]] = None,
                 training=False) -> None:

        working_dir = WORKING_DIRECTORY_PATH
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.input_folder = os.path.join(working_dir, samples_path)
        self.input_paths = list(sorted([pth for pth in os.listdir(self.input_folder)]))

        self.labels = pd.read_csv(os.path.join(working_dir, labels_path))
        self.labels.drop(["Unnamed: 0"], axis=1, inplace=True)

        self.label_shape = self.labels.drop(["image_id", "color", "area"], axis=1)
        self.label_shape = pd.get_dummies(self.label_shape['shape'])

        self.label_color = self.labels.drop(["image_id", "shape", "area"], axis=1)
        self.label_color = pd.get_dummies(self.label_color['color'])

        self.label_area = self.labels.drop(["image_id", "shape", "color"], axis=1)
        self.label_area = pd.get_dummies(self.label_area['area'])

        self.transforms = transforms
        self.training = training

    @staticmethod
    def to_tensor(source: Image.Image,
                  label_shape: np.float64,
                  label_color: np.float64,
                  label_area: np.float64) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Convert a PIL Image to PyTorch tensor. The result is a tensor with values between 0. and 1.
        :param source: A source image containing an X-ray.
        :param label: A label image consisting of a binary mask of either the spine or the vertebrae.
        :return: The tensor equivalent of the image.
        """
        source = tf.to_tensor(source)
        label_shape = torch.from_numpy(label_shape)
        label_color = torch.from_numpy(label_color)
        label_area = torch.from_numpy(label_area)
        return source, label_shape, label_color, label_area

    def __getitem__(self,
                    index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Return the image at the specified index.
        :param index: An integer representing an index from the list of input data.
        :return: The source and the label at the index, taken through transforms.
        """
        source = Image.open(osp.join(self.input_folder, self.input_paths[index]))
        label_shape = np.array([self.label_shape.iloc[index]]).astype('float')
        label_color = np.array([self.label_color.iloc[index]]).astype('float')
        label_area = np.array([self.label_area.iloc[index]]).astype('float')

        #TODO: maybe for future use implement transforms, but it's not necessary atm

        source, label_shape, label_color, label_area = self.to_tensor(source=source,
                                                                      label_shape=label_shape,
                                                                      label_color=label_color,
                                                                      label_area=label_area)

        sample={'image': source, 
                'label_shape': label_shape,
                'label_color': label_color,
                'label_area': label_area}

        return sample

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        :return: The length of the dataset.
        """
        return len(self.input_paths)

        


