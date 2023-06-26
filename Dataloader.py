
from typing import List, Tuple
from torch.utils.data import Dataset
import torch
from icecream import ic

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from preprocess import get_image_square_with_bounding_boxes, convert_objects_to_np_array_with_background_full

class SemiSupervicedLoader(Dataset):
    
    CLASSES:List[str]
    superviced_dataset:np.ndarray
    unsuperviced_dataset:np.ndarray
    output_shape:Tuple[int,int]

    def __init__(self, classes:List[str], superviced_dataset:np.ndarray, semi_superviced_dataset:np.ndarray, semi_superviced_predictions:np.ndarray, output_shape:Tuple[int,int], transform=None):
        self.output_shape = output_shape
        self.CLASSES = classes
        self.superviced_dataset = superviced_dataset
        self.semi_superviced_dataset = semi_superviced_dataset
        self.semi_superviced_predictions = semi_superviced_predictions # Shape(num_images, size_box, size_box)

        self.transform = transform


    def get_distribution(self):
        dist = np.empty((6,)) # [0 for i in self.CLASSES]

        for i in range(self.superviced_dataset.shape[0]):
            file = str(self.superviced_dataset[i]).zfill(6)

            img, objects = get_image_square_with_bounding_boxes(file, self.CLASSES)
            img_classes = convert_objects_to_np_array_with_background_full(img.shape, objects, self.CLASSES, self.output_shape)

            dist = np.vstack((dist, np.sum(np.sum(img_classes, axis=1), axis = 1)))

        ic(self.semi_superviced_predictions.shape, dist.shape)
        dist = np.vstack((dist, np.array([np.sum(self.semi_superviced_predictions == i) for i in range(len(self.CLASSES) +1)])))
        
        return torch.from_numpy(np.sum(dist , axis=0))


    def __len__(self):
        return self.superviced_dataset.size + self.semi_superviced_dataset.size
    

    def __getitem__(self, idx):
        img = None
        if idx >= self.superviced_dataset.size: # use semi-superviced-data
            file = str(self.semi_superviced_dataset[idx - self.superviced_dataset.size]).zfill(6)

            img_classes = np.zeros((len(self.CLASSES)+1, *self.output_shape))
            # ic(img_classes.shape, img_classes[0], (self.semi_superviced_predictions[idx - self.superviced_dataset.size] == 0).shape)
            for i in range(len(self.CLASSES) + 1):
                img_classes[i] += (self.semi_superviced_predictions[idx - self.superviced_dataset.size] == i)

            img_classes = img_classes.astype(bool).astype(int)
            img, _ = get_image_square_with_bounding_boxes(file, self.CLASSES)

        else: # use annotated-data
            file = str(self.superviced_dataset[idx]).zfill(6)
            img, objects = get_image_square_with_bounding_boxes(file, self.CLASSES)
            img_classes = convert_objects_to_np_array_with_background_full(img.shape, objects, self.CLASSES, self.output_shape)
            np.zeros
        
        if self.transform:
            img = self.transform(img)

        return img, torch.from_numpy(img_classes)


class DataSetLoader(Dataset):

    CLASSES:List[str]
    dataset:np.ndarray
    output_shape:Tuple[int,int]

    def __init__(self, classes:List[str], dataset:np.ndarray, output_shape:Tuple[int,int], transform=None):
        self.output_shape = output_shape
        self.CLASSES = classes
        self.dataset = dataset

        self.transform = transform

    def get_distribution(self):
        dist = np.empty((6,)) # [0 for i in self.CLASSES]

        for i in range(self.dataset.shape[0]):
            file = str(self.dataset[i]).zfill(6)

            img, objects = get_image_square_with_bounding_boxes(file, self.CLASSES)
            img_classes = convert_objects_to_np_array_with_background_full(img.shape, objects, self.CLASSES, self.output_shape)

            dist = np.vstack((dist, np.sum(np.sum(img_classes, axis=1), axis = 1)))
        
        return torch.from_numpy(np.sum(dist , axis=0))

    def __len__(self):
        return self.dataset.size
    
    def __getitem__(self, idx):
        file = str(self.dataset[idx]).zfill(6)

        img, objects = get_image_square_with_bounding_boxes(file, self.CLASSES)
        img_classes = convert_objects_to_np_array_with_background_full(img.shape, objects, self.CLASSES, self.output_shape)
        
        if self.transform:
            img = self.transform(img)

        return img, torch.from_numpy(img_classes)


class DataSetHandler:
    CLASSES:List[str]
    superviced_dataset:np.ndarray
    unsuperviced_dataset:np.ndarray
    output_shape:Tuple[int,int]


    def __init__(self, superviced_df:np.ndarray, unsuperviced_df:np.ndarray, validate:np.ndarray, test:np.ndarray, classes:List[str], output_shape:Tuple[int,int], batch_size:int=8, shuffle:bool=True, transform=None):
        self.output_shape = output_shape
        self.CLASSES = classes
        self.superviced_dataset = np.unique(superviced_df)
        self.unsuperviced_dataset = np.unique(unsuperviced_df)
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validate = np.unique(validate)
        self.test = np.unique(test)


    def update_dataset(self, elements:list):
        self.superviced_dataset = np.append(self.superviced_dataset, elements)
        self.unsuperviced_dataset = np.setdiff1d(self.unsuperviced_dataset, elements)
    

    def get_superviced(self, batch_size=None, shuffle=None) -> DataLoader:
        dsL = DataSetLoader(self.CLASSES, self.superviced_dataset, self.output_shape, self.transform)
        return DataLoader(dsL, batch_size=self.batch_size if batch_size is None else batch_size, shuffle=self.shuffle if shuffle is None else shuffle), dsL.get_distribution()
    

    def get_unsuperviced(self, batch_size=None, shuffle=None) -> DataLoader:
        return DataLoader(DataSetLoader(self.CLASSES, self.unsuperviced_dataset, self.output_shape, self.transform), batch_size=self.batch_size if batch_size is None else batch_size, shuffle=self.shuffle if shuffle is None else shuffle)

    def get_semi_superviced(self, semi_superviced_predictions:np.ndarray) -> DataLoader:
        ssL = SemiSupervicedLoader(self.CLASSES, self.superviced_dataset, self.unsuperviced_dataset, semi_superviced_predictions, self.output_shape, self.transform)
        return DataLoader(ssL, batch_size=self.batch_size, shuffle=self.shuffle), ssL.get_distribution()

    def get_all(self) -> DataLoader:
        return DataLoader(DataSetLoader(self.CLASSES, np.concatenate([self.superviced_dataset, self.unsuperviced_dataset]), self.output_shape, self.transform), batch_size=self.batch_size, shuffle=self.shuffle)
    

    def get_validate(self) -> DataLoader:
        return DataLoader(DataSetLoader(self.CLASSES, self.validate, self.output_shape, self.transform), batch_size=self.batch_size, shuffle=self.shuffle)
    
    def get_test(self) -> DataLoader:
        return DataLoader(DataSetLoader(self.CLASSES, self.test, self.output_shape, self.transform), batch_size=self.batch_size, shuffle=self.shuffle)
    




