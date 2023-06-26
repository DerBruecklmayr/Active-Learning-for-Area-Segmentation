from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import DataLoader


from sklearn.metrics import pairwise_distances
from Dataloader import DataSetHandler, DataSetLoader

import cv2
import numpy as np
from argparse import ArgumentParser
from typing import List, Tuple
from generate_dataset import get_all_data, get_conjunced_subset, TRANSFORMATION, CLASSES, CLASSES_COLORS

from model import CustomModel

from PIL import Image

from os.path import abspath, join

from Dataloader import DataSetHandler

import torch

from icecream import ic

from classifie_image import expand2square

def get_features(model, loader, device):
    features = []
    model.eval()

    count = 0
    with torch.no_grad():
        for data, target in loader:
            
            data, target = data.to(device), target.to(device)
            output = model.get_features(data)
            # pdb.set_trace()

            count += 1
            # if count > 10000:
            #     break

            features.append(torch.flatten(output, start_dim=1).cpu().numpy())
            # features.append((img_name, output.cpu().numpy()))
    return features


def active_sample(dataHandler:DataSetHandler, sample_size, method='random', model=None, device = "cpu"):
    if method == 'random':
        np.random.shuffle(dataHandler.unsuperviced_dataset)
        sample_rows = dataHandler.unsuperviced_dataset[:sample_size]
        return sample_rows
    

    if method == 'coreset':
        model = model.to(device)
        # Get Data

        # unlabeld Data
        unlab_loader = dataHandler.get_unsuperviced(1, False)

        # labeled Data
        lab_loader, _ = dataHandler.get_superviced(1, False)

        # get labeled features
        labeled_features = get_features(model, lab_loader, device) # (img_name, features)
        # get unlabeled features
        unlabeled_features = get_features(model, unlab_loader, device)# (img_name, features)

        all_features = labeled_features + unlabeled_features
        labeled_indices = np.arange(0,len(labeled_features))

        coreset = Coreset_Greedy(all_features)
        new_batch, max_distance = coreset.sample(labeled_indices, sample_size)
        
        # unlabeled rows start after labeled rows in all_features
        # so offset the indices
        new_batch = [i - len(labeled_features) for i in new_batch]
        
        sample_rows = dataHandler.unsuperviced_dataset[new_batch]

        return sample_rows


class Coreset_Greedy:
    def __init__(self, all_pts):
        self.all_pts = np.array(all_pts)
        self.dset_size = len(all_pts)
        self.min_distances = None
        self.already_selected = []

        # reshape
        feature_len = self.all_pts[0].shape[1]
        self.all_pts = self.all_pts.reshape(-1,feature_len)

        # self.first_time = True

    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]
        
        if centers is not None:
            x = self.all_pts[centers] # pick only centers
            dist = pairwise_distances(self.all_pts, x, metric='euclidean')

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)
    
    def sample(self, already_selected, sample_size):

        # initially updating the distances
        self.update_dist(already_selected, only_new=False, reset_dist=True)
        self.already_selected = already_selected

        # epdb.set_trace()

        new_batch = []
        # pdb.set_trace()
        for _ in range(sample_size):
            if self.already_selected == []:
                ind = np.random.choice(np.arange(self.dset_size))
            else:
                ind = np.argmax(self.min_distances)
            
            assert ind not in already_selected
            self.update_dist([ind],only_new=True, reset_dist=False)
            new_batch.append(ind)
        
        max_distance = max(self.min_distances)
        print("Max distance from cluster : %0.2f" % max_distance)

        return new_batch, max_distance


if __name__ == "__main__":
    model = CustomModel("34")
    model = model.to("cuda")

    print(f"setup Data with classes: {CLASSES} ... ", end='')
    data = get_all_data()
    include, others = get_conjunced_subset(data['train'], 0.5, None)
    validate = get_conjunced_subset(data['trainval'], 1, None)
    test = get_conjunced_subset(data['val'], 1, None)
    print(" -> done")

    handler = DataSetHandler(include.to_numpy(), others.to_numpy(), validate.to_numpy(), test.to_numpy(), CLASSES, (16,16), 16, True, TRANSFORMATION)


    batch = active_sample(handler, 16, "coreset", model, "cuda")

    exit()





