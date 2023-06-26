from argparse import ArgumentParser

from torchmetrics import JaccardIndex
from loss import CustomCrossEntropyLoss
from Dataloader import SemiSupervicedLoader
from train_superviced import save_learning_curves, superviced_training
from generate_dataset import get_conjunced_subset, get_all_data
from random import randint

from generate_dataset import get_all_data, get_conjunced_subset, CLASSES, TRANSFORMATION
from active_selection import active_sample

from icecream import ic
import numpy as np
import torch
from os.path import abspath, join
from os import mkdir
import os

from statistics import mean

from tqdm import tqdm


from model import CustomModel
from torch.utils.data import DataLoader
from Dataloader import DataSetHandler, DataSetLoader
from train_superviced import test_model, save_states, save_learning_curves

def gen_semi_superviced_predictions(model, dataloader:DataLoader, batch_size:int, device):
    predictions = torch.Tensor().to(device)
    softmax = torch.nn.Softmax2d().to(device)
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data = data.to(device)
            output = torch.argmax(softmax(model(data)), dim=1)
            predictions = torch.concat((predictions, output))

    return predictions

def semi_superviced_training(model, optimizer, handler:DataSetHandler, epochs:int, device:str, batch_size:int, max_epoche_no_improvement:int, folder:str):
    os.mkdir(abspath(join('results', 'train', folder)))

    
    print("Starting Semi-superviced learning")

    # generate data-loader
    val_dataloader = handler.get_validate()

    # training objects
    max_epoch = -1

    model = model.to(device)

    softmax = torch.nn.Softmax2d().to(device)
    IoU = JaccardIndex("multiclass", num_classes=6).to(device)

    # store training-states
    acc_train_list = []
    IoU_train_list = []
    model_train_loss = []

    acc_val_list = []
    IoU_val_list = []
    model_val_loss = []

    max_epoch = 0

    for e_id in range(epochs):
        epoch_train_acc = torch.Tensor().to(device)
        epoch_train_IoU = []
        epoch_train_loss = []

        print("predict annotations for not annotated images:")
        semi_predictions = gen_semi_superviced_predictions(model, handler.get_unsuperviced(), batch_size, device).cpu().numpy()
        semi_superviced_loader, data_distribution = handler.get_semi_superviced(semi_predictions)

        criterion = CustomCrossEntropyLoss(data_distribution).to(device) # seting loss after each epoch, because data_distribution can change!

        print(f"Epoch {e_id}")
        for data, target in tqdm(semi_superviced_loader):
            data = data.to(device)
            target = target.to(device)

            predictions = model.forward(data)

            accuracy = softmax(predictions)
            loss, adapted_target = criterion(predictions, target, accuracy)

            epoch_train_loss.append(loss.detach().cpu().tolist())
            epoch_train_acc = torch.concat((epoch_train_acc, torch.max(accuracy, dim=1).values))

            epoch_train_IoU.append((data.shape[0], IoU(predictions, adapted_target).cpu()))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        acc_train_list.append(torch.mean(epoch_train_acc).detach().cpu().tolist())
        IoU_train_list.append(sum([num*iou for num, iou in epoch_train_IoU])/sum([num for num, iou in epoch_train_IoU]))
        model_train_loss.append(float(mean(epoch_train_loss)))

        # validate model
        acc_val_results, IoU_val_results, model_val_loss_results = test_model(model, val_dataloader, criterion, IoU, device)
        acc_val_list.append(acc_val_results)
        IoU_val_list.append(IoU_val_results)
        model_val_loss.append(model_val_loss_results)

        print(f'Model acc: {acc_val_results} IoU: {IoU_val_results} loss: {model_val_loss_results}')

        if save_states(model, optimizer, acc_train_list, IoU_train_list, model_train_loss, acc_val_list, IoU_val_list, model_val_loss, folder):
            max_epoch = e_id

        if len(IoU_train_list) - IoU_train_list.index(max(IoU_train_list)) >= max_epoche_no_improvement:
            print(f'no updates in the last {max_epoche_no_improvement} epochs -> stop training')
            return model, (acc_train_list, IoU_train_list, model_train_loss), (acc_val_list, IoU_val_list, model_val_loss), max_epoch

    return model, (acc_train_list, IoU_train_list, model_train_loss), (acc_val_list, IoU_val_list, model_val_loss), max_epoch


