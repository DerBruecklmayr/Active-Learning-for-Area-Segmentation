from argparse import ArgumentParser
from typing import List, Tuple
from generate_dataset import get_conjunced_subset, get_all_data
from random import randint

from generate_dataset import get_all_data, get_conjunced_subset, CLASSES, TRANSFORMATION
from active_selection import active_sample

from icecream import ic
import numpy as np

import torch
from tqdm import tqdm
from model import CustomModel
from Dataloader import DataSetHandler, DataSetLoader
from torch.utils.data import DataLoader

from statistics import mean

from loss import CustomCrossEntropyLoss

from torchmetrics import JaccardIndex

import pandas as pd

import os
from os.path import abspath, join


def save_learning_curves(acc_train_list:list, IoU_train_list:list, model_train_loss:list, acc_val_list, IoU_val_list, model_val_loss, folder:str):
    df = pd.DataFrame(data={'train_IoU':[float(i) for i in IoU_train_list], 
                            "train_acc":acc_train_list, 
                            "train_loss":[float(i) for i in model_train_loss], 
                            "val_IoU":[float(i) for i in IoU_val_list], 
                            "val_acc":acc_val_list, 
                            "val_loss":model_val_loss})
    df.to_csv(join('results', 'train', folder, "data.csv"))


def save_states(model, optimizer, acc_train_list:list, IoU_train_list:list, model_train_loss:list, acc_val_list, IoU_val_list, model_val_loss, folder:str):
    max_epoch = False
    if IoU_val_list[-1] == max(IoU_val_list) and model_val_loss[-1] < model_val_loss[-1]*3:
        torch.save(model.state_dict(), abspath(join('results', 'train', folder, "model_weights.pt")))
        torch.save(optimizer.state_dict(), abspath(join('results', 'train', folder, "optimizer_states.pt")))
        max_epoch = True

    save_learning_curves(acc_train_list, IoU_train_list, model_train_loss, acc_val_list, IoU_val_list, model_val_loss, folder)
    
    return max_epoch


def test_model(model:torch.nn.Module, dataloader:DataLoader, criterion, IoU, device) -> Tuple[list, int, int]:
    print(f"Validating ...")

    model.eval()

    with torch.no_grad():
        acc = torch.Tensor().to(device)
        IoU_results = []
        loss_results = []

        softmax = torch.nn.Softmax2d().to(device)
        for data, target in tqdm(dataloader):
            data = data.to(device)
            target = target.to(device)

            predictions = model.forward(data)

            accuracy = softmax(predictions)

            loss, adapted_target = criterion(predictions, target, accuracy)

            loss_results.append(loss.detach().cpu().tolist())
            acc = torch.concat((acc, torch.max(accuracy, dim=1).values))

            IoU_results.append((data.shape[0], IoU(predictions, adapted_target).cpu()))
            
    model.train(True)
    
    return torch.mean(acc).detach().cpu().tolist(), sum([num*iou for num, iou in IoU_results])/sum([num for num, iou in IoU_results]), mean(loss_results)


def superviced_training(model:CustomModel, dataloader:DataLoader, train_data_distribution, val_dataloader:DataLoader, epochs:int, max_epoche_no_improvement:int, lr:int, device, folder:str, optimizer=None) -> Tuple[CustomModel, List[float]]:
    os.mkdir(abspath(join('results', 'train', folder)))

    model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = CustomCrossEntropyLoss(train_data_distribution).to(device)
    softmax = torch.nn.Softmax2d().to(device)
    IoU = JaccardIndex("multiclass", num_classes=6).to(device)

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


        print(f"Epoch {e_id}")
        for data, target in tqdm(dataloader):
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



if __name__ == "__main__":
    parser = ArgumentParser(
                    prog='Area ResNet with dynamic loss')
    
    parser.add_argument('-p', '--pretrain-percentage', required=True, type=int)
    parser.add_argument('-e', '--number-of-epochs', required=True, type=int)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--epoch-max-no-imp', type=int, default=100)
    parser.add_argument('-l', '--learning-rate', type=int, default=0.00001)
    parser.add_argument('-seed', '--random-seed', type=int, default=None)
    parser.add_argument('--model-size', type=str, default="34")
    parser.add_argument('-o', '--output-folder', required=True, type=str)
    
    args = parser.parse_args()

    # setup Data
    data = get_all_data()
    include, others = get_conjunced_subset(data['train'], args.pretrain_percentage/100, args.random_seed)
    validate = get_conjunced_subset(data['trainval'], 1, args.random_seed)
    test = get_conjunced_subset(data['val'], 1, args.random_seed)
    handler = DataSetHandler(include.to_numpy(), others.to_numpy(), validate.to_numpy(), test.to_numpy(), CLASSES, (16,16), args.batch_size, True, TRANSFORMATION)

    # model
    model = CustomModel(args.model_size)
    # craete active sample batch
    model, (acc_train_list, IoU_train_list, model_train_loss), (acc_val_list, IoU_val_list, model_val_loss), max_epoch = superviced_training(model, *handler.get_superviced(), handler.get_validate(), args.number_of_epochs, args.epoch_max_no_imp, args.learning_rate, 'cuda', args.output_folder)
    
    print(f"finished Training:\n Best epoch: {max_epoch} (Test -acc: {acc_val_list[max_epoch]} -IoU: {float(IoU_val_list[max_epoch])} -Loss: {model_val_loss[max_epoch]})")
    exit(0)
    

    args = parser.parse_args()

    # setup Dataset
    dataset, unsuperviced_dataset = get_conjunced_subset(get_all_data(), args.pretrain_percentage)


    # train model


