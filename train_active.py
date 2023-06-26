from argparse import ArgumentParser
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

from tqdm import tqdm

from model import CustomModel

from model import CustomModel
from torch.utils.data import DataLoader
from Dataloader import DataSetHandler, DataSetLoader
from semi_superviced import semi_superviced_training


def gen_data(datasethandler:DataSetHandler, batch_size:int, model, device:str) -> DataSetHandler:
    # craete active sample batch
    batch = active_sample(datasethandler, batch_size, "coreset", model, device)
    datasethandler.update_dataset(batch)
    return datasethandler

def active_loop(model, train_loader, train_dist, val_loader, epochs, epoch_max_no_imp,  run_folder, optimizer):
    # craete active sample batch
    model, (acc_train_list, IoU_train_list, model_train_loss), (acc_val_list, IoU_val_list, model_val_loss), max_epoch = superviced_training(model, train_loader, train_dist, val_loader, epochs, epoch_max_no_imp, 0, 'cuda', run_folder, optimizer)
    
    return (acc_train_list[:max_epoch+1], IoU_train_list[:max_epoch+1], model_train_loss[:max_epoch+1]), (acc_val_list[:max_epoch+1], IoU_val_list[:max_epoch+1], model_val_loss[:max_epoch+1]), max_epoch


if __name__ == "__main__":
    
    parser = ArgumentParser(
                    prog='Train Active Learning Area ResNet with dynamic loss',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('-e', '--pre-trained-epochs', required=True, type=int, help="epochs for pre-training the model")
    parser.add_argument('--superviced-epochs', required=True, type=int, help="epochs per superviced training intervall")
    parser.add_argument('--epoch-max-no-imp', type=int, default=30)
    parser.add_argument('-l', '--learning-rate', type=int, default=0.00001)

    parser.add_argument('--semi-superviced-lerning', type=int, default=None, help="epochs of semi-superviced-training with the not used training-data")

    parser.add_argument('-b', '--batch-size', required=True, type=int)

    parser.add_argument('-ptb', '--pretrain-budget', required=True, type=int)
    parser.add_argument('-active-budget', '--active-budget', required=True, type=int)
    parser.add_argument('-active-steps', '--active-steps', required=True, type=int)
    parser.add_argument('--random-seed', default=None, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model-size', default='34', type=str)

    parser.add_argument('-o', '--output-folder', required=True, type=str)

    args = parser.parse_args()

    # setup Data
    print(f"setup Data with classes: {CLASSES} ... ", end='')
    data = get_all_data()
    include, others = get_conjunced_subset(data['train'], args.pretrain_budget/100, args.random_seed)
    validate = get_conjunced_subset(data['trainval'], 1, args.random_seed)
    test = get_conjunced_subset(data['val'], 1, args.random_seed)
    print(" -> done")

    handler = DataSetHandler(include.to_numpy(), others.to_numpy(), validate.to_numpy(), test.to_numpy(), CLASSES, (16,16), args.batch_size, True, TRANSFORMATION)

    # model
    model = CustomModel(args.model_size)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    print("\n-----------------------------------------------------------------\nPretraining Model")
    mkdir(abspath(join('results', 'train', args.output_folder)))

    total_acc_train_list, total_IoU_train_list, total_model_train_loss, total_acc_val_list, total_IoU_val_list, total_model_val_loss = [], [], [], [], [], []

    last_model_dir = join(args.output_folder, "pretrain")
    (acc_train_list, IoU_train_list, model_train_loss), (acc_val_list, IoU_val_list, model_val_loss), max_epoch =  active_loop(model, *handler.get_superviced(), handler.get_validate(), args.pre_trained_epochs, args.epoch_max_no_imp, last_model_dir, optimizer)
   
    total_acc_train_list.extend(acc_train_list)
    total_IoU_train_list.extend(IoU_train_list)
    total_model_train_loss.extend(model_train_loss)
    total_acc_val_list.extend(acc_val_list)
    total_IoU_val_list.extend(IoU_val_list)
    total_model_val_loss.extend(model_val_loss)
    
    last_max_epoch = max_epoch 

    active_batch_size = int(((len(include.index) + len(others.index)) * args.active_budget / 100) / args.active_steps) if args.active_steps > 0 else 0
    
    for i in range(args.active_steps):        
        print("Starting active Learning ... ", end="")
        model.load_state_dict(torch.load(abspath(join('results', 'train', last_model_dir, "model_weights.pt"))))
        optimizer.load_state_dict(torch.load(abspath(join('results', 'train', last_model_dir, "optimizer_states.pt"))))

        last_model_dir = join(args.output_folder, f"run{i+1}")

        # craete active sample batch
        batch = active_sample(handler, active_batch_size, "coreset", model, args.device)
        handler.update_dataset(batch)
        print(f'  found batch of size {active_batch_size}')
        (acc_train_list, IoU_train_list, model_train_loss), (acc_val_list, IoU_val_list, model_val_loss), max_epoch =  active_loop(model, *handler.get_superviced(), handler.get_validate(), args.superviced_epochs, args.epoch_max_no_imp, last_model_dir, optimizer)
    
        total_acc_train_list.extend(acc_train_list)
        total_IoU_train_list.extend(IoU_train_list)
        total_model_train_loss.extend(model_train_loss)
        total_acc_val_list.extend(acc_val_list)
        total_IoU_val_list.extend(IoU_val_list)
        total_model_val_loss.extend(model_val_loss)
        save_learning_curves(total_acc_train_list, total_IoU_train_list, total_model_train_loss, total_acc_val_list, total_IoU_val_list, total_model_val_loss, args.output_folder)
    
        last_max_epoch += max_epoch + 1 # +1 because epochs start at 0 but list is already contain elements -> shift by one 

    save_learning_curves(total_acc_train_list, total_IoU_train_list, total_model_train_loss, total_acc_val_list, total_IoU_val_list, total_model_val_loss, args.output_folder)
    
    print(f"finished active Training:\n Best epoch: {last_max_epoch} (Test -acc: {total_acc_val_list[last_max_epoch]} -IoU: {float(total_IoU_val_list[last_max_epoch])} -Loss: {total_model_val_loss[last_max_epoch]})")
   
    model.load_state_dict(torch.load(abspath(join('results', 'train', last_model_dir, "model_weights.pt"))))
    optimizer.load_state_dict(torch.load(abspath(join('results', 'train', last_model_dir, "optimizer_states.pt"))))

    semi_superviced_training(model, optimizer, handler, args.semi_superviced_lerning, args.device, args.batch_size, args.epoch_max_no_imp, join(args.output_folder, f"semi-superviced"))

     
    exit(0)