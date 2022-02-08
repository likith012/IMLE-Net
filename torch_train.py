"""Main script to run the training of the model(ECGNet, Resnet101).
"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


import os
import random
import argparse
import json
from tqdm import tqdm
import numpy as np
import wandb

from preprocessing.preprocess import preprocess
from utils.torch_dataloader import DataGen

# Random seed
seed = 42
random.seed(seed)
np.random.seed(seed)


def train_epoch(model, optimizer, loss_func, dataset, epoch):

    model.train()
    
    pred_all = []
    loss_all = []
    gt_all = []
    
    for batch_step in tqdm(range(len(dataset)) , desc="train"):
        batch_x, batch_y = dataset[batch_step]    
        batch_x = batch_x.cuda()
        batch_x = batch_x.permute(0,2,1)
        batch_y = batch_y.cuda()

        pred = model(batch_x)
        pred_all.append(pred.cpu().detach().numpy())
        print(batch_y.type(), pred.type())
        loss = loss_func(pred, batch_y)
        loss_all.append(loss.cpu().detach().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gt_all.extend(batch_y.cpu().detach().numpy())

    print('epoch {0} '.format(epoch))
    print('train_loss ', np.mean(loss_all))
    pred_all = np.concatenate(pred_all, axis=0)

    _, mean_acc, roc_score = metrics(np.array(gt_all), pred_all )
    wandb.log({'train_mean_accuracy' : mean_acc, 'epoch':epoch})
    wandb.log({'train_roc_score' : roc_score, 'epoch':epoch})
    wandb.log({'train_loss' : np.mean(loss_all) , 'epoch':epoch})

    return np.mean(loss_all)



def test_epoch(model, loss_func, dataset):

    model.eval()
    
    pred_all = []
    loss_all = []
    gt_all = []
    
    for batch_step in tqdm(range(len(dataset)) , desc="test"):
        batch_x, batch_y = dataset[batch_step]
        batch_x = batch_x.cuda()
        batch_x = batch_x.permute(0,2,1)
        batch_y = batch_y.cuda()
        
        pred = model(batch_x)
        pred_all.append(pred.cpu().detach().numpy())
       
        loss = loss_func(pred, batch_y)
        loss_all.append(loss.cpu().detach().numpy())
        gt_all.extend(batch_y.cpu().detach().numpy())

    print('test_loss ', np.mean(loss_all))
    pred_all = np.concatenate(pred_all, axis=0)

    _, mean_acc, roc_score = metrics(np.array(gt_all), pred_all )
    wandb.log({'test_mean_accuracy' : mean_acc, 'epoch':epoch})
    wandb.log({'test_roc_score' : roc_score, 'epoch':epoch})
    wandb.log({'test_loss' : np.mean(loss_all) , 'epoch':epoch})

    return np.mean(loss_all), mean_acc, roc_score




def train(model, path: str = 'data/ptb', batch_size: int = 32, epochs: int = 60, loggr: bool = False, name: str = 'imle_net'):
    """Data preprocessing and training of the model.

    Parameters
    ----------
    model: nn.Module
        Model to be trained.
    path: str, optional
        Path to the directory containing the data. (default: 'data/ptb')
    batch_size: int, optional
        Batch size. (default: 32)
    epochs: int, optional
        Number of epochs. (default: 60)
    loggr: bool, optional
        To log wandb metrics. (default: False)
    name: str, optional
        Name of the model. (default: 'imle_net')
        
    """
    
    X_train_scale, y_train, _, _, X_val_scale, y_val = preprocess(path = path)
    train_gen = DataGen(X_train_scale, y_train, batch_size = batch_size)
    val_gen = DataGen(X_val_scale, y_val, batch_size = batch_size)
    
    
    
if __name__ == '__main__':
    """Main function to run the training of the model.
    """
    
    # Args parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, default = 'data/ptb', help = "Ptb-xl dataset location")
    parser.add_argument("--model", type = str, default = 'ecgnet', help = "Select the model to train. (ecgnet, resnet101)")
    parser.add_argument("--batchsize", type = int, default = 32, help = "Batch size")
    parser.add_argument("--epochs", type = int, default = 60, help = "Number of epochs")
    parser.add_argument("--loggr", type = bool, default = False, help = "Enable wandb logging")
    args = parser.parse_args()
    
    if args.model == 'ecgnet':
        from models.ECGNet import ECGNet
        
        model = ECGNet()
    else:
        from models.resnet101 import resnet101
        
        model = resnet101()
        
    if args.loggr:
        wandb = wandb.init(project='IMLE-Net', name=args.model, notes=f'Model: {args.model} with batch size: {args.batchsize} and epochs: {args.epochs}', save_code=True)
        args.logger = wandb
        
    train(model, path = args.data_dir, batch_size = args.batchsize, epochs = args.epochs, loggr = args.loggr, name = args.model)
