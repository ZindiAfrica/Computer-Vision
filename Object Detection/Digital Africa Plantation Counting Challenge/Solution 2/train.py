import torch
import torch.nn as nn
import cv2
import numpy as np
import random
import os
import argparse
import yaml
from argparse import ArgumentParser
from loguru import logger
from fold import create_folds
from trainer import *
from dataset import PlantationDataset
from model import PlantationModel
import torch.optim as optim

from augment import *
import pandas as pd
import time
from torch.utils.data import Dataset, DataLoader
import copy
from torch.optim import lr_scheduler
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, required=False, default="configs/exp.yaml"
)
args = parser.parse_args()

logger.info("Loading config")
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(config['seed'])

def get_train_file_path(id):
    return f"{config['train_dir']}/{id}"


df = pd.read_csv(f"{config['root_dir']}/Train.csv")

drop_label = ['Id_lh8b1k1lx8.png', 'Id_u45dpub99b.png', 'Id_lp4yl8q9n2.png', 'Id_6ls94ewz47.png', 'Id_lvua92vvg5.png', 'Id_w4fnd54go8.png']
df = df[~df['ImageId'].isin(drop_label)].reset_index(drop=True)
df['file_path'] = df['ImageId'].apply(get_train_file_path)
df = create_folds(df, n_s=config['n_fold'], n_grp=14)
print(df)

def fetch_scheduler(optimizer):
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['T_max'], 
                                                   eta_min=config['min_lr'])
    elif config['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=config['T_0'], 
                                                             eta_min=config['min_lr'])
    elif config['scheduler'] == None:
        return None
        
    return scheduler

def run_training(fold, model, optimizer, scheduler, device, num_epochs):

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_transforms = data_transforms(phase='train', img_size=config['img_size'])
    valid_transforms = data_transforms(phase='valid', img_size=config['img_size'])
    

    train_dataset = PlantationDataset(config['train_dir'], df_train, transforms=train_transforms)
    valid_dataset = PlantationDataset(config['train_dir'], df_valid, transforms=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], 
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['valid_batch_size'], 
                              num_workers=4, shuffle=False, pin_memory=True)
    
    
    if torch.cuda.is_available():
        logger.info("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_rmse = np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=device, epoch=epoch, config=config)
        
        val_epoch_loss, val_epoch_rmse = valid_one_epoch(model, valid_loader, optimizer,
                                                         device=device, 
                                                         epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Valid RMSE'].append(val_epoch_rmse)
        
        logger.info(f'Valid RMSE: {val_epoch_rmse}')
        
        if val_epoch_rmse <= best_epoch_rmse:
            logger.info(f"====> Validation Loss Improved ({best_epoch_rmse} ---> {val_epoch_rmse})")
            best_epoch_rmse = val_epoch_rmse
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"{config['checkpoint_dir']}/best_RMSE_fold{fold}.pt"
            torch.save(model.state_dict(), PATH)
            logger.info(f"Model Saved")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    logger.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    logger.info("Best RMSE: {:.4f}".format(best_epoch_rmse))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(10 * '*'  + " Start training " + 10 * '*')
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    log_name = f"{config['log_dir']}/{config['backbone']}_imgsz_{config['img_size']}_bs_{config['train_batch_size']}_eps_{config['epochs']}.log"
    if os.path.exists(f"{config['log_dir']}/{log_name}"):
        os.remove(f"{config['log_dir']}/{log_name}")
    logger.add(
        f"{config['log_dir']}/{config['backbone']}_imgsz_{config['img_size']}_bs_{config['train_batch_size']}_eps_{config['epochs']}.log"
    )
    for fold in range(config['n_fold']):
        logger.info(f'Fold {fold} start training')
        model = PlantationModel(config['backbone'])
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = fetch_scheduler(optimizer)
        model, history = run_training(fold, model, optimizer, scheduler,
                                  device=device,
                                  num_epochs=config['epochs'])