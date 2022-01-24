import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models
import pandas as pd
#import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.model_selection import StratifiedKFold
import datetime
from PIL import Image
import torch.nn.functional as F
from dataset import ICLRDataset
from utils import test, train_model_snapshot
from config import *
import argparse

parser = argparse.ArgumentParser(description='single kfold training script')
parser.add_argument('-cid','--config_id', help='configuration id (1, 2, 3, or 4)', required=True, type=int)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#read train and test data
train_imgs = np.load('unique_train_imgs_rot_fixed.npy')
train_gts = np.load('unique_train_gts_rot_fixed.npy')

test_imgs = np.load('test_imgs_rot_fixed.npy')
test_gts = np.load('test_gts.npy')
ids = np.load('ids.npy').tolist()

#create selected configuration
if args.config_id == 1:
    seed, data_transforms, arch, lr, num_cycles, num_epochs_per_cycle = config1()
elif args.config_id == 2:
    seed, data_transforms, arch, lr, num_cycles, num_epochs_per_cycle = config2()
elif args.config_id == 3:
    seed, data_transforms, arch, lr, num_cycles, num_epochs_per_cycle = config3()
elif args.config_id == 4:
    seed, data_transforms, arch, lr, num_cycles, num_epochs_per_cycle = config4()


sss = StratifiedKFold(n_splits=5, shuffle = True, random_state=seed)

models_arr = [] #array of snapshot models for each fold
sc_arr = []
fold = 0

val_prob = torch.zeros((train_imgs.shape[0], 3), dtype = torch.float32).to(device)    
#train a model for each split
for train_index, val_index in sss.split(train_imgs, train_gts):
    print(fold)
    fold += 1
    #define dataset and loader for training and validation
    image_datasets = {'train': ICLRDataset(train_imgs, train_gts, 'train', train_index, data_transforms['train']),
		      'val': ICLRDataset(train_imgs, train_gts, 'val', val_index, data_transforms['val'])}

    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=16),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=16, shuffle=False, num_workers=16)}
    #create model instance
    model_ft = arch(pretrained=True)
    try:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 3)
    except:
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, 3)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    dataset_sizes = {x:len(image_datasets[x]) for x in ['train', 'val']}
    #start training
    model_ft_arr, ensemble_loss, _, fold_val_prob = train_model_snapshot(model_ft, criterion, lr, dataloaders, dataset_sizes, device,
                           num_cycles=num_cycles, num_epochs_per_cycle=num_epochs_per_cycle)
    val_prob[val_index] = fold_val_prob
    models_arr.extend(model_ft_arr)
    sc_arr.append(ensemble_loss)

save_dir='trails'
idx = 93 + args.config_id
#save validation probabilites for ensemble
np.save(os.path.join(save_dir, 'val_prob_trail_%d'%(idx)), val_prob.detach().cpu().numpy())

print('mean val loss:', F.nll_loss(torch.log(val_prob), torch.tensor(train_gts).to(device)))

#predict on test set using average of snapshots of kfold training
image_datasets['test'] = ICLRDataset(test_imgs, test_gts, 'test', None, data_transforms['val'])
test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=4,shuffle=False, num_workers=16)
res = test(models_arr, test_loader, device)
#save test probabilites for ensemble
np.save(os.path.join(save_dir, 'test_prob_trail_%d'%(idx)), res)
