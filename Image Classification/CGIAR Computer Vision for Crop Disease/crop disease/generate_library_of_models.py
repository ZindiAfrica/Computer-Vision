import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
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
from utils import train_model_snapshot, test
from sklearn.metrics import confusion_matrix
from hyperopt import hp, tpe, fmin, Trials
from collections import OrderedDict


def score(params):
    global test_prob, val_prob, trails_sc_arr,idx
    print(params)
    k = 5
    sss = StratifiedKFold(n_splits=k, shuffle = True, random_state=seed_arr[idx])
    #define trail data augmentations
    data_transforms = {
        'train': transforms.Compose([
            transforms.ColorJitter(contrast = params['contrast'], hue = params['hue'], brightness = params['brightness']),
            transforms.RandomAffine(degrees = params['degrees']),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p = 0.5 if params['h_flip'] else 0.0),
            transforms.RandomVerticalFlip(p = 0.5 if params['v_flip'] else 0.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((params['val_img_size'], params['val_img_size'])),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    trail_test_prob = np.zeros((test_imgs.shape[0], 3), dtype = np.float32)
    trail_val_prob = torch.zeros((train_imgs.shape[0], 3), dtype = torch.float32).to(device)
    
    sc_arr = []
    models_arr = []
    fold = 0
    #train a model for each split
    for train_index, val_index in sss.split(train_imgs, train_gts):
        #define dataset and loader for training and validation
        image_datasets = {'train': ICLRDataset(train_imgs, train_gts, 'train', train_index, data_transforms['train'], params['img_mix_enable']),
	                      'val': ICLRDataset(train_imgs, train_gts, 'val', val_index, data_transforms['val'])}

        dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=16),
                       'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=16, shuffle=False, num_workers=16)}

        #create model instance
        model_ft = params['arch'](pretrained=True)
        try:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 3)
        except:
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, 3)
        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        dataset_sizes = {x:len(image_datasets[x]) for x in ['train', 'val']}
        
        model_ft_arr, ensemble_loss, _, fold_val_prob = train_model_snapshot(model_ft, criterion, params['lr'], dataloaders, dataset_sizes, device,
                               num_cycles=params['num_cycles'], num_epochs_per_cycle=params['num_epochs_per_cycle'])
        models_arr.extend(model_ft_arr)
        fold += 1
        sc_arr.append(ensemble_loss)
        trail_val_prob[val_index] = fold_val_prob
    
    #predict on test data using average of kfold models
    image_datasets['test'] = ICLRDataset(test_imgs, test_gts, 'test', None, data_transforms['val'])
    test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=4,shuffle=False, num_workers=16)
    trail_test_prob = test(models_arr, test_loader, device)

    print('mean val loss:', np.mean(sc_arr))

    test_prob.append(trail_test_prob)
    val_prob.append(trail_val_prob)

    #save validation and test results for further processing 
    np.save(os.path.join(save_dir, 'val_prob_trail_%d'%(idx)), trail_val_prob.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'test_prob_trail_%d'%(idx)), trail_test_prob)
    idx += 1
    
    trails_sc_arr.append(np.mean(sc_arr))

    torch.cuda.empty_cache()
    del models_arr

    return np.mean(sc_arr)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#read train data
train_imgs = np.load('unique_train_imgs_rot_fixed.npy')
train_gts = np.load('unique_train_gts_rot_fixed.npy')

#read test data
test_imgs = np.load('test_imgs_rot_fixed.npy')
test_gts = np.load('test_gts.npy')
ids = np.load('ids.npy').tolist()

test_prob = []
val_prob = []
trails_sc_arr = []

n_trails = 50
seed_arr = np.random.randint(low=0, high=1000000, size=n_trails)

#create search space for hyperparameter optimization
space = OrderedDict([('lr', hp.choice('lr', [i*0.001 for i in range(1,4)])),
                     ('num_cycles', hp.choice('num_cycles', range(3, 6))),
                     ('num_epochs_per_cycle', hp.choice('num_epochs_per_cycle', range(3, 6))),
                     ('arch', hp.choice('arch', [models.densenet201, models.densenet121, models.densenet169,
                                                 models.wide_resnet50_2, models.resnet152, 
                                                 models.resnet101, models.resnet50, models.resnet34, models.resnet18])),
                     ('img_mix_enable', hp.choice('img_mix_enable', [True, False])),
                     ('v_flip', hp.choice('v_flip', [True, False])),
                     ('h_flip', hp.choice('h_flip', [True, False])),
                     ('degrees', hp.choice('degrees', range(1, 90))),
                     ('contrast', hp.uniform('contrast', 0.0, 0.3)),
                     ('hue', hp.uniform('hue', 0.0, 0.3)),
                     ('brightness', hp.uniform('brightness', 0.0, 0.3)),
                     ('val_img_size', hp.choice('val_img_size', range(224, 512, 24))),
                     ])

trials = Trials()

idx = 0
save_dir = 'trails'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

#use tpe algorithm in hyperopt to generate a library of differnet models
best = fmin(fn=score,
    space=space,
    algo=tpe.suggest,
    trials=trials,
    max_evals=n_trails)

print(best)

np.save(os.path.join(save_dir, 'scores.npy'), np.array(trails_sc_arr))
