import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
from PIL import Image, ExifTags

def train_model_snapshot(model, criterion, lr, dataloaders, dataset_sizes, device, num_cycles, num_epochs_per_cycle):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1000000.0
    model_w_arr = []
    prob = torch.zeros((dataset_sizes['val'], 3), dtype = torch.float32).to(device)
    lbl = torch.zeros((dataset_sizes['val'],), dtype = torch.long).to(device)
    for cycle in range(num_cycles):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)#, weight_decay = 0.0005)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs_per_cycle*len(dataloaders['train']))
        for epoch in range(num_epochs_per_cycle):
            #print('Cycle {}: Epoch {}/{}'.format(cycle, epoch, num_epochs_per_cycle - 1))
            #print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                idx = 0
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        if (epoch == num_epochs_per_cycle-1) and (phase == 'val'):
                            prob[idx:idx+inputs.shape[0]] += F.softmax(outputs, dim = 1)
                            lbl[idx:idx+inputs.shape[0]] = labels
                            idx += inputs.shape[0]
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            #print(optimizer.param_groups[0]['lr'])
                    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                #    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            #print()
        model_w_arr.append(copy.deepcopy(model.state_dict()))

    prob /= num_cycles
    ensemble_loss = F.nll_loss(torch.log(prob), lbl)  
    ensemble_loss = ensemble_loss.item()
    time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 60, time_elapsed % 60))
    #print('Ensemble Loss : {:4f}, Best val Loss: {:4f}'.format(ensemble_loss, best_loss))

    # load best model weights
    model_arr =[]
    for weights in model_w_arr:
        model.load_state_dict(weights)   
        model_arr.append(model) 
    return model_arr, ensemble_loss, best_loss, prob

def test(models_arr, loader, device):
    res = np.zeros((610, 3), dtype = np.float32)
    for model in models_arr:
        model.eval()
        res_arr = []
        for inputs, _ in loader:
            inputs = inputs.to(device)
            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = F.softmax(model(inputs), dim = 1)    
                res_arr.append(outputs.detach().cpu().numpy())
        res_arr = np.concatenate(res_arr, axis = 0)
        res += res_arr
    return res / len(models_arr)

def read_train_data(p):
    imgs = []
    labels = []
    for i, lbl in enumerate(os.listdir(p)):
        for fname in os.listdir(os.path.join(p, lbl)):
            #read image
            img = Image.open(os.path.join(p, lbl, fname))
            #rotate image to original view
            try:
                exif=dict((ExifTags.TAGS[k], v) for k, v in img._getexif().items() if k in ExifTags.TAGS)
                if exif['Orientation'] == 3:
                    img=img.rotate(180, expand=True)
                elif exif['Orientation'] == 6:
                    img=img.rotate(270, expand=True)
                elif exif['Orientation'] == 8:
                    img=img.rotate(90, expand=True)
            except:
                pass
            #resize all images to the same size
            img = np.array(img.convert('RGB').resize((512,512), Image.ANTIALIAS))
            imgs.append(img)
            labels.append(i)
    return imgs, labels

def read_test_data(p):
    imgs = []
    labels = []
    ids = []
    for fname in os.listdir(p):
        #read image
        img = Image.open(os.path.join(p, fname))
        #rotate image to original view
        try:
            if not('DMWVNR' in fname):
                exif=dict((ExifTags.TAGS[k], v) for k, v in img._getexif().items() if k in ExifTags.TAGS)
                if exif['Orientation'] == 3:
                    img=img.rotate(180, expand=True)
                elif exif['Orientation'] == 6:
                    img=img.rotate(270, expand=True)
                elif exif['Orientation'] == 8:
                    img=img.rotate(90, expand=True)
        except:
            pass
        #resize all images to the same size
        img = img.convert('RGB').resize((512,512), Image.ANTIALIAS)
        imgs.append(np.array(img.copy()))
        labels.append(0)
        ids.append(fname.split('.')[0])
        img.close()
    return imgs, labels, ids
