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

def cross_entropy(y, p):
    p /= p.sum(1).reshape(-1,1)
    return F.nll_loss(torch.log(torch.tensor(p)), torch.tensor(y)).numpy()

def weighted_cross_entropy(y, p):
    p /= p.sum(1).reshape(-1,1)
    w_arr = np.array([0.53, 0.3, 0.0])
    return np.sum([F.nll_loss(torch.log(torch.tensor(p[y==c])), torch.tensor(y[y==c])).numpy()*w_arr[c] for c in range(3)])

class ensembleSelection:

    def __init__(self, metric):
        self.metric = metric
        
    def _compare(self, sc1, sc2):
        if sc1 < sc2:
            return True
        return False
        
    def _initialize(self, X_p, y):
        """
        This function finds the id of the best validation probabiltiy
        """
        current_sc = self.metric(y, X_p[0])
        ind = 0
        for i in range(1, X_p.shape[0]):
            sc = self.metric(y, X_p[i])
            if self._compare(sc, current_sc):
                current_sc = sc
                ind = i
        return ind, current_sc
        
    def es_with_replacement(self, X_p, Xtest_p, y):
        best_ind, best_sc = self._initialize(X_p, y)
        current_sc = best_sc
        sumP = np.copy(X_p[best_ind])
        sumP_test = np.copy(Xtest_p[best_ind])
        i = 1
        # find the best combintation of input models' reuslts
        while True:
            i += 1
            ind = -1
            for m in range(X_p.shape[0]):
                #check if adding model m to the combination of best models will improve the results or not
                sc = self.metric(y, (sumP*X_p[m])**(1/i))
                if self._compare(sc, current_sc):
                    current_sc = sc
                    ind = m
            if ind>-1:
                sumP *= X_p[ind]
                sumP_test *= Xtest_p[ind]
            else:
                break
        sumP = sumP**(1/(i-1))
        sumP_test = sumP_test**(1/(i-1))

        sumP /= sumP.sum(1).reshape(-1,1)
        sumP_test /= sumP_test.sum(1).reshape(-1,1)
        
        return current_sc, sumP, sumP_test
        
    def es_with_bagging(self, X_p, Xtest_p, y, f = 0.5, n_bags = 20):
        list_of_indecies = [i for i in range(X_p.shape[0])]
        bag_size = int(f*X_p.shape[0])
        sumP = None
        sumP_test = None
        for i in range(n_bags):
            #create a random subset (bag) of models
            model_weight = [0 for j in range(X_p.shape[0])]
            rng = np.copy(list_of_indecies)
            np.random.shuffle(rng)
            rng = rng[:bag_size]
            #find the best combination from the input bag
            sc, p, ptest = self.es_with_replacement(X_p[rng], Xtest_p[rng], y)
            print('bag: %d, sc: %f'%(i, sc))
            if sumP is None:
                sumP = p
                sumP_test = ptest
            else:
                sumP *= p
                sumP_test *= ptest
                
        #combine the reuslts of all bags
        sumP = sumP**(1/n_bags)
        sumP_test = sumP_test**(1/n_bags)

        sumP /= sumP.sum(1).reshape(-1,1)
        sumP_test /= sumP_test.sum(1).reshape(-1,1)

        sumP[sumP < 1e-6] = 1e-6
        sumP_test[sumP_test < 1e-6] = 1e-6

        final_sc = self.metric(y, sumP)
        print('avg sc: %f'%(final_sc))
        return (final_sc, sumP, sumP_test)

np.random.seed(4321)

n = 50

#read training gt
train_gts = np.load('unique_train_gts_rot_fixed.npy')

#read validation probability on training data generated from automatuic hypropt trails and manual trails
#and create a matrix of (N,D,3) where N is the number of models and D is the data size
train_prob = np.array([np.load('trails/val_prob_trail_%d.npy'%(i)) for i in range(94,98)] + 
                      [np.load('trails/val_prob_trail_%d.npy'%(i)) for i in range(n)])

#read test probability generated from hypropt trails and manual trails
#and create a matrix of (N,D,3) where N is the number of models and D is the data size
test_prob = np.array([np.load('trails/test_prob_trail_%d.npy'%(i)) for i in range(94,98)] + 
                     [np.load('trails/test_prob_trail_%d.npy'%(i)) for i in range(n)]) 

ids = np.load('ids.npy').tolist()

#use ensemble selection algorithm to find best combination of models using geometric average
es_obj = ensembleSelection(cross_entropy)
sc, es_train_prob, es_test_prob = es_obj.es_with_bagging(train_prob, test_prob, train_gts, n_bags = 10, f = 0.65)

#detect samples with high confidence for healthy wheat
idx = (np.max(es_test_prob, 1) > 0.7) & (np.argmax(es_test_prob, 1) == 2)

#create another ensemble with more weights for leaf and stem classes
es_obj = ensembleSelection(weighted_cross_entropy)
sc, es_train_prob, es_test_prob = es_obj.es_with_bagging(train_prob, test_prob, train_gts, n_bags = 10, f = 0.65)

#increase the probability of confident samples for healthy wheat
es_test_prob[idx, 0] = 1e-6 
es_test_prob[idx, 1] = 1e-6
es_test_prob[idx, 2] = 1.0

#create submission
sub = pd.read_csv('sample_submission.csv')
sub['ID'] = ids
lbl_names = os.listdir('train_data')
for i, name in enumerate(lbl_names):
    sub[name] = es_test_prob[:,i].tolist()
sub.to_csv('final_sub.csv', index = False)

