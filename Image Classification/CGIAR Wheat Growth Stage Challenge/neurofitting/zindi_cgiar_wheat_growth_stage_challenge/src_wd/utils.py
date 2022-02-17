import os
import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics import f1_score

def set_random_state(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    
class Metric_Meter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.y_pred = []
        self.y_true = []
    def update(self, y_pred, y_true, mode):
        assert mode in ['single', 'multiple'] 
        y_true = y_true.cpu().data.numpy()
        if mode == 'single':
            y_pred = y_pred[:,0].cpu().data.numpy()
            #y_pred = y_pred #.round().clip(min=0, max=6)
        else:
            y_pred = y_pred.cpu().softmax(dim=1).argmax(1)
        self.y_true += y_true.tolist()  
        self.y_pred += y_pred.tolist()                             
    def feedback(self):
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)        
        return f1_score(y_true, y_pred.round().clip(min=0, max=6), average='macro'), np.sqrt(np.mean((y_true-y_pred.clip(min=0, max=6))**2))

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, logits, y_true):
        return torch.sqrt( self.mse(logits, y_true) )