import os
import random
import numpy as np
import torch

from model import *

def set_random_state(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    
def get_model(load_info):
    if load_info['version'] == 1:
        model = CNN_Model_v1(load_info['model_name'], load_info['global_dim'])
    elif load_info['version'] == 2:
        model = CNN_Model_v2(load_info['model_name'], load_info['global_dim'])

    checkpoint_dict = torch.load(load_info['path'])                
    model.load_state_dict(checkpoint_dict['Model_state_dict'])
                    
    print('(Val) loss is {}'.format(checkpoint_dict['Current_val_Loss']))
    print('(Val) f1 score is {}'.format(checkpoint_dict['Current_val_f1_score']))
    print('(Val) rmse is {}'.format(checkpoint_dict['Current_val_rmse']))    
    print('(Train) loss is {}'.format(checkpoint_dict['Current_train_Loss']))
    print('(Train) f1 score is {}'.format(checkpoint_dict['Current_train_f1_score']))
    print('(Train) rmse is {}'.format(checkpoint_dict['Current_train_rmse']))
    
    model = model.cuda()
    model.eval()
    return model