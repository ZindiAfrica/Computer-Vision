from tqdm import tqdm
import torch
import numpy as np
import argparse
import yaml
from argparse import ArgumentParser
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from dataset import TestDataset
import pandas as pd
from augment import *
from model import *

parser = ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, required=False, default="configs/tf_efficientnetv2_s.yaml"
)
parser.add_argument(
    "-f", "--fold", type=int, required=False, default=0
)
args = parser.parse_args()

logger.info("Loading config")
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    
def get_test_file_path(id):
    return f"{config['test_dir']}/{id}"



def infer_fn(test_loader, model, device):
    model.eval()
    preds = []
    bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, (images) in bar:
        
        images = images.to(device)
        batch_size = images.size(0)
        probs = torch.zeros(images.shape[0]).to(device)
        with torch.no_grad():
            out = model(images)
            probs += out.squeeze()
        preds.append(probs.to('cpu').numpy())
        
    preds = np.concatenate(preds)
    return preds

if __name__ == '__main__':
    test_df = pd.read_csv(f"{config['root_dir']}/Test.csv")
    test_df['file_path'] = test_df['ImageId'].apply(get_test_file_path)
    valid_transforms = data_transforms(phase='valid', img_size=config['img_size'])
    test_dataset = TestDataset(config['test_dir'], test_df, transforms=valid_transforms)

    test_loader = DataLoader(test_dataset, batch_size=config['valid_batch_size'], 
                              num_workers=4, shuffle=False, pin_memory=True)
    
    device = torch.device(config['device'])
    model = PlantationModel(config['backbone'])
    model.to(device)
    fold = args.fold
    model.load_state_dict(torch.load(f"{config['checkpoint_dir']}/best_RMSE_fold{fold}.pt"))
    preds = infer_fn(test_loader, model, device)
    preds[preds < 0] = 0
    sub_df = pd.read_csv(f"{config['root_dir']}/SampleSubmission.csv")
    sub_df['Target'] = preds
    sub_df.to_csv(f"submission/{config['backbone']}_fold{fold}.csv", index=False)