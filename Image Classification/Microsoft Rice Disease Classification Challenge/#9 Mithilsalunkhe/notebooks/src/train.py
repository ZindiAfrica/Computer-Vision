import argparse
from pathlib import Path
import wandb
from sklearn.model_selection import StratifiedKFold

import pandas as pd
# Deep learning Stuff
import yaml
from sklearn import preprocessing
from torch.optim import *
from torch.utils.data import DataLoader

# Function Created by me
from dataset import *
from model import *
from train_func import *
from loss import *


def main(cfg):
    wandb.init(project="RiceComp", entity="mithil", config=cfg)
    train_df = pd.read_csv(cfg['train_file_path'])

    train_df['file_path'] = train_df['Image_id'].apply(lambda x: return_filpath(x, folder=cfg['train_dir']))
    test_df = pd.read_csv(cfg['pseudo_file_path'])
    test_df['file_path'] = test_df['filename'].apply(lambda x: return_filpath(x, folder=cfg['train_dir']))
    test_df['Label'] = list(
        map(return_label, test_df['blast'].values, test_df['brown'].values, test_df['healthy'].values))
    seed_everything(cfg['seed'])
    gc.enable()
    device = return_device()
    label_encoder = preprocessing.LabelEncoder()
    train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    df = pd.DataFrame()
    for fold, (trn_index, val_index) in enumerate(skf.split(test_df, test_df['Label'])):
        valid = test_df.iloc[val_index]
        valid = valid.reset_index(drop=True)
        valid['fold'] = fold
        df = pd.concat([df, valid]).reset_index(drop=True)
    df.head()
    train_df = pd.concat([train_df, df]).reset_index(drop=True)
    for fold in range(5):

        if fold in cfg['folds']:
            best_model_name = None
            best_loss = np.inf

            train = train_df[train_df['fold'] != fold].reset_index(drop=True)
            # train = oversample(train)

            valid = train_df[train_df['fold'] == fold].reset_index(drop=True)

            train_path = train['file_path']
            train_labels = train['Label']
            valid_path = valid['file_path']
            valid_labels = valid['Label']
            model = BaseModelFeature(cfg)

            model.to(device)
            criterion = nn.CrossEntropyLoss()

            optimizer_args = cfg['optimizer_args']

            optimizer = eval(cfg['optimizer'])(model.parameters(), **optimizer_args)
            train_dataset = Cultivar_data(image_path=train_path,
                                          cfg=cfg,
                                          targets=train_labels,
                                          transform=get_train_transforms(cfg['image_size']),
                                          transform_rgn=get_train_transforms_rgn(cfg['image_size']))
            valid_dataset = Cultivar_data(image_path=valid_path,
                                          cfg=cfg,
                                          targets=valid_labels,
                                          transform=get_valid_transforms(cfg['image_size']),
                                          transform_rgn=get_valid_transforms_rgn(cfg['image_size']))
            train_loader = DataLoader(
                train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
            )

            val_loader = DataLoader(
                valid_dataset, batch_size=cfg['batch_size'] * 2, shuffle=False,
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
            )

            scheduler = get_scheduler(optimizer, cfg)
            for epoch in range(cfg['epochs']):
                train_fn(train_loader, model, criterion, optimizer, epoch, cfg, fold, scheduler)

                log_loss = validate_fn(val_loader, model, criterion, epoch, cfg, fold)
                torch.cuda.empty_cache()
                if log_loss < best_loss:
                    best_loss = log_loss
                    if best_model_name is not None:
                        os.remove(best_model_name)
                    torch.save(model.state_dict(),
                               f"{cfg['model_dir']}/{cfg['model']}_fold{fold}_epoch{epoch}_loss_{str(round(log_loss, 4))}.pth")
                    best_model_name = f"{cfg['model_dir']}/{cfg['model']}_fold{fold}_epoch{epoch}_loss_" \
                                      f"{str(round(log_loss, 4))}.pth"

                gc.collect()

            gc.collect()
            torch.cuda.empty_cache()
            del train_dataset
            del valid_dataset
            del train_loader
            del val_loader
            del model
            del optimizer
            del scheduler
            torch.cuda.empty_cache()


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument("--file", type=Path)
    parser.add_argument("--wandb_api_key")
    args = parser.parse_args()
    with open(str(args.file), "r") as stream:
        cfg = yaml.safe_load(stream)
    os.environ["WANDB_API_KEY"] = args.wandb_api_key

    os.makedirs(cfg['model_dir'], exist_ok=True)
    main(cfg)
