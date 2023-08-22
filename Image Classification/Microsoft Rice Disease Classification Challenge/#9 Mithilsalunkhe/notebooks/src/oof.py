# Utils
import argparse
import glob
from pathlib import Path
import pandas as pd
from sklearn import preprocessing
import yaml

# Deep learning Stuff
from torch import nn
import ttach as tta
from torch.utils.data import DataLoader

# Function Created by me
from dataset import *
from model import *
from train_func import *


def main(cfg):
    train_df = pd.read_csv(cfg['train_file_path'])

    train_df['file_path'] = train_df['Image_id'].apply(lambda x: return_filpath(x, folder=cfg['train_dir']))
    seed_everything(cfg['seed'])
    gc.enable()
    device = return_device()
    label_encoder = preprocessing.LabelEncoder()
    train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
    oof_preds = None
    oof_probablity = None
    oof_ids = []
    oof_targets = []
    for fold in range(5):
        if fold in cfg['folds']:
            valid = train_df[train_df['fold'] == fold].reset_index(drop=True)

            valid_path = valid['file_path']
            valid_labels = valid['Label']
            valid_dataset = Cultivar_data_oof(valid_path, cfg, valid_labels, ids=valid['Image_id'].values,
                                              transform=get_valid_transforms(cfg['image_size']))
            val_loader = DataLoader(
                valid_dataset, batch_size=cfg['batch_size'], shuffle=False,
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
            )
            model = BaseModelFeature(cfg)
            path = glob.glob(f"{cfg['model_dir']}/{cfg['model']}_fold{fold}*.pth")
            model.load_state_dict(torch.load(path[0]))
            model = tta.ClassificationTTAWrapper(model, tta.aliases.flip_transform())
            model = model.to(device)
            ids, target, preds, probablity, accuracy = oof_fn(val_loader, model, cfg)
            print(f"Fold: {fold} Accuracy: {accuracy}")
            oof_preds = np.concatenate([oof_preds, preds]) if oof_preds is not None else preds
            oof_probablity = np.concatenate([oof_probablity, probablity]) if oof_probablity is not None else probablity
            oof_ids.extend(ids)
            oof_targets.extend(target)

            del model
            del val_loader
            del valid_dataset
            del ids, target, preds, probablity, accuracy
            torch.cuda.empty_cache()
            gc.collect()
    oof_pred_real = label_encoder.inverse_transform(oof_preds)
    oof_targets_real = label_encoder.inverse_transform(oof_targets)
    loss = nn.NLLLoss()
    print(f"Loss {(loss(torch.tensor(oof_probablity), torch.tensor(oof_targets)).item())}")
    oof_probablity = np.array(torch.exp(torch.tensor(oof_probablity)))
    blast = []
    brown = []
    healthy = []
    for i in oof_probablity:
        blast.append(i[0])
        brown.append(i[1])
        healthy.append(i[2])

    oof_df = pd.DataFrame.from_dict(
        {'image_id': oof_ids, 'label': oof_targets_real, 'prediction': oof_pred_real, 'cultivar_int': oof_preds,
         'target_int': oof_targets, 'blast': blast, 'brown': brown, 'healthy': healthy})
    oof_df.to_csv(cfg['oof_file_path'], index=False)
    np.save(cfg['oof_probablity_path'], oof_probablity)


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    with open(str(args.file), "r") as stream:
        cfg = yaml.safe_load(stream)
    main(cfg)
