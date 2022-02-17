import time
import math
import numpy
import pandas
import random
import argparse
from glob import glob
from PIL import Image

import albumentations
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from gradual_warmup_scheduler import GradualWarmupScheduler

from image_model import ImageModel
from wheat_dataset import WheatDataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from config import *
from sklearn.metrics import mean_squared_error


def files_in_path(file_extension: str, path: str, recursive: bool = True):
    return glob(path + f'/**/*.{file_extension}', recursive=recursive)


def seed_all(seed=42561):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_score(y_trues, y_preds, config):
    if config.regression:
        y_true = y_trues
        y_pred = y_preds
    else:
        y_true = numpy.argmax(y_trues, axis=1) + 1
        y_pred = sum([y_preds[:, i] * (i + 1) for i in range(7)])
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def get_train_test_df(root_dir: str, images_dir: str) -> tuple:
    train_df = pandas.read_csv(os.path.join(root_dir, 'Train.csv'))
    all_files = files_in_path(path=images_dir, file_extension='jpeg')
    dir_split_char = '/' if '/' in all_files[0] else '\\'
    all_ids = [file.split(dir_split_char)[-1].split('.')[0] for file in all_files]

    test_ids = set(all_ids) - set(train_df.UID.values.tolist())
    test_df = pandas.DataFrame(test_ids, columns=['UID'])
    return train_df, test_df


def create_folds_dataset(train_df: pandas.DataFrame, folds: int):
    print("Creating folds dataset!")
    train_df.loc[:, "kfold"] = -1

    train_df = train_df.sample(frac=1, random_state=57543).reset_index(drop=True)

    X = train_df.UID.values
    y = train_df.growth_stage.values

    kfold = StratifiedKFold(n_splits=folds, random_state=15435)

    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        train_df.loc[val_index, "kfold"] = fold

    return train_df


def prepare_data(config, label_quality: int = None, force_creation: bool = False):
    root_dir = config.root_dir
    images_dir = config.images_dir
    num_folds = config.folds
    train_folds_csv = os.path.join(root_dir, 'train_fold.csv')
    test_folds_csv = os.path.join(root_dir, 'test.csv')

    if force_creation or not os.path.exists(train_folds_csv) or not os.path.exists(test_folds_csv):
        train_df, test_df = get_train_test_df(root_dir=root_dir, images_dir=images_dir)
        print('growth_stage:\n', train_df.growth_stage.value_counts())
        print('label_quality:\n', train_df.label_quality.value_counts())
        train_df = create_folds_dataset(train_df=train_df, folds=num_folds)
        train_df.to_csv(train_folds_csv, index=False)
        test_df.to_csv(test_folds_csv, index=False)

    train_df = pandas.read_csv(train_folds_csv)
    test_df = pandas.read_csv(test_folds_csv)

    if label_quality:
        train_df = train_df[train_df.label_quality == label_quality].reset_index(drop=True)

    return train_df, test_df


def get_train_test_val(fold, config, debug_mode):
    train_df, test_df = prepare_data(config=config, label_quality=2, force_creation=False)
    df_train = train_df[train_df.kfold != fold].reset_index(drop=True)
    df_val = train_df[train_df.kfold == fold].reset_index(drop=True)
    if debug_mode:
        df_train = df_train[0:200]
        df_val = df_val[0:40]
    return df_train, df_val, test_df


def get_transforms():
    transforms_train = albumentations.Compose([
            # albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.5),
            albumentations.Blur(p=0.5),
            albumentations.CoarseDropout(p=0.5),

            albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0, p=0.5),
        ])

    transforms_val = albumentations.Compose([])
    return transforms_train, transforms_val


def resize_transform(image):
    return numpy.asarray(Image.fromarray(image.astype(numpy.uint8)).resize((config.image_size, config.image_size)))


def get_test_loader(test_df, config, transforms):
    test_dataset = WheatDataset(df=test_df, config=config, tile_mode=0, rand=False, transform=transforms,
                                resize_transform=resize_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.val_batch_size, sampler=SequentialSampler(test_dataset),
                             num_workers=config.num_workers)
    return test_loader


def get_dataloaders(train_df, val_df, config):
    transforms_train, transforms_val = get_transforms()
    train_dataset = WheatDataset(df=train_df, config=config, tile_mode=0, rand=True, transform=transforms_train,
                                 resize_transform=resize_transform)
    val_dataset = WheatDataset(df=val_df, config=config, tile_mode=0, rand=False, transform=transforms_val,
                               resize_transform=resize_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, sampler=RandomSampler(train_dataset),
                              num_workers=config.num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, sampler=SequentialSampler(val_dataset),
                              num_workers=config.num_workers)
    return train_loader, valid_loader


def get_val_dataloader(val_df, config, transforms):
    val_dataset = WheatDataset(df=val_df, config=config, tile_mode=0, rand=False, transform=transforms,
                               resize_transform=resize_transform)
    valid_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, sampler=SequentialSampler(val_dataset),
                              num_workers=config.num_workers)
    return valid_loader


def perform_predictions(model, device, test_df, val_df, config, filename):
    tta_transforms = [
        dict(name='none', transform=albumentations.Compose([])),
        dict(name='horizontal_flip', transform=albumentations.Compose([albumentations.HorizontalFlip(p=1)])),
        dict(name='vertical_flip', transform=albumentations.Compose([albumentations.VerticalFlip(p=1)]))
    ]

    y_trues = val_df.growth_stage
    del val_df['growth_stage']
    avg_preds = None

    for transform_info in tta_transforms:
        transform_name = transform_info.get('name')
        transforms = transform_info.get('transform')
        val_loader = get_val_dataloader(val_df, config, transforms)
        preds = predict(model, val_loader, device)
        avg_preds = preds if avg_preds is None else avg_preds + preds
        score = get_score(y_trues=y_trues, y_preds=preds, config=config)
        print(f"transform {transform_name} --> score = {score}")

    avg_preds = avg_preds / len(tta_transforms)
    score = get_score(y_trues=y_trues, y_preds=avg_preds, config=config)
    print("TTA val score =", score)

    print("Prediction test set")
    avg_preds = None
    for transform_info in tta_transforms:
        transform_name = transform_info.get('name')
        transforms = transform_info.get('transform')
        test_loader = get_val_dataloader(test_df, config, transforms)
        preds = predict(model, test_loader, device)
        save_preds(preds=preds, filename=transform_name + "_" + filename, output_dir=config.output_dir)
        avg_preds = preds if avg_preds is None else avg_preds + preds

    avg_preds = avg_preds / len(tta_transforms)
    if len(tta_transforms) > 1:
        filename = 'tta_' + filename

    save_preds(preds=avg_preds, filename=filename, output_dir=config.output_dir)


def save_preds(preds, filename, output_dir):
    sub_df = test_df[['UID']]
    sub_df['growth_stage'] = preds
    sub_df.to_csv(os.path.join(output_dir, filename), index=False)


def perform_train(model, device, train_df, val_df, config, best_file, fine_tune=False):
    train_loader, valid_loader = get_dataloaders(train_df, val_df, config)
    criterion = nn.MSELoss() if config.regression else nn.BCEWithLogitsLoss()

    best_loss, best_acc, best_score = val_epoch(loader=valid_loader, device=device, criterion=criterion)
    print(f"Initial val scores. score = {best_score} loss = {best_loss} accuracy = {best_acc}")

    lr = config.init_lr / config.warmup_factor
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if config.use_gradual_lr_warmup and not fine_tune:
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs - config.warmup_epo)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=config.warmup_factor, total_epoch=config.warmup_epo,
                                           after_scheduler=scheduler_cosine)
    else:
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config.learning_rate_factor, verbose=True, eps=1e-6,
                                      patience=config.learning_rate_patience)

    epochs = 20 if fine_tune else config.epochs + 1

    print("Total epochs", epochs)

    for epoch in range(1, epochs):
        print(time.ctime(), 'Epoch:', epoch)
        train_loss, train_acc, train_score = train_epoch(loader=train_loader, device=device, optimizer=optimizer,
                                                         criterion=criterion)
        scheduler.step(epoch - 1)

        val_loss, acc, score = val_epoch(loader=valid_loader, device=device, criterion=criterion)
        content = time.ctime() + ' ' + f'Epoch {epoch}/{epochs}, lr: {optimizer.param_groups[0]["lr"]:.7f}, ' \
                                       f'train loss: {numpy.mean(train_loss):.5f}, train acc: {train_acc:.5f} ' \
                                       f'train score: {train_score:.5f} --> ' \
                                       f'val loss: {numpy.mean(val_loss):.5f}, val acc: {acc:.5f} ' \
                                       f'val score: {score:.5f}'

        if val_loss < best_loss:
            content += '\n\tval_loss ({:.6f} --> {:.6f}).  Saving model ...'.format(best_loss, val_loss)
            torch.save(model.state_dict(), os.path.join(config.output_dir, best_file))
            best_loss = val_loss
        else:
            print(f"\tval_loss did not improve from {best_loss:.6f}")

        print(content)

        with open(os.path.join(config.output_dir, log_filename), 'a') as appender:
            if epoch == 1:
                appender.write(config_str + '\n')

            appender.write(content + '\n')

    print("Best Loss was", best_loss)


def train_epoch(loader, device, optimizer, criterion):
    model.train()
    train_loss = []
    PREDS = []
    TARGETS = []

    bar = tqdm(loader)
    for (data, target) in bar:
        data, target = data.to(device), target.to(device)
        loss_func = criterion
        optimizer.zero_grad()
        logits = model(data)

        if config.regression:
            logits = logits.squeeze()

        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

        if config.regression:
            pred = logits.detach()
        else:
            # pred = torch.nn.functional.softmax(logits, dim=1).detach()
            pred = logits.sigmoid().detach()

        PREDS.append(pred)
        TARGETS.append(target)

    if len(PREDS[-1].shape) == 0:
        # correct for batches of size 1, where output would be a scaler
        PREDS[-1] = torch.FloatTensor([PREDS[-1]]).to(device)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    if config.regression:
        PREDS = PREDS.squeeze()
        acc = (PREDS.round() == TARGETS).mean() * 100.
    else:
        acc = (numpy.argmax(PREDS, axis=1) == numpy.argmax(TARGETS, axis=1)).mean() * 100.
    score = get_score(y_trues=TARGETS, y_preds=PREDS, config=config)

    return train_loss, acc, score


def val_epoch(loader, device, criterion, get_output=False):
    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)

            if config.regression:
                pred = logits
            else:
                pred = torch.nn.functional.softmax(logits, dim=1).detach()

            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target)

            if config.regression:
                logits = logits.squeeze()

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

        val_loss = numpy.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    if config.regression:
        PREDS = PREDS.squeeze()
        acc = (PREDS.round() == TARGETS).mean() * 100.
    else:
        acc = (numpy.argmax(PREDS, axis=1) == numpy.argmax(TARGETS, axis=1)).mean() * 100.
    score = get_score(y_trues=TARGETS, y_preds=PREDS, config=config)

    if get_output:
        return LOGITS
    else:
        return val_loss, acc, score


def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            logits = model(data)
            if config.regression:
                preds.append(logits.detach())
            else:
                preds.append(torch.nn.functional.softmax(logits, dim=1).detach())

    preds = torch.cat(preds).cpu().numpy()

    if not config.regression:
        preds = sum([preds[:, i] * (i + 1) for i in range(7)])

    return preds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', required=True)
    parser.add_argument('--output_folder', required=True)
    return parser.parse_args()


args = parse_args()
DATA_DIR = args.data_folder
OUTPUT_DIR = args.output_folder
GPU_ID = 0
image_dir = os.path.join(DATA_DIR, 'Images')
seed_all(seed=354)

debug_mode = False
folds = list(range(0, 5))
device = torch.device(f'cuda:{GPU_ID}')

print(f"Creating output directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

required_files = ['SampleSubmission.csv', 'test.csv', 'Train.csv']

for file in required_files:
    if not os.path.exists(os.path.join(DATA_DIR, file)):
        print(f"File {file} is missing.")
        exit()

image_files = files_in_path(file_extension='jpeg', path=image_dir)
if len(image_files) < 14000:
    print(f"Expected at least 14000 images in {image_dir}. Got {len(image_files)} images")
    exit()

stages = [
    dict(configs=[Resnet50Config, EfficientNetB0, SeResNextConfig], ensemble_filters=['horizontal', 'none', 'vertical'],
         fine_tune=False, output_file='combined_test_preds.csv'),
    dict(configs=[Resnet50Config, EfficientNetB0, SeResNextConfig], ensemble_filters=['horizontal', 'none'],
         fine_tune=True, output_file='submission.csv')
]

for stage in stages:
    configs = stage['configs']
    fine_tune = stage['fine_tune']
    ensemble_filters = stage['ensemble_filters']

    for config in configs:
        config.root_dir = DATA_DIR
        config.output_dir = OUTPUT_DIR
        config.images_dir = os.path.join(config.root_dir, 'Images')

        config_str = "Config:\n\t" + '\n\t'.join("%s: %s" % item for item in vars(config).items() if '__' not in item[0])
        print(config_str)

        for fold in folds:
            print(f"\tFold = {fold}")
            train_df, val_df, test_df = get_train_test_val(config=config, debug_mode=debug_mode, fold=fold)

            if fine_tune:
                pseudo_test_df = pandas.read_csv('combined_test_preds.csv')
                train_df = train_df.append(pseudo_test_df)
                train_df.sample(frac=1, random_state=354).reset_index(drop=True)

            print(f"{len(train_df)} Training Samples.\n{len(val_df)} Validation Samples.\n")

            best_file = f'{config.neural_backbone}_best_fold{fold}_dropout_{config.dropout}.pth'
            if config.regression:
                best_file = 'regression_' + best_file

            log_filename = best_file.replace('.pth', '.txt')

            model = ImageModel(model_name=config.neural_backbone, device=device, dropout=config.dropout, neurons=0,
                               num_classes=config.num_classes, extras_inputs=[], base_model_pretrained_weights=None)
            model.load(directory=config.output_dir, filename=best_file)
            model = model.to(device)

            if fine_tune:
                best_file = 'finetune_' + best_file

            checkpoint_path = os.path.join(config.output_dir, best_file)
            if os.path.exists(checkpoint_path):
                print(f"WARNING: TRAINED CHECKPOINT ALREADY EXISTS IN {checkpoint_path}. "
                      f"SKIPPING TRAINING FOR THIS MODEL/FOLD")
            else:
                print("Training model!")
                perform_train(model, device, train_df, val_df, config, best_file, fine_tune)

            print(f"Predicting model {config.neural_backbone} for fold {fold}!")
            model.load(directory=config.output_dir, filename=best_file)
            filename = "submission_" + best_file.replace('.pth', '.csv')
            perform_predictions(model, device, test_df, val_df, config, filename)

    csvs = files_in_path(path=config.output_dir, file_extension='csv')
    files_to_ensemble = []
    for filter in ensemble_filters:
        if fine_tune:
            files = [csv for csv in csvs if filter in csv and 'finetu' in csv]
        else:
            files = [csv for csv in csvs if filter in csv]
        files_to_ensemble.extend(files)

    print(f"Combining {len(files_to_ensemble)} predictions. Files: {files_to_ensemble}")
    final = pandas.read_csv(files_to_ensemble[0])
    del final['growth_stage']
    for index, submission in enumerate(files_to_ensemble):
        sub = pandas.read_csv(submission)
        final[str(index)] = sub.growth_stage

    pred_columns = [str(pred_column) for pred_column in range(len(files_to_ensemble))]
    preds = final[pred_columns].mean(axis=1)
    sub.growth_stage = preds
    sub.to_csv(stage['output_file'], index=False)



