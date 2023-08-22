import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from joblib import Parallel, delayed

train_df = pd.read_csv('/home/mithil/PycharmProjects/Rice/data/train.csv')
label_encoder = preprocessing.LabelEncoder()
train_df['Label'] = label_encoder.fit_transform(train_df['Label'])
labels = train_df['Label']
labels = torch.tensor(labels)
probablity_1 = torch.tensor(
    np.load('/home/mithil/PycharmProjects/Rice/oof/swin_v2_base_25_epoch_no_mixup_tta.npy',
            allow_pickle=True))
probablity_2 = torch.tensor(
    np.load('/home/mithil/PycharmProjects/Rice/oof/swinv2_large_window12to24_192to384_22kft1k_mixup_25_epoch_tta.npy',
            allow_pickle=True))
probablity_3 = torch.tensor(
    np.load('/home/mithil/PycharmProjects/Rice/oof/swinv2_large_window12to24_192to384_22kft1k_25_epoch_tta.npy',
            allow_pickle=True))
probablity_4 = torch.tensor(
    np.load(
        '/home/mithil/PycharmProjects/Rice/oof/swin_large_patch4_window12_384_pseudo_25_epoch_tta.npy',
        allow_pickle=True))

probablity_5 = torch.tensor(
    np.load(
        '/home/mithil/PycharmProjects/Rice/oof/swin_large_25_epoch_tta.npy',
        allow_pickle=True))
probablity_6 = torch.tensor(
    np.load(
        '/home/mithil/PycharmProjects/Rice/oof/swinv2_large_window12to24_192to384_22kft1k_pseudo_25_epoch_diff_type_tta.npy',
        allow_pickle=True))
best_loss = np.inf
best_weight = 0
loss_list = []
weights_list = []
loss = nn.NLLLoss()

for x in tqdm(range(1000000)):

    i = np.random.random(6).astype(np.float16)
    i /= i.sum()

    probablity = torch.log(
        probablity_1 * i[0] + probablity_2 * i[1] + probablity_3 * i[2] + probablity_4 * i[3] + probablity_5 * i[
            4] + probablity_6 * i[5])
    loss_item = (loss(probablity, labels).item())
    if loss_item < best_loss:
        best_weight = i
        best_loss = loss_item
    loss_list.append(loss_item)
print(best_loss)
print(best_weight)
