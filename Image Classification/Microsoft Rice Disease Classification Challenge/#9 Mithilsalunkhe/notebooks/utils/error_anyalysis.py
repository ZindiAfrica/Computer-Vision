import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('/home/mithil/PycharmProjects/Rice/oof/swinv2_base_window12to24_192to384_22kft1k_cutout_tta.csv')
probablity_2 = torch.log(torch.tensor(
    np.load('/home/mithil/PycharmProjects/Rice/oof/swinv2_large_window12to24_192to384_22kft1k_pseudo_25_epoch_diff_type_tta.npy',
            allow_pickle=True)))
blast_index = list(df.index[df['label'] == 'blast'])
brown_index = list(df.index[df['label'] == 'brown'])
healthy_index = list(df.index[df['label'] == 'healthy'])
label_encoder = preprocessing.LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

loss = nn.NLLLoss()
blast_probality = probablity_2[blast_index]
blast_labels = torch.tensor(df['label'][blast_index].values)
blast_loss = loss(blast_probality, blast_labels).item()
print(f"blast Label Loss : {blast_loss}")
brown_probality = probablity_2[brown_index]
brown_labels = torch.tensor(df['label'][brown_index].values)
brown_loss = (loss(brown_probality, brown_labels).item())
print(f"brown Label Loss : {brown_loss}")
healthy_probality = probablity_2[healthy_index]
healthy_labels = torch.tensor(df['label'][healthy_index].values)
healthy_loss = loss(healthy_probality, healthy_labels).item()
print(f"healthy Label Loss : {healthy_loss}")
data = {"healthy Label Loss": healthy_loss, "brown Label Loss": brown_loss, "blast Label Loss": blast_loss}
names = list(data.keys())
values = list(data.values())

plt.bar(range(len(data)), values, tick_label=names)
plt.show()
