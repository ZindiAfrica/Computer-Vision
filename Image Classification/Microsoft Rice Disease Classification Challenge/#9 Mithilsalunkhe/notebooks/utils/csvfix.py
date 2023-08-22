import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/mithil/PycharmProjects/Rice/data/Train.csv')
index_to_use = []
for i, x in enumerate(df['Image_id'].values):
    if "_rgn" not in x:
        index_to_use.append(i)
df = df.iloc[index_to_use]
df = df.reset_index(drop=True)
df.to_csv('/home/mithil/PycharmProjects/Rice/data/train.csv', index=False)
df['Label'].hist()
plt.show()
