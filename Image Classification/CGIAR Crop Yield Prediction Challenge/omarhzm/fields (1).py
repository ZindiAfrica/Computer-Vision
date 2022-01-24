import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import glob, random
import sklearn
from sklearn.decomposition import PCA
from xgboost.sklearn import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor, RandomForestRegressor,VotingRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

import catboost
from catboost import CatBoostRegressor

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#import warnings
#warnings.filterwarnings('ignore')

folder = os.path.dirname(os.path.abspath(__file__))
train_new = pd.read_csv(folder+'/Train.csv')

bands_of_interest = ['S2_B5', 'S2_B4', 'S2_B3', 'S2_B2', 'CLIM_pr', 'CLIM_soil']
band_names = [l.strip() for l in open(folder + '/band_names.txt', 'r').readlines()]

def process_train(fid, folder= folder+'/imtrain'):
  fn = f'{folder}/{fid}.npy'
  arr = np.load(fn)
  values = {}
  for month in range(12):
    bns = [str(month) + '_' + b for b in bands_of_interest] # Bands of interest for this month 
    idxs = np.where(np.isin(band_names, bns)) # Index of these bands
    vs = arr[idxs, 20, 20] # Sample the im at the center point
    for bn, v in zip(bns, vs[0]):
      values[bn] = v
  return values

def process_test(fid, folder= folder+'/imtest'):
  fn = f'{folder}/{fid}.npy'
  arr = np.load(fn)
  values = {}
  for month in range(12):
    bns = [str(month) + '_' + b for b in bands_of_interest] # Bands of interest for this month 
    idxs = np.where(np.isin(band_names, bns)) # Index of these bands
    vs = arr[idxs, 20, 20] # Sample the im at the center point
    for bn, v in zip(bns, vs[0]):
      values[bn] = v
  return values

# Make a new DF with the sampled values from each field 
train_sampled = pd.DataFrame([process_train(fid) for fid in train_new['Field_ID'].values])

#MODEL
X = train_sampled.copy()
y = train_new['Yield'].values
print(X.head)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)
model=BaggingRegressor(CatBoostRegressor(silent=True),n_estimators=55)
model.fit(X_train, y_train)
print('Score:', mean_squared_error(y_test, model.predict(X_test), squared=False))

#SUBMITTING 
ss = pd.read_csv(folder+'/SampleSubmission.csv')
test_sampled = pd.DataFrame([process_test(fid) for fid in ss['Field_ID'].values])
preds = model.predict(test_sampled)
ss['Yield'] = preds
ss.to_csv(folder+'/Sub.csv', index=False)

