import pandas as pd, numpy as np
import sys, os, random, shutil,time  
from tqdm.auto import tqdm                                                                                                                                                    
from fastai.vision.all import *
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as skm

DATA_DIR = sys.argv[1] #'data'
df_train = pd.read_csv(f'{DATA_DIR}/Train.csv')
df_test = pd.read_csv(f'{DATA_DIR}/Test.csv')
df_train['path'] = DATA_DIR+'/Images/'+df_train.img_IDS+'.jpg'
df_test['path'] = DATA_DIR+'/Images/'+df_test.img_IDS+'.jpg'

print(df_train.shape[0],df_test.shape[0])

TARGET_CLASSES = sorted(df_train.Label.unique())
CODES = OrderedDict({ TARGET_CLASSES[i] : i  for i in range(0, len(TARGET_CLASSES) ) })
df_train['code'] = df_train.Label.map(lambda x: CODES[x])
ss = pd.read_csv(f'{DATA_DIR}/SampleSubmission.csv')
PREDS=[]
OOFS = []
PREDS_TEST=[]
PREDS_TEST_TTA=[]
dfs_oof = []
#use tta version for 1948
RANDOM_STATE=1948
dir = f'results/res_R{RANDOM_STATE}'
d = pd.read_csv(f'{dir}/oof.csv')
dfs_oof.append(d)
preds_oof = np.load(f'{dir}/preds_oof_tta.npz',allow_pickle=True)['arr_0']
preds_oof = np.concatenate(preds_oof,axis=0)
PREDS.append(preds_oof)
preds_test_tta = np.load(f'{dir}/preds_test_tta.npz',allow_pickle=True)['arr_0']
PREDS_TEST_TTA.append(preds_test_tta)

#use non-tta for other models
for RANDOM_STATE in [42,7,11,888,999]:
  dir = f'results/res_R{RANDOM_STATE}'
  d = pd.read_csv(f'{dir}/oof.csv')
  dfs_oof.append(d)
  preds_oof = np.load(f'{dir}/preds_oof.npz',allow_pickle=True)['arr_0']
  preds_oof = np.concatenate(preds_oof,axis=0)
  PREDS.append(preds_oof)
  preds_test_tta = np.load(f'{dir}/preds_test.npz',allow_pickle=True)['arr_0'] 
  PREDS_TEST_TTA.append(preds_test_tta)

df_oof = pd.concat(dfs_oof)
preds_oof = np.concatenate(PREDS)
print(preds_oof.shape)
df_oof['pred'] = np.argmax(preds_oof,axis=1)
df_oof['pred_label']=df_oof['pred'].map({v:k for k,v in CODES.items()})
print('loss:', skm.log_loss(df_oof.code,preds_oof),'acc: ',skm.accuracy_score(df_oof.code,df_oof.pred),)

sub_id = 'Submission'
preds_test_tta = np.concatenate(PREDS_TEST_TTA).mean(axis=0)

df_preds = pd.DataFrame(preds_test_tta,columns=CODES.keys())
df_preds['img_IDS'] = df_test.img_IDS.values
df_preds = df_preds[ss.columns]
df_preds.to_csv(f'{sub_id}.csv',index=False)