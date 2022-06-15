##v17 original dataset

import pandas as pd, numpy as np
import sys, os, random, shutil,time  
from tqdm.auto import tqdm                                                                                                                                                    
from fastai.vision.all import *
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as skm
import timm


DATA_DIR = sys.argv[1] #'data'

N_SPLITS = 5
RANDOM_STATE = 1948

def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    try:
        dls.rng.seed(seed)
    except NameError:
        pass
      
fix_seed(RANDOM_STATE)

df_train = pd.read_csv(f'{DATA_DIR}/Train.csv')
df_test = pd.read_csv(f'{DATA_DIR}/Test.csv')
df_train['path'] = DATA_DIR+'/Images/'+df_train.img_IDS+'.jpg'
df_test['path'] = DATA_DIR+'/Images/'+df_test.img_IDS+'.jpg'
print(df_train.shape[0],df_test.shape[0])

hard=['ImageID_4RWF9LFI', 'ImageID_2CDKNXIT', 'ImageID_NC3CMZ3Q', 'ImageID_TXO0H454', 'ImageID_IZ36LCDF', 'ImageID_ESNB6PB2', 'ImageID_RJ6NSD31', 'ImageID_0JWKRKB2', 'ImageID_BYO9GTY4', 'ImageID_PZK3YPGK', 'ImageID_R1QIOHJK', 'ImageID_SP8Y5BEN', 'ImageID_LUS51Y3F', 'ImageID_ME1FCH3Q', 'ImageID_CZBK9Q7Q', 'ImageID_HL6A88H0', 'ImageID_KZM9QBI8', 'ImageID_5DNQQX0W', 'ImageID_S04NALW7', 'ImageID_1S0EL0PZ', 'ImageID_XYI2692O', 'ImageID_1NRD163V', 'ImageID_DWLTNPQB', 'ImageID_OFU85NTT', 'ImageID_AT9GTDEK', 'ImageID_G88I8RLC', 'ImageID_567HER96', 'ImageID_F2J2XWM9', 'ImageID_LGUHW2W7', 'ImageID_78PRIVA1', 'ImageID_N1FE792X', 'ImageID_4USMC1LO', 'ImageID_PM4VCKTG', 'ImageID_U5ZFK8FW', 'ImageID_916A2OCG', 'ImageID_HWGOU8HF', 'ImageID_DVKF7GVM', 'ImageID_JGU6TM68', 'ImageID_SPLFHVZM', 'ImageID_JEWEXI85', 'ImageID_N1HEC4RN', 'ImageID_A2EF4NXZ', 'ImageID_V4JLVXPR', 'ImageID_BB1KA2XL', 'ImageID_M3MCZG82', 'ImageID_4PSJSA0T', 'ImageID_AND2BBTY', 'ImageID_B7WV1LRV', 'ImageID_ESAOSPFK', 'ImageID_DRZPQNVU', 'ImageID_CSGAYC3C', 'ImageID_14TGAY37', 'ImageID_NWD9NEBV']
relabels={'ImageID_5YSMJUI6': 'Mosque', 'ImageID_0TKQDJKJ': 'Mosque', 'ImageID_J7I0YTO0': 'Mosque', 'ImageID_AI50JYFL': 'Mosque', 'ImageID_0R1JN7AC': 'You', 'ImageID_W9OPURXL': 'Church', 'ImageID_DMA4OTN2': 'Church', 'ImageID_WLB1RUMX': 'Mosque', 'ImageID_KQ5Y9URW': 'Mosque', 'ImageID_NBGWJNCE': 'Enough/Satisfied', 'ImageID_2OK3T60A': 'Church', 'ImageID_SI9XXLPJ': 'Mosque', 'ImageID_2EY0HV23': 'Mosque', 'ImageID_2SCX7Z3S': 'Mosque', 'ImageID_6VMTI1EJ': 'Mosque', 'ImageID_VSZF93W9': 'Mosque', 'ImageID_X2TFUPVD': 'Church', 'ImageID_I4F7ODR8': 'Mosque', 'ImageID_JLKXCT7V': 'Church', 'ImageID_NX8DEEOB': 'Mosque', 'ImageID_HV7LCCQN': 'You', 'ImageID_YE1WRYCD': 'Mosque', 'ImageID_PW62ES5Z': 'Mosque', 'ImageID_J6LS2SQV': 'Mosque', 'ImageID_77FIKBDD': 'Enough/Satisfied', 'ImageID_Z9R5974U': 'Mosque', 'ImageID_0893LRQN': 'Mosque', 'ImageID_CKV4NWZT': 'Mosque', 'ImageID_44IZ9AMM': 'Mosque', 'ImageID_39VDKR2I': 'Mosque', 'ImageID_2QJPC70S': 'Temple', 'ImageID_U5TG4J5C': 'Mosque', 'ImageID_ECB3B12X': 'Mosque', 'ImageID_I8GAF61T': 'Church', 'ImageID_1OCXJ34Y': 'Mosque', 'ImageID_0C7FSZ03': 'You', 'ImageID_HB7CEIP5': 'Church', 'ImageID_KQKFLP8X': 'Mosque', 'ImageID_FKHHDWH5': 'You', 'ImageID_NILYX14J': 'Mosque', 'ImageID_IUZW3SZ3': 'Mosque', 'ImageID_115ZH0AT': 'Mosque', 'ImageID_SRISKMLL': 'Church', 'ImageID_1JVQDHLV': 'Church', 'ImageID_DB9ZNG3E': 'Enough/Satisfied', 'ImageID_3COIPDH2': 'Seat', 'ImageID_4BW25QML': 'Friend', 'ImageID_YN6X4FJB': 'Church', 'ImageID_DAN12R0K': 'Seat', 'ImageID_5EOFFKDH': 'Enough/Satisfied', 'ImageID_MU851JXT': 'Mosque', 'ImageID_XAPEY2YL': 'Mosque', 'ImageID_BXWESWG5': 'Mosque', 'ImageID_TSDTGVSP': 'Church', 'ImageID_ZJME63GP': 'Me'}

unkv2=['ImageID_7284IFK4', 'ImageID_29X0RTMO', 'ImageID_SKGYBKI4', 'ImageID_SHE4L9M5', 'ImageID_ZEDCL5S6', 'ImageID_SOXPPFY7', 'ImageID_AB0WLXTQ', 'ImageID_WK885JRM', 'ImageID_8DG4BZVU', 'ImageID_3LWI6IA6', 'ImageID_K2C3NJYV', 'ImageID_DKH9T5LV', 'ImageID_PC16QYTV', 'ImageID_XUXIYJII', 'ImageID_VJPPH215', 'ImageID_HQDPRBI0', 'ImageID_QALYD6AQ', 'ImageID_EX8EVQ7A', 'ImageID_CFLDK4S2', 'ImageID_XJG9CFXX', 'ImageID_HY2MYAXO', 'ImageID_HEA0X15D', 'ImageID_MYXZT7PW', 'ImageID_J49XESA8', 'ImageID_W3SGXK8K', 'ImageID_WMETEGZQ', 'ImageID_NDHNF3UO', 'ImageID_0DYM84MH', 'ImageID_WY7ECZNS', 'ImageID_SFIGFOYT', 'ImageID_2CIWXSFB', 'ImageID_5FDDPJL1', 'ImageID_FEFS5R1Q', 'ImageID_57RH69VM', 'ImageID_LHENI058', 'ImageID_79S7PKIE', 'ImageID_0A9XXFWR', 'ImageID_6IWDCR84', 'ImageID_D4KRNJ7O', 'ImageID_HJPNWHLO', 'ImageID_5MCMKNHT', 'ImageID_D0LSOBJZ', 'ImageID_NQ5X2D3L', 'ImageID_7HGUT7K6', 'ImageID_RM5Q91K8', 'ImageID_TUE1CAYK', 'ImageID_L3XB0109']
relabelsv2={'ImageID_U9P07SMT': 'Mosque', 'ImageID_TZIEJZGY': 'Mosque', 'ImageID_U3SPAU5Q': 'Church', 'ImageID_WAVQ0V61': 'Mosque'}
hardv2=['ImageID_MGLTX0Y0', 'ImageID_VFESLNFO', 'ImageID_774FYZIW', 'ImageID_EYNZZ1I1', 'ImageID_20EMNL7G', 'ImageID_1T1I2ZGG', 'ImageID_B5WLWFPY', 'ImageID_ZWQV9POB', 'ImageID_16GQVRKU', 'ImageID_9F29Y3IG', 'ImageID_T8UQI4GU', 'ImageID_EBV27Y5P', 'ImageID_VDKWJYNE', 'ImageID_4X89UAWO', 'ImageID_7YLAAGEN', 'ImageID_N8F9Y28D', 'ImageID_YH6DPYV6', 'ImageID_8QEYQ70E', 'ImageID_MBAKX8OC', 'ImageID_R6GM0XS0', 'ImageID_BS1G2HYD', 'ImageID_KM63Q2O6', 'ImageID_XG6GY82B', 'ImageID_LU4013XR', 'ImageID_TXHRX9EI', 'ImageID_32UZELCO', 'ImageID_72BCVI3A', 'ImageID_EWHQ26EV', 'ImageID_ICGK602T', 'ImageID_LGTMYR8S', 'ImageID_UENT3UDW', 'ImageID_1W1XC6IZ', 'ImageID_WB8AIGOF', 'ImageID_EILC4VD0', 'ImageID_7XFH50C2', 'ImageID_SCPSMEZR', 'ImageID_8YRAETTK', 'ImageID_Y4Y0XFX0', 'ImageID_FGG5K8Q3', 'ImageID_L6DF7L4T', 'ImageID_G6HRMIIJ', 'ImageID_8UZ18E4F', 'ImageID_TMLXRMAI', 'ImageID_H0DZW9R5', 'ImageID_X1VX6VI1', 'ImageID_LW7CYKZL', 'ImageID_R6VIZ8AN', 'ImageID_8ZYAVLW6', 'ImageID_G87FKBQ0', 'ImageID_13Y51PQI', 'ImageID_WA59WTSP', 'ImageID_OTF5A3UJ', 'ImageID_9V0N5NHL', 'ImageID_2S9C9XC5', 'ImageID_TCCO0UU2', 'ImageID_NEMTW35D', 'ImageID_NRTP0S7K', 'ImageID_SW0JF9OR', 'ImageID_O3JMHKBC', 'ImageID_L3ZSA4QV', 'ImageID_VQO646N2', 'ImageID_PRFOVKSS']
hardv3=['ImageID_894KLHCZ', 'ImageID_UQ93XEM0', 'ImageID_SK9EQ6YV', 'ImageID_X5W0OX0W', 'ImageID_1YE5BXZ9', 'ImageID_2QGNJFAF', 'ImageID_DWQ0VTU7', 'ImageID_4VEZYWGH', 'ImageID_XT1PFNWN', 'ImageID_JMOSC2LJ', 'ImageID_WY0G6I1C', 'ImageID_MR3GLBRR', 'ImageID_AKSO2IW0', 'ImageID_CX5OD6JU', 'ImageID_7MZY8XIG', 'ImageID_1H7FK7XA', 'ImageID_R37KFSRM', 'ImageID_GUEX83ZJ', 'ImageID_TH2QSCMN', 'ImageID_FRH4WPHY', 'ImageID_1TZIMSZP', 'ImageID_3DB710Q6', 'ImageID_MUQ9MKHE', 'ImageID_BRI1AGQF', 'ImageID_5NBO7QO4', 'ImageID_MS6IR7SR', 'ImageID_WVXBFVV3', 'ImageID_K16E8MPL', 'ImageID_6CLGYDCX', 'ImageID_4E28RJGO']
hardv4 = ['ImageID_4RKU1XR8', 'ImageID_SLAMKIWY', 'ImageID_PLSTS5BX', 'ImageID_NBGWJNCE', 'ImageID_X2TFUPVD', 'ImageID_CW871NYN', 'ImageID_PW62ES5Z', 'ImageID_9I7RQ6TO', 'ImageID_28UZL0RP', 'ImageID_J21HQ1LK', 'ImageID_605OCFVM', 'ImageID_Q80L0PZT', 'ImageID_I8GAF61T', 'ImageID_F0ELMTF9', 'ImageID_Q4O1T3OE', 'ImageID_4VTAZ2D5', 'ImageID_FUW25XHX', 'ImageID_BXWESWG5']

hard = list(set(hard+hardv2+hardv3+hardv4+unkv2))
print('hard samples ',len(hard))


TARGET_CLASSES = sorted(df_train.Label.unique())
CODES = OrderedDict({ TARGET_CLASSES[i] : i  for i in range(0, len(TARGET_CLASSES) ) })
df_train['code'] = df_train.Label.map(lambda x: CODES[x])
def get_dls(df,fold_id,bs,size,max_size):
    fix_seed(RANDOM_STATE)
    skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    df['fold'] = -1
    for i,(train_index, val_index) in enumerate(skf.split(df.path,df.Label)):
        df.loc[val_index,'fold'] = i
        
    df['is_valid'] = False
    df.loc[df.fold==fold_id,'is_valid']=True
    
    ##keep hard samples in train
    df.loc[df.img_IDS.isin(hard),'fold'] = -1
    df.loc[df.img_IDS.isin(hard),'is_valid'] = False


    dls = ImageDataLoaders.from_df(df[['path','code','is_valid']],'.',
                               valid_col='is_valid',
                                item_tfms = [Resize(max_size,method='squish', pad_mode='zeros')],
                               batch_tfms=[
                                          *aug_transforms(size=size,min_scale=0.8,
                                        do_flip=True,
                                        flip_vert=False,
                                        p_affine=0.75,
                                        p_lighting=0.75,
                                        mult=1.0,xtra_tfms=None,mode='bilinear',pad_mode='zeros'
                                        ),
                                          Normalize.from_stats(*imagenet_stats)
                                          ],
                              bs=bs,
                              seed = RANDOM_STATE
                              )
    return df,dls
  
def get_model(arch,pretrained=True):
    model = timm.create_model(arch, num_classes=len(TARGET_CLASSES),pretrained=pretrained)
    return model

ss = pd.read_csv(f'{DATA_DIR}/SampleSubmission.csv')
print(f'n test {df_test.shape[0]}')
PREDS = []
PREDS_TTA = []
PREDS_TEST=[]
PREDS_TEST_TTA = []
OOFS = []

for fold_id in range(N_SPLITS):
# for fold_id in range(1):
    print(f'running inference for fold {fold_id}')
    fix_seed(RANDOM_STATE)
    size=224;max_size=384
    df,dls = get_dls(df_train,fold_id,bs=32,size=size,max_size=max_size)
    print(df[df.is_valid].shape,df[~df.is_valid].shape)

    arch = 'swin_large_patch4_window7_224_in22k'
    MODEL_DIR = f'model/R_{RANDOM_STATE}_F{fold_id}'
    print(MODEL_DIR)
    os.makedirs(MODEL_DIR,exist_ok=True)
    model = get_model(arch,pretrained=False)
    learn = Learner(dls, model, cbs=[BnFreeze,
                                     CutMix(1.),
                                     SaveModelCallback(monitor='valid_loss')],metrics=[accuracy])

    learn.model_dir = MODEL_DIR
    model_fn = f'{MODEL_DIR}/model.pth'
    learn.model.load_state_dict(torch.load(model_fn))
    with learn.no_bar():
        df_val = df[df.is_valid==1].copy()
        preds = learn.get_preds(ds_idx=1)[0].numpy()
        preds_tta = learn.tta(ds_idx=1)[0].numpy()
        loss,acc = skm.log_loss(df_val.code,preds),skm.accuracy_score(df_val.code,np.argmax(preds,axis=1))
        loss_t,acc_t = skm.log_loss(df_val.code,preds_tta),skm.accuracy_score(df_val.code,np.argmax(preds_tta,axis=1))
        print('fold ',fold_id, 'loss: ',loss, 'loss_tta:', loss_t)
        dl_test = learn.dls.test_dl(df_test['path'].values)
        preds_test = learn.get_preds(dl=dl_test)[0].numpy()
        preds_test_tta = learn.tta(dl=dl_test)[0].numpy() 
    
    OOFS.append(df_val)
    PREDS.append(preds)
    PREDS_TTA.append(preds_tta)
    PREDS_TEST.append(preds_test)
    PREDS_TEST_TTA.append(preds_test_tta)

sub_id = 'submission'
results_dir = f'results/res_R{RANDOM_STATE}'
os.makedirs(results_dir,exist_ok=True)

df_oof = pd.concat(OOFS)
df_oof.to_csv(f'{results_dir}/oof.csv',index=False)

preds = np.concatenate(PREDS)
preds_tta = np.concatenate(PREDS_TTA)
loss,acc = skm.log_loss(df_oof.code,preds, labels=[i for i in range(len(TARGET_CLASSES))]),skm.accuracy_score(df_oof.code,np.argmax(preds,axis=1))
loss_t,acc_t = skm.log_loss(df_oof.code,preds_tta, labels=[i for i in range(len(TARGET_CLASSES))]),skm.accuracy_score(df_oof.code,np.argmax(preds_tta,axis=1))
print(f'OOF loss: {loss} acc: {acc} loss_tta: {loss_t} acc_tta: {acc_t}')
np.savez_compressed(f'{results_dir}/preds_oof.npz',np.array(PREDS))
np.savez_compressed(f'{results_dir}/preds_oof_tta.npz',np.array(PREDS_TTA))

preds_test = np.mean(PREDS_TEST,axis=0)
preds_test_tta = np.mean(PREDS_TEST_TTA,axis=0)
print('test predictions: ', preds_test_tta.shape)
np.savez_compressed(f'{results_dir}/preds_test.npz',np.array(PREDS_TEST))
np.savez_compressed(f'{results_dir}/preds_test_tta.npz',np.array(PREDS_TEST_TTA))

df_preds = pd.DataFrame(preds_test,columns=CODES.keys())
df_preds['img_IDS'] = df_test.img_IDS.values
df_preds = df_preds[ss.columns]
df_preds.to_csv(f'{results_dir}/{sub_id}.csv',index=False)

df_preds = pd.DataFrame(preds_test_tta,columns=CODES.keys())
df_preds['img_IDS'] = df_test.img_IDS.values
df_preds = df_preds[ss.columns]
df_preds.to_csv(f'{results_dir}/{sub_id}_tta.csv',index=False)
