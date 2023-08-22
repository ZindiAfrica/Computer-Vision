import numpy as np, pandas as pd
from tqdm.auto import tqdm
import sys, os, random, shutil,time,argparse, glob
from tqdm.auto import tqdm                                                                                                                                                    
import timm
import matplotlib.pyplot as plt
import cv2, PIL                                                                                                                                                 
from fastai.vision.all import *
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as skm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_DIR = sys.argv[1]#'/data'
MODEL_DIR = 'models'
SUB_ID = 'Submission'

N_SPLITS = 5
NUM_CLASSES = 3
RANDOM_STATE = 384
mean=(0.485, 0.456, 0.406); std=(0.229, 0.224, 0.225)
hard_s6 = ['id_t0wzm5u7b9.jpg', 'id_m65pg6tge7.jpg', 'id_20idcnwgt2.jpg', 'id_sj6i04jn5y.jpg', 'id_1z4xx5omh5.jpg', 'id_gp7qwbrayd.jpg', 'id_jo0rzddfg8.jpg', 'id_yv6242snxj.jpg', 'id_xck8ldqalq.jpg', 'id_35q6ubuf8u.jpg', 'id_ll7vhrwcwt.jpg', 'id_zja13qs1gz.jpg', 'id_1qc0fuh6qi.jpg', 'id_zz6gzk7p97.jpg', 'id_euzlag4tge.jpg', 'id_rzbqo99wh8.jpg', 'id_fwjmvxf7sb.jpg', 'id_807k552gkl.jpg', 'id_kytvg5jp8m.jpg', 'id_9e5pkcqhbr.jpg', 'id_zc7jyiweh2.jpg', 'id_xtq70h3oay.jpg', 'id_4yvkffuexi.jpg', 'id_sx89tzx918.jpg', 'id_n1r041bxq1.jpg', 'id_cg8j9szr2h.jpg', 'id_tlpg7dz8g4.jpg', 'id_c8sl86v71p.jpg', 'id_gyfpd9go3a.jpg', 'id_pbi4y230pj.jpg', 'id_uql279c07s.jpg', 'id_rqqkvc20ri.jpg', 'id_a6cdnzuc2w.jpg', 'id_4aiotezgvc.jpg', 'id_uu598mdsec.jpg', 'id_lnc6zoay5e.jpg', 'id_7n3trma97d.jpg', 'id_go188iml0i.jpg', 'id_ppu159bbts.jpg', 'id_9pl1vuty9t.jpg', 'id_u8xc5omqao.jpg', 'id_epn3kdoyws.jpg', 'id_2e67u0oabg.jpg', 'id_nrulqwvzc0.jpg', 'id_5k0gr2r8nj.jpg', 'id_m3ukgm8ks6.jpg', 'id_hvztaeci31.jpg', 'id_gzg9j24fpd.jpg', 'id_g0m3edrhhv.jpg', 'id_zil68oveco.jpg', 'id_25ghc3to5j.jpg', 'id_780vcduwq1.jpg', 'id_01z6i8am9b.jpg', 'id_olbfbcwm9i.jpg']
hard_s12 = ['id_2k0n8ihd2n.jpg', 'id_34iwgxr20a.jpg', 'id_4gd6uw5eeh.jpg', 'id_5ao31as6ct.jpg', 'id_5z7hi23b4r.jpg', 'id_61qkwobnsw.jpg', 'id_6cwdxo8a5z.jpg', 'id_94sw9ihvyv.jpg', 'id_9qofsjzy3t.jpg', 'id_e0grd1jf78.jpg', 'id_icjw1lihy8.jpg', 'id_joinlebqmb.jpg', 'id_lqwpbdb49p.jpg', 'id_muxr52g4fz.jpg', 'id_nqwmljip6u.jpg', 'id_ns3w6150w0.jpg', 'id_ob0maoob4f.jpg', 'id_p9ov7g5bpc.jpg', 'id_pw13vxbdk7.jpg', 'id_rel36t55vr.jpg', 'id_sde4pcooi0.jpg', 'id_taxrsylqp7.jpg', 'id_tf2f362hp9.jpg', 'id_tx4mlgrah4.jpg', 'id_ueazifzqgs.jpg', 'id_vov5gyqf7l.jpg', 'id_wvhx1h7wp5.jpg']
hard = hard_s6+hard_s12


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        dls.rng.seed(seed)
    except NameError:
        pass

fix_seed(RANDOM_STATE)

df_train = pd.read_csv(f'{DATA_DIR}/Train.csv')
df_train = df_train[~df_train.Image_id.str.contains('_rgn')].reset_index(drop=True)
df_train['path0'] = DATA_DIR + '/Images/' + df_train.Image_id
df_train['path1'] = df_train.path0.str.split('.',expand=True)[0]+'_rgn.jpg'

TARGET_CLASSES = ['blast', 'brown', 'healthy']
CODES = OrderedDict({ TARGET_CLASSES[i] : i  for i in range(0, len(TARGET_CLASSES) ) })
df_train['code'] = df_train.Label.map(lambda x: CODES[x])



def get_dls(df,bs,size,max_size,random_state,fold):
    skf  = StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=random_state)
    df['fold'] = -1
    for fold_id, (train_index, test_index) in enumerate(skf.split(df, df.Label)):
        df.loc[test_index,'fold'] = fold_id
    
    df['is_valid'] = False
    df.loc[df.fold==fold,'is_valid']=True
    df[df.is_valid].shape[0],df[~df.is_valid].shape[0]
    
    df.loc[df.Image_id.isin(hard),'is_valid']=False
    #df = df[~df.Image_id.isin(todel)]
    dls = ImageDataLoaders.from_df(df[['path0','Label','is_valid']],path='/',
                               valid_col='is_valid',
                                item_tfms = [
                                    #  Resize(max_size,method='squish', pad_mode='zeros')
                                      RandomResizedCrop(max_size,min_scale=0.75, ratio=(0.1, 10),val_xtra=0.,)
                                ],
                               batch_tfms=[
                                          *aug_transforms(size=size,min_scale=1,
                                        do_flip=True,
                                        flip_vert=False,
                                        max_rotate=10.0,
                                        min_zoom=1,
                                        max_zoom=1.1,
                                        max_lighting=0.2,
                                        max_warp=0.2,
                                        p_affine=0.75,
                                        p_lighting=0.75,
                                        mult=1.0,xtra_tfms=None,mode='bilinear',pad_mode='zeros'
                                        ),
                                          Normalize.from_stats(mean,std)
                                          ],
                              bs=bs,
                              seed = RANDOM_STATE
                              )
    return df, dls


df_test = pd.read_csv(f'{DATA_DIR}/Test.csv')
df_test = df_test[~df_test.Image_id.str.contains('_rgn')] 
df_test['Image_id'] = DATA_DIR + '/Images/' + df_test.Image_id

PREDS = []
PREDS_TEST=[]
OOFS = []
arch = 'swin_large_patch4_window12_384'
batch_size = 10
size=384; max_size=384

random_state = 384

MODEL_DIR = 'models'
for fold_id in range(N_SPLITS):
    fix_seed(random_state)
    df, dls = get_dls(df_train,bs=batch_size,size=size,max_size=max_size,random_state=random_state,fold=fold_id)
    model = timm.create_model(arch, num_classes=NUM_CLASSES,pretrained=False)
    learn = Learner(dls,model)

    model_fn = f'{MODEL_DIR}/{arch}_f{fold_id}.pth'
    learn.model.load_state_dict(torch.load(model_fn))

    df_val = df[df.is_valid==1].copy()
    preds = learn.get_preds(ds_idx=1)[0].numpy()
    
    loss,acc = skm.log_loss(df_val.code,preds, labels=[i for i in range(NUM_CLASSES)]),skm.accuracy_score(df_val.code,np.argmax(preds,axis=1))
    print('fold ',fold_id, 'loss: ',loss, 'acc: ',acc,)
    dl_test = dls.test_dl(df_test)
    
    preds_test = learn.get_preds(dl=dl_test)[0].numpy()

    OOFS.append(df_val)
    PREDS.append(preds)
    PREDS_TEST.append(preds_test)

os.makedirs(SUB_ID)
df_oof = pd.concat(OOFS)
preds = np.concatenate(PREDS)
df_oof.to_csv(f'{SUB_ID}/oof.csv')
np.savez_compressed(f'{SUB_ID}/preds.npz',preds)

loss,acc = skm.log_loss(df_oof.code,preds, labels=[i for i in range(len(TARGET_CLASSES))]),skm.accuracy_score(df_oof.code,np.argmax(preds,axis=1))
print('loss: ',loss, 'acc: ',acc,)

preds_test = np.min(PREDS_TEST,axis=0)
np.savez_compressed(f'{SUB_ID}/preds_test.npz',preds_test)

ss = pd.read_csv(f'{DATA_DIR}/SampleSubmission.csv')
df_preds = pd.DataFrame(preds_test,columns=CODES.keys())
df_preds['Image_id'] = df_test.Image_id.str.split('/').str[-1].values
df_preds = df_preds[ss.columns]
df_preds.to_csv(f'{SUB_ID}/{SUB_ID}.csv',index=False)

