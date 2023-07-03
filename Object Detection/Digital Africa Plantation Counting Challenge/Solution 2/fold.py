from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd

def create_folds(df, n_s=5, n_grp=None, config=None):
    df['kfold'] = -1
    if n_grp is None:
        skf = KFold(n_splits=n_s, random_state=config['seed'])
        target = df['Target']
    else:
        skf = StratifiedKFold(n_splits=n_s, shuffle=False)
        df['grp'] = pd.cut(df['Target'], n_grp, labels=False)
        target = df.grp
    for fold_no, (t, v) in enumerate(skf.split(target, target)):
        df.loc[v, 'kfold'] = fold_no

    df = df.drop('grp', axis=1)
    
    return df