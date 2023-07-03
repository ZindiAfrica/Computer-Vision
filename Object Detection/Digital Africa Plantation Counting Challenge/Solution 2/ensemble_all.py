import os
import pandas as pd
import yaml
from argparse import ArgumentParser
from loguru import logger
import numpy as np

parser = ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, required=False, default="configs/tf_efficientnetv2_m.yaml"
)
parser.add_argument(
    "-f", "--fold", type=int, required=False, default=0
)
args = parser.parse_args()
logger.info("Loading config")
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

sub_df = pd.read_csv(f"{config['root_dir']}/SampleSubmission.csv")
logits = []
df_subs = os.listdir('results')
for df_id in df_subs:
    sub = pd.read_csv(f'results/{df_id}')
    logits.append(np.array(sub['Target'].values.tolist()))

preds = np.mean(logits, 0)
preds[preds < 0] = 0

sub_df['Target'] = preds
os.makedirs('submission', exist_ok=True)
sub_df.to_csv(f"submission/submission_ensemble_final.csv", index=False)