import argparse
import os

import pandas as pd
from hydra import compose, initialize


def preprocess(df, config):
    df["image_path"] = df[config.dataset.id].apply(
        lambda x: os.path.join(
            config.preprocess.base_dir, config.preprocess.test_image_dir, x
        )
    )

    return df


def main(args):
    with initialize(config_path="configs", job_name="config"):
        config = compose(config_name=args.config_name)

    df = pd.read_csv(
        os.path.join(config.preprocess.base_dir, config.preprocess.test_csv)
    )
    df = preprocess(df, config)
    df.to_csv(config.dataset.test_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
