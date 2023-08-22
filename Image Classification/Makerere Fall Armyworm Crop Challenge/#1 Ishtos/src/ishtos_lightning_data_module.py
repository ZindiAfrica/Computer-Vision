import os

import pandas as pd
from ishtos_datasets import get_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class MyLightningDataModule(LightningDataModule):
    def __init__(self, config, fold=0):
        super(MyLightningDataModule, self).__init__()
        self.config = config
        self.fold = fold

    def _split_train_and_valid_df(self):
        df = pd.read_csv(self.config.dataset.train_csv)

        train_df = df[df["fold"] != self.fold].reset_index(drop=True)
        valid_df = df[df["fold"] == self.fold].reset_index(drop=True)

        return train_df, valid_df

    def setup(self, stage):
        self.train_df, self.valid_df = self._split_train_and_valid_df()

    def _get_dataframe(self, phase):
        if phase == "train":
            return self.train_df
        elif phase == "valid":
            return self.valid_df
        elif phase == "test":
            test_df = pd.read_csv(
                os.path.join(self.config.dataset.base_dir, self.config.dataset.test_csv)
            )
            return test_df

    def _get_dataset(self, phase):
        df = self._get_dataframe(phase)
        return get_dataset(self.config, df, phase)

    def _get_dataloader(self, phase):
        dataset = self._get_dataset(phase=phase)
        return DataLoader(
            dataset,
            batch_size=self.config.dataset.loader.batch_size,
            num_workers=self.config.dataset.loader.num_workers,
            shuffle=phase == "train",
            drop_last=phase == "train",
            pin_memory=True,
        )

    def len_dataloader(self, phase):  # TODO: refactor
        return len(self._get_dataloader(phase=phase))

    def train_dataloader(self):
        return self._get_dataloader(phase="train")

    def val_dataloader(self):
        return self._get_dataloader(phase="valid")

    def test_dataloader(self):
        return self._get_dataloader(phase="test")

    def predict_dataloader(self):
        return self._get_dataloader(phase="test")
