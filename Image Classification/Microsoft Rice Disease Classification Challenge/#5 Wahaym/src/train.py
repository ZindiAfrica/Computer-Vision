import argparse
import os
import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import timm
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from timm.data.mixup import Mixup
from torch.utils.data import ConcatDataset, DataLoader

from config import Config, load_config
from dataset import RiceDataset, load_df
from src.loss import *
from src.models import *
from src.utils import WarmupCosineLambda


def parse():
    parser = argparse.ArgumentParser(description="Training for RiceDiseases")
    parser.add_argument("--out_base_dir", default="result")
    parser.add_argument("--in_base_dir", default="input")
    parser.add_argument("--exp_name", default="tmp")
    parser.add_argument("--load_snapshot", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--config_path", default="config/debug.yaml")
    return parser.parse_args()


class RiceDataModule(LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        img_format: str,
        image_dir: str,
        fold: int,
        additional_dataset: RiceDataset = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.img_format = img_format
        self.image_dir = image_dir
        self.additional_dataset = additional_dataset
        if cfg.n_data != -1:
            df = df.iloc[: cfg.n_data]
        self.all_df = df
        if fold == -1:
            self.train_df = df
        else:
            skf = StratifiedKFold(n_splits=cfg.n_splits,
                                  shuffle=True, random_state=cfg.seed)
            train_idx, val_idx = list(skf.split(df, df.Label))[fold]
            self.train_df = df.iloc[train_idx].copy()
            self.val_df = df.iloc[val_idx].copy()

    def get_dataset(self, df, data_aug):
        return RiceDataset(df, self.cfg, self.image_dir, self.img_format, data_aug)

    def train_dataloader(self):
        dataset = self.get_dataset(self.train_df, True)
        if self.additional_dataset is not None:
            dataset = ConcatDataset([dataset, self.additional_dataset])
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.cfg.n_splits == -1:
            return None
        return DataLoader(
            self.get_dataset(self.val_df, False),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def all_dataloader(self):
        return DataLoader(
            self.get_dataset(self.all_df, False),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )


class DiseaseClassifier(LightningModule):
    def __init__(self, cfg: dict, pretrained=True):
        super().__init__()
        if not isinstance(cfg, Config):
            cfg = Config(cfg)
        self.save_hyperparameters(cfg)
        self.test_results_fp = None

        if cfg.model_type == "hybrid":
            self.model = HybridNet(cfg, pretrained)
        elif cfg.model_type == "dlog":
            self.model = Dlog(cfg, pretrained)
        elif cfg.model_type == "effnet":
            self.model = EffNet(cfg, pretrained)
        else:
            self.model = SimpleNet(cfg, pretrained)
        if cfg.loss == 'smooth':
            self.metric_crit = LabelSmoothingLoss(
                classes=cfg.num_classes, smoothing=cfg.smoothing)
            self.metric_crit_val = nn.CrossEntropyLoss(
                weight=None, reduction="mean")
        elif cfg.loss == 'focal':
            self.metric_crit = FocalLoss(gamma=cfg.focal_loss_gamma)
            self.metric_crit_val = nn.CrossEntropyLoss(
                weight=None, reduction="mean")
        elif cfg.loss == 'bince':
            self.metric_crit = nn.BCEWithLogitsLoss()
            self.metric_crit_val = nn.CrossEntropyLoss(
                weight=None, reduction="mean")
        elif cfg.loss == 'ce':
            self.metric_crit = nn.CrossEntropyLoss()
            self.metric_crit_val = nn.CrossEntropyLoss(
                weight=None, reduction="mean")
        if cfg.mixup is not None:
            mixup_args = dict(mixup_alpha=cfg.mixup["mixup"], cutmix_alpha=cfg.mixup["cutmix"], cutmix_minmax=cfg.mixup["cutmix_minmax"],
                              prob=cfg.mixup["mixup_prob"], switch_prob=cfg.mixup["mixup_switch_prob"], mode=cfg.mixup["mixup_mode"],
                              label_smoothing=cfg.mixup["smoothing"], num_classes=cfg.num_classes)
            self.mixup_fn = Mixup(**mixup_args)

    def forward(self, x):

        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_true = batch["image"], batch["label"]

        if self.hparams.mixup is not None:
            x, y_true = self.mixup_fn(x, y_true)
        logits = self(x)

        loss = loss_fn(self.metric_crit, y_true, logits)

        step = self.global_step*self.hparams.batch_size * \
            self.hparams.gradient_accumulation_steps
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        tb_dict = {'train_loss': loss,  'step': step}

        output = OrderedDict({
            'loss': loss,
            'log': tb_dict,
        })

        return output

    def validation_step(self, batch, batch_idx):
        x, y_true = batch["image"], batch["label"]

        out1 = self(x)
        out2 = self(x.flip(3))
        output = (out1 + out2) / 2
        y_pred = torch.nn.functional.softmax(output, 1)
        self.log_dict({"val_loss": self.metric_crit_val(output, y_true)},
                      on_step=False, on_epoch=True, logger=True)

        return y_pred

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        out1 = self(x)
        out2 = self(x.flip(3))
        pred_logit = ((out1 + out2) / 2).cpu()
        return {
            "original_index": batch["original_index"],
            "pred_logit": pred_logit,
            "label": batch["label"]
        }

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        outputs = self.all_gather(outputs)
        if self.trainer.global_rank == 0:
            epoch_results: Dict[str, np.ndarray] = {}
            for key in outputs[0].keys():
                if torch.cuda.device_count() > 1:
                    result = torch.cat(
                        [x[key] for x in outputs], dim=1).flatten(end_dim=1)
                else:
                    result = torch.cat([x[key] for x in outputs], dim=0)
                epoch_results[key] = result.detach().cpu().numpy()
            np.savez_compressed(self.test_results_fp, **epoch_results)

    def configure_optimizers(self):
        backbone_params = list(self.model.backbone.parameters())
        head_params = list(self.model.fc.parameters())

        params = [
            {"params": backbone_params, "lr": self.hparams.lr_backbone},
            {"params": head_params, "lr": self.hparams.lr_head},
        ]
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                params, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                params, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(
                params, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params,  momentum=0.9, nesterov=True, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler["method"] == "cosine":
            warmup_steps = self.hparams.max_epochs * self.hparams.warmup_steps_ratio
            cycle_steps = self.hparams.max_epochs - warmup_steps
            lr_lambda = WarmupCosineLambda(
                warmup_steps, cycle_steps, self.hparams.lr_decay_scale)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.hparams.scheduler["method"] == "plateau":
            base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.2, mode="min", patience=1, verbose=True, min_lr=1.0e-8)
            scheduler = {'scheduler': base_scheduler, 'interval': 'epoch',
                         'reduce_on_plateau': True, 'monitor': 'val_loss'}
        elif self.hparams.scheduler["method"] == "step":
            base_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.hparams.scheduler["step_size"], gamma=self.hparams.scheduler["gamma"], last_epoch=-1)
            scheduler = {'scheduler': base_scheduler, 'interval': 'epoch'}
        return [optimizer], [scheduler]


def train(
    df: pd.DataFrame,
    args: argparse.Namespace,
    cfg: Config,
    fold: int,
    do_inference: bool = False,
    additional_dataset: RiceDataset = None,
):

    out_dir = f"{args.out_base_dir}/{args.exp_name}/{fold}"
    model = DiseaseClassifier(cfg)
    data_module = RiceDataModule(
        df, cfg, cfg.img_format, f"{args.in_base_dir}/{cfg.img_dir}", fold, additional_dataset=additional_dataset
    )
    loggers = [pl_loggers.CSVLogger(out_dir)]
    callbacks = [LearningRateMonitor("epoch")]

    if args.save_checkpoint:
        callbacks.append(ModelCheckpoint(
            out_dir, save_last=True, save_top_k=0))
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=cfg["max_epochs"],
        logger=loggers,
        callbacks=callbacks,
        checkpoint_callback=args.save_checkpoint,
        precision=16,
        sync_batchnorm=True,
    )
    ckpt_path = f"{out_dir}/last.ckpt"
    if not os.path.exists(ckpt_path) or not args.load_snapshot:
        ckpt_path = None
    trainer.fit(model, ckpt_path=ckpt_path, datamodule=data_module)
    model.test_results_fp = f"{out_dir}/val_{cfg.img_format}_results.npz"
    trainer.test(model, data_module.val_dataloader())

    if do_inference:
        model.test_results_fp = f"{out_dir}/test_{cfg.img_format}_results.npz"
        df_test = load_df(args.in_base_dir, cfg,
                          "SampleSubmission.csv")
        test_data_module = RiceDataModule(
            df_test, cfg, cfg.img_format, f"{args.in_base_dir}/{cfg.img_dir}", -1)
        trainer.test(model, test_data_module.all_dataloader())


def main():
    args = parse()

    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    cfg = load_config(args.config_path, "config/default.yaml")
    print(cfg)
    if cfg.seed == -1:
        cfg.seed = np.random.randint(0, 1000000)
    print("Seed", cfg.seed)
    set_seed(cfg.seed)
    df = load_df(args.in_base_dir, cfg, "Train")
    pseudo_dataset = None
    if cfg.pseudo_label is not None:
        pseudo_df = load_df(args.in_base_dir, cfg, cfg.pseudo_label)
        pseudo_dataset = RiceDataset(
            pseudo_df[pseudo_df.conf >
                      cfg.pseudo_conf_threshold], cfg, f"{args.in_base_dir}/{cfg.img_dir}",  "rgb", True
        )
    if cfg["n_splits"] == -1:
        train(df, args, cfg, -1, do_inference=True,
              additional_dataset=pseudo_dataset)
    else:
        for f in range(cfg["n_splits"]):  # range(cfg["n_splits"]-3):
            train(df, args, cfg, f, do_inference=True,
                  additional_dataset=pseudo_dataset)


if __name__ == "__main__":
    main()
