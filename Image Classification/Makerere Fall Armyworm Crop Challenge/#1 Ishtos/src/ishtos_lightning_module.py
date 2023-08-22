import numpy as np
import torch
from ishtos_losses import get_losses
from ishtos_metrics import get_metrics
from ishtos_models import get_model
from ishtos_optimizers import get_optimizer
from ishtos_schedulers import get_scheduler
from pytorch_lightning import LightningModule


class MyLightningModule(LightningModule):
    def __init__(self, config, fold=0, len_train_loader=0):
        super(MyLightningModule, self).__init__()
        self.save_hyperparameters(config)

        self.config = config
        self.fold = fold
        self.len_train_loader = len_train_loader
        self.model = get_model(config.model)
        self.losses = get_losses(config.loss)
        self.metrics = get_metrics(config.metric)

    def configure_optimizers(self):
        optimizer = get_optimizer(
            parameters=self.model.parameters(), config=self.config.optimizer
        )
        scheduler = get_scheduler(
            optimizer=optimizer,
            config=self.config.scheduler,
            len_train_loader=self.len_train_loader,
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": self.config.scheduler.interval,
            "monitor": self.config.scheduler.monitor,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }

    def training_step(self, batch, batch_idx):
        loss, preds, target = self.__step(batch, "train")
        return {"loss": loss, "preds": preds, "target": target}

    def validation_step(self, batch, batch_idx):
        loss, preds, target = self.__step(batch, "val")
        return {"loss": loss, "preds": preds, "target": target}

    def __step(self, batch, phase):
        images, target = batch
        if do_mixup(phase, self.current_epoch, self.config):
            images, target_a, target_b, lam = mixup_data(images, target)
            logits = self.model(images).squeeze(1)
            for i, loss_weight, loss_func in enumerate(self.losses):
                if i == 0:
                    loss = loss_weight * mixup_loss(
                        loss_func, logits, target_a, target_b, lam
                    )
                else:
                    loss += loss_weight * mixup_loss(
                        loss_func, logits, target_a, target_b, lam
                    )
        else:
            logits = self.model(images).squeeze(1)
            for i, (loss_weight, loss_func) in enumerate(self.losses):
                if i == 0:
                    loss = loss_weight * loss_func(logits, target)
                else:
                    loss += loss_weight * loss_func(logits, target)

        preds = logits.softmax(dim=1).detach().cpu()
        target = target.detach().cpu()
        return loss, preds, target

    def training_epoch_end(self, outputs):
        for i, loss_name in enumerate(self.config.loss.names):
            if loss_name == "OUSMLoss":
                self.losses[i][1].update()  # TODO: refactor
        self.__epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.__epoch_end(outputs, "val")

    def __epoch_end(self, outputs, phase):
        d = dict()
        loss = []
        preds = []
        target = []
        for output in outputs:
            loss.append(output["loss"])
            preds.append(output["preds"])
            target.append(output["target"])
        preds = torch.cat(preds)
        target = torch.cat(target)
        loss = torch.stack(loss).mean().item()

        d[f"{phase}_loss"] = loss
        for i, (metric_name, metric_func) in enumerate(self.metrics):
            if i == 0:
                d[f"{phase}_score"] = metric_func(preds=preds, target=target)
            d[f"{phase}_{metric_name}"] = metric_func(preds=preds, target=target)
        self.log_dict(d, prog_bar=True, logger=True, on_step=False, on_epoch=True)


def do_mixup(phase, current_epoch, config):
    return (
        phase == "train"
        and config.train.mixup.enable
        and np.random.rand() < config.train.mixup.p
        and current_epoch + config.train.mixup.duration < config.trainer.max_epochs
    )


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_loss(loss, pred, y_a, y_b, lam):
    return lam * loss(pred, y_a) + (1 - lam) * loss(pred, y_b)
