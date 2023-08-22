from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)
from transformers.optimization import get_cosine_schedule_with_warmup
from warmup_scheduler import GradualWarmupScheduler


# --------------------------------------------------
# GradualWarmupSchedulerV2
# --------------------------------------------------
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler
        )

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]


# --------------------------------------------------
# getter
# --------------------------------------------------
def get_scheduler(optimizer, config, len_train_loader=0):
    scheduler_name = config.name
    if scheduler_name == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, **config.CosineAnnealingLR.params)
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestarts(
            optimizer, **config.CosineAnnealingWarmRestarts.params
        )
    elif scheduler_name == "cosine_schedule_with_warmup":
        num_training_steps = (
            config.cosine_schedule_with_warmup.params.max_epochs * len_train_loader
        )
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps
            / config.cosine_schedule_with_warmup.params.num_warmup_steps_factor,
            num_training_steps=num_training_steps,
        )
    elif scheduler_name == "GradualWarmupSchedulerV2":
        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=config.GradualWarmupSchedulerV2.params.total_epoch
            - config.GradualWarmupSchedulerV2.params.warmup_epoch,
            eta_min=config.GradualWarmupSchedulerV2.eta_min,
        )
        return GradualWarmupSchedulerV2(
            optimizer,
            multiplier=config.GradualWarmupSchedulerV2.params.warmup_factor,
            total_epoch=config.GradualWarmupSchedulerV2.params.total_epoch,
            after_scheduler=scheduler_cosine,
        )
    elif scheduler_name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, **config.ReduceLROnPlateau.params)
    else:
        raise ValueError(f"Not supported scheduler: {scheduler_name}.")
