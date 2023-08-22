import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import NLLLoss


# --------------------------------------------------
# FocalLoss
# --------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean", eps=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1.0 - self.eps)
        loss = F.nll_loss(torch.log(logit), target, reduction="none")
        loss = (
            loss
            * (1 - logit.gather(1, target.view((target.shape[0], 1))).squeeze(1))
            ** self.gamma
        )

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")


# --------------------------------------------------
# NLLLoss
# --------------------------------------------------
class NLLoss(nn.Module):
    def __init__(self, reduction="mena"):
        super(NLLoss, self).__init__()
        self.criterion = nn.NLLLoss(reduction=reduction)

    def forward(self, input, target):
        logit = torch.log_softmax(input, dim=1)
        loss = self.criterion(logit, target)
        return loss


# --------------------------------------------------
# OUSMLoss
# --------------------------------------------------
class OUSMLoss(nn.Module):
    def __init__(self, base_loss_name, base_reduction, k=1, trigger=5):
        super(OUSMLoss, self).__init__()
        self.base_loss_name = base_loss_name
        self.k = k
        self.loss1 = get_base_loss(
            base_loss_name=base_loss_name, reduction=base_reduction
        )
        self.loss2 = get_base_loss(base_loss_name=base_loss_name, reduction="none")
        self.trigger = trigger
        self.ousm = False
        self.current_epoch = 0

    def forward(self, input, target):
        if self.ousm:
            losses = self.loss2(input, target)
            if len(losses.shape) == 2:
                losses = losses.mean(1)
            _, idxs = losses.topk(input.shape[0] - self.k, largest=False)
            losses = losses.index_select(0, idxs)
            return losses.mean()
        else:
            return self.loss1(input, target)

    def update(self):
        self.current_epoch += 1
        if self.current_epoch == self.trigger:
            self.ousm = True
            print("loss: OUSM is True.")

    def __repr__(self):
        return f"OUSM(loss={self.base_loss_name}, k={self.k}, trigger={self.trigger})"


def get_base_loss(base_loss_name, reduction):
    if base_loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(reduction=reduction)
    elif base_loss_name == "MSELoss":
        return nn.MSELoss(reduction=reduction)
    elif base_loss_name == "L1Loss":
        return nn.L1Loss(reduction=reduction)
    elif base_loss_name == "SmoothL1Loss":
        return nn.SmoothL1Loss(reduction=reduction)
    elif base_loss_name == "RMSELoss":
        return RMSELoss(reduction=reduction)
    else:
        raise ValueError(f"Not supported base loss: {base_loss_name}.")


# --------------------------------------------------
# RMSELoss
# --------------------------------------------------
class RMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, inputs, targets):
        return torch.sqrt(self.mse_loss(inputs, targets))


# --------------------------------------------------
# getter
# --------------------------------------------------
def get_losses(config):
    losses = []
    loss_names = config.names
    loss_weights = config.weights
    for loss_name, loss_weight in zip(loss_names, loss_weights):
        if loss_name == "BCEWithLogitsLoss":
            losses.append(
                (loss_weight, nn.BCEWithLogitsLoss(**config.BCEWithLogitsLoss.params))
            )
        elif loss_name == "CrossEntropyLoss":
            losses.append(
                (loss_weight, nn.CrossEntropyLoss(**config.CrossEntropyLoss.params))
            )
        elif loss_name == "FocalLoss":
            losses.append((loss_weight, FocalLoss(**config.FocalLoss.params)))
        elif loss_name == "L1Loss":
            losses.append((loss_weight, nn.L1Loss(**config.L1Loss.params)))
        elif loss_name == "MSELoss":
            losses.append((loss_weight, nn.MSELoss(**config.MSELoss.params)))
        elif loss_name == "NLLLoss":
            losses.append((loss_weight, NLLLoss(**config.NLLLoss.params)))
        elif loss_name == "OUSMLoss":
            losses.append((loss_weight, OUSMLoss(**config.OUSMLoss.params)))
        elif loss_name == "RMSELoss":
            losses.append((loss_weight, RMSELoss(**config.RMSELoss.params)))
        elif loss_name == "SmoothL1Loss":
            losses.append((loss_weight, nn.SmoothL1Loss(**config.SmoothL1Loss.params)))
        else:
            raise ValueError(f"Not supported loss: {loss_name}.")
    return losses
