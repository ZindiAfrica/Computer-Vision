import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, CosineAnnealingLR
from collections import defaultdict
from sklearn.metrics import log_loss
import pandas as pd
from multiprocessing import Pool, cpu_count


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def return_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def accuracy_score(output, labels):
    output = output.detach().cpu()
    labels = labels.detach().cpu()
    output = torch.softmax(output, 1)
    accuracy = (output.argmax(dim=1) == labels).float().mean()
    accuracy = accuracy.detach().cpu().numpy()
    return accuracy


def metric_log_loss(output, labels):
    output = output.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    accuracy = log_loss(labels, output)
    return accuracy


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def get_scheduler(optimizer, scheduler_params, train_loader=None):
    if scheduler_params['scheduler_name'] == 'CosineAnnealingWarmRestarts':
        schedulers = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params['T_0'],
            eta_min=scheduler_params['min_lr'],
            last_epoch=-1
        )

    elif scheduler_params['scheduler_name'] == 'OneCycleLR':
        schedulers = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                         epochs=scheduler_params['epochs'],
                                                         steps_per_epoch=len(train_loader),
                                                         max_lr=scheduler_params['max_lr'],
                                                         pct_start=scheduler_params['pct_start'],

                                                         )
    elif scheduler_params['scheduler_name'] == 'CosineAnnealingLR':
        schedulers = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params['T_max'],
            eta_min=scheduler_params['min_lr'],
            last_epoch=-1
        )
    return schedulers


def return_filpath(name, folder):
    path = os.path.join(folder, f'{name}')
    return path


def return_label(blast, brown, healthy):
    probablity = np.array([blast, brown, healthy])
    return np.argmax(probablity)


class AWP(object):
    """ [Adversarial weight perturbation helps robust generalization](https://arxiv.org/abs/2004.05884)
    """

    def __init__(
            self,
            model,
            emb_name="weight",
            epsilon=0.001,
            alpha=1.0,
    ):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.param_backup = {}
        self.param_backup_eps = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        if self.alpha == 0: return
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                # save
                if is_first_attack:
                    self.param_backup[name] = param.data.clone()
                    grad_eps = self.epsilon * param.abs().detach()
                    self.param_backup_eps[name] = (
                        self.param_backup[name] - grad_eps,
                        self.param_backup[name] + grad_eps,
                    )
                # attack
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.alpha * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data,
                            self.param_backup_eps[name][0]
                        ),
                        self.param_backup_eps[name][1]
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.param_backup:
                param.data = self.param_backup[name]
        self.param_backup = {}
        self.param_backup_eps = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if name in self.grad_backup:
                param.grad = self.grad_backup[name]
        self.grad_backup = {}


class AWP_fast(object):
    def __init__(self, model, optimizer, *, adv_param='weight',
                 adv_lr=0.001, adv_eps=0.001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


def oversample(df):
    classes = df.Label.value_counts().to_dict()
    most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df['Label'] == key])
    classes_sample = []
    for i in range(1, len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe, classes_list[0]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df


# Credit https://gist.github.com/rrherr/ae9cb03bcdb0d322eeac156db9494fcd
@pd.api.extensions.register_dataframe_accessor('apply_parallel')
@pd.api.extensions.register_series_accessor('apply_parallel')
class ApplyParallel:
    """Registers a custom method `apply_parallel` for dataframes & series."""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self, func, axis=0):
        """Applies func to a pandas object along the axis.
        Useful with big pandas objects when func can't be easily vectorized.
        func can't be a lambda function because multiprocessing can't pickle it.
        """
        # df.apply(func, axis=1) applies func to each row,
        # but np.array_split(df, n, axis=0) splits df into n chunks of rows,
        # so swap the axis number when working with dataframes.
        if isinstance(self._obj, pd.DataFrame):
            axis = 1 - axis
        # But if the pandas object is a series, the axis is always 0.
        else:
            assert axis == 0
        # Use all the CPUs
        num_jobs = cpu_count()
        with Pool(num_jobs) as p:
            # Split your pandas object into chunks
            split_objects = np.array_split(self._obj, num_jobs, axis=axis)
            # Map your function to each split in parallel
            results = p.map(func, split_objects)
        # Concat them back together
        return pd.concat(results)
