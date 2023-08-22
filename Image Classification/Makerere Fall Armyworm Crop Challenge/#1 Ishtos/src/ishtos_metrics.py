import torchmetrics


# --------------------------------------------------
# getter
# --------------------------------------------------
def get_metrics(config):
    metrics = []
    metric_names = config.names
    for metric_name in metric_names:
        if metric_name == "Accuracy":
            metrics.append(
                (metric_name, torchmetrics.Accuracy(**config.Accuracy.params))
            )
        elif metric_name == "AUROC":
            metrics.append((metric_name, torchmetrics.AUROC(**config.AUROC.params)))
        elif metric_name == "MeanAbsoluteError":
            metrics.append((metric_name, torchmetrics.MeanAbsoluteError()))
        elif metric_name == "MeanAbsolutePercentageError":
            metrics.append((metric_name, torchmetrics.MeanAbsolutePercentageError()))
        elif metric_name == "MeanSquaredError":
            metrics.append(
                (
                    metric_name,
                    torchmetrics.MeanSquaredError(**config.MeanSquaredError.params),
                )
            )
        elif metric_name == "MeanSquaredLogError":
            metrics.append((metric_name, torchmetrics.MeanSquaredLogError()))
        else:
            raise ValueError(f"Not supported metric: {metric_name}.")
    return metrics
