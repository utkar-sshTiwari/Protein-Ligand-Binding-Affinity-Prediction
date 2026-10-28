import numpy as np


def concordance_index(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num, den = 0.0, 0
    for i in range(len(y_true)):
        mask = y_true[i] > y_true
        n = int(mask.sum())
        if n == 0:
            continue
        diff = y_pred[i] - y_pred[mask]
        num += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        den += n
    return float(num / den) if den else float("nan")


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    if y_true.std() == 0 or y_pred.std() == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"mse": mse, "rmse": mse ** 0.5, "ci": concordance_index(y_true, y_pred), "pearson": pearson}
