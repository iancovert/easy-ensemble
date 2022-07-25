import numpy as np
from sklearn import metrics


EPS = 1e-6


def log_loss(preds, targets):
    return metrics.log_loss(targets, preds)


def log_loss_logits(preds, targets):
    if len(preds.shape) == 2 and preds.shape[1] > 1:
        preds = softmax(preds)
    else:
        preds = sigmoid(preds)
    return metrics.log_loss(targets, preds)


def accuracy(preds, targets):
    if len(preds.shape) == 1:
        argmax = (preds > 0.5).astype(int)
    else:
        argmax = np.argmax(preds, axis=1)
    return np.mean(targets == argmax)


def mean_squared_error(preds, targets):
    return metrics.mean_squared_error(targets, preds)


def mean_absolute_error(preds, targets):
    return metrics.mean_absolute_error(targets, preds)


def softmax(preds):
    m = preds.max(axis=1, keepdims=True)
    return np.exp(preds - m) / np.exp(preds - m).sum(axis=1, keepdims=True)


def sigmoid(preds):
    return 1 / (1 + np.exp(- preds))
    
    
def apply_ensemble(preds_iterable, weights, transform=None):
    preds = np.array([w * pred for w, pred in zip(weights, preds_iterable)]).sum(axis=0)
    if transform:
        preds = transform(preds)
    return preds


def stable_log(preds):
    return np.log(preds + EPS)


def stable_logit(preds):
    return np.log(preds + EPS) - np.log(1 - preds + EPS)
