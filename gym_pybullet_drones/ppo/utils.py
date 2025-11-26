'''Additional PPO utilities.'''

import numpy as np
import torch

def normalize_tensor(tensor, epsilon=1e-8):
    '''Normalize a tensor to zero mean and unit variance.'''
    return (tensor - tensor.mean()) / (tensor.std() + epsilon)

def explained_variance(y_pred, y_true):
    '''Computes fraction of variance that ypred explains about y.'''
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = torch.var(y_true)
    return 1 - torch.var(y_true - y_pred) / var_y