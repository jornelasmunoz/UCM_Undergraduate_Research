import os
from datetime import datetime, timedelta
import torch
import numpy as np


def wandb_config(params, hyperparams={}):
    '''
    Returns config dictionary for Weights and Biases runs.
    Inputs: parameter dictionary and hyperparameter dictionary
    Output: one configuration dictionary (includes both parameters and hyperparameters)
    '''
    synonyms = {
        'learning_rate': ['learning_rate', 'lr'],
        'epochs': ['epochs'],
        'batch_size': ['batch_size'],
        'loss_function': ['loss_finction', 'loss'],
        'model': ['model', 'architecture'],
        'dataset': ['dataset', 'data'],
        'optimizer': ['optimizer', 'optim'],
        'kernel_size': ['kernel_size', 'kernel'],
    }
    
    defaults = {
        "dataset": "encoded_MNIST",
        "model": params['kind'] + '_' + params['suffix'],
        "learning_rate": 1e-3,
        "batch_size": 100,
        "epochs": 50,
        "kernel_size":23,
        "p": 23,
        "image_size":23,
        "suffix": '',
    }
    
    # The config will be set by looking through synonyms and given params
    # If "None", provided reverts to defaults. Also if not provided
    config = {}
    for key, syn in synonyms.items():
        for s in syn:
            if s in hyperparams and hyperparams[s] is not None: config[key] = hyperparams[s]; continue
            if s in      params and      params[s] is not None: config[key] =      params[s]; continue
            
    # Second dict replaces any common elements with first dict
    config = {**defaults, **config}
    
    return config