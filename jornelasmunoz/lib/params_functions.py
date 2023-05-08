import numpy as np
import torch
import MURA as mura

def model_params_defaults(params, kind = ''):
    defaults = {
      "dataset": f"MNIST_mura",#_{SNR}_reconstructed_{method}_method",
      "learning_rate": 1e-3,
      "batch_size": 100,
      "epochs": 50,
      "p": 23, # size of array (has to be prime)
      "kernel_size": 3,
      "SNR": 'noiseless',
      "method": 'direct',
      "kind": kind,
    }
    # the second arg replaces any common elements with the first
    params = {**defaults, **params}
    
    params['model'] = params['kind']+'_' + params['suffix'] 
    params['model_save_path'] = f'../models/{params["kind"]}/{params["model"]}.pth'

    # Compute MURA encoder and decoder
    params['A'] = mura.create_binary_aperture_arr(params['p'])
    params['G'] = mura.create_decoding_arr(params['A'])
    
    return params
    