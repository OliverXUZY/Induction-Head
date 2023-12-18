# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import os
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import *
warnings.simplefilter("ignore")
print(torch.__version__)


#####################################################
##################### Setting configuration ################
#####################################################

def get_config(MC_type, params=None, seed=2023):
    ## some default values
    setting_params = {
    'vocab_size' : 4, 
    'order': 1,
    'sig': 1,
    'max_seq_len': 60,
    'sample_size': 5000,
    'sample_size_test': 500,
    'num_epoch': 500,
    'batch_size': 50,
    'num_heads': 1,
    'num_layers': 1,
    'train_embed': True,
    'add_embed': True,
    'dropout': 0.2, 
    'resid': True, # used in 2-layer transformer
    'norm': True,
    'use_MLP': False, 
    'lr': 0.005,
    'wd': 1e-5,
    'LS': 0.1, # label smoothing
    'pattern': None,
    'save_folder': 'Figs/markov3',
    'seed': seed
    }
        
    if MC_type == '1st order':
        setting_params['sig'] = 2
    elif MC_type == '1st order-2':
        setting_params['save_folder'] = 'Figs/markov3-2'
        setting_params['vocab_size'] = 10
        setting_params['sig'] = 2
    elif MC_type == '2nd order':
        setting_params['save_folder'] = 'Figs/markov4'
        setting_params['order'] = 2
    elif MC_type == '2nd order-2':
        setting_params['save_folder'] = 'Figs/markov4-2'
        setting_params['order'] = 2
        setting_params['vocab_size'] = 10
        setting_params['sig'] = 2
    elif MC_type == '3rd order':
        setting_params['save_folder'] = 'Figs/markov5'
        setting_params['order'] = 3
    elif MC_type == '3rd order-2': 
        setting_params['save_folder'] = 'Figs/markov5-2'
        setting_params['order'] = 3
        setting_params['vocab_size'] = 10
        setting_params['sig'] = 2
    elif MC_type == 'mixed1st':
        setting_params['save_folder'] = 'Figs/markov6'
        setting_params['pattern'] = np.array([9]) 
    elif MC_type == 'mixed1st-2':
        setting_params['save_folder'] = 'Figs/markov6-2'
        setting_params['vocab_size'] = 10
        setting_params['sig'] = 2
        setting_params['pattern'] = np.array([9]) 
    elif MC_type == 'linear high order':
        setting_params['save_folder'] = 'Figs/markov7'
        setting_params['pattern'] = np.arange(5, dtype=int)
    elif MC_type == 'linear high order-2':
        setting_params['save_folder'] = 'Figs/markov7-2'
        setting_params['vocab_size'] = 10
        setting_params['sig'] = 2
        setting_params['pattern'] = np.arange(5, dtype=int)
    else:
        warnings.warn('Not valid MC type')
    
    return setting_params

def get_optimizer(model, params):
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], betas=(0.9, 0.98), eps=1e-9, weight_decay=params['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, params['num_epoch']//4)
    criterion = nn.CrossEntropyLoss(label_smoothing=params['LS'])  
    return optimizer, scheduler, criterion


class Config:
    '''
    This is the configuration class to store the configuration of a `simpleTF`. It is used to
    instantiate a model according to the specified arguments, defining the model architecture.
    '''
    def __init__(
            self,
            vocab_size, num_heads, max_seq_len, add_embed=False, init_weight = None, train_from_scratch=False,
                residual=False, dropout=None, norm=False, outdim_truncate=False, trainable=[False, False], ff_dim=None,
                d_model = None,
                 **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = vocab_size + max_seq_len
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.add_embed = add_embed
        self.init_weight = init_weight
        self.train_from_scratch = train_from_scratch
        self.residual = residual
        self.dropout = dropout
        self.norm = norm
        self.outdim_truncate = outdim_truncate
        self.trainable = trainable
        self.ff_dim = ff_dim
        self.__dict__.update(kwargs)

