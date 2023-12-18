# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import os
import json 
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
# warnings.simplefilter("ignore")
print(torch.__version__)
import argparse
import yaml

from src.config import Config 
from src.utils import fix_random_seed, create_folder, ensure_path
from src.data import gen_simple_data
from src.model import TFModel, simpleT
from src.train import train
from src.utils import plot_err_curve, Timer, time_str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./save",
                        help="save path for model ckpt")
    parser.add_argument("--yaml", type=str, help="yaml file to load from")

    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")

def main():
    
    seed = 2023
    fix_random_seed(seed)

            
    save_plot = True
    save_plot_dir = f'Figs/seed_{seed}'

    create_folder('Figs')
    create_folder(f'Figs/seed_{seed}')

    
    ###### set up config
    # Load YAML file
    with open('./config/config.yaml', 'r') as file:
        config_args = yaml.safe_load(file)
    # Create config object
    config = Config(**config_args)
    
    
    
    

    vocab = torch.arange(config.vocab_size).type(torch.LongTensor)

    src, src_test = torch.zeros(config.sample_size,config.max_seq_len).long(), torch.zeros(config.sample_size_test,config.max_seq_len).long()
    src[range(0,config.sample_size),:] = gen_simple_data(vocab, config.max_seq_len, config.sample_size, pattern='random')
    src_test[range(0,config.sample_size_test),:] = gen_simple_data(vocab, config.max_seq_len, config.sample_size_test, pattern='random')
    
    print("src_test.shape", src_test.shape)


    ###### set up model
    model = TFModel(config, num_hidden_layers=config.num_layers)

    # print(model)

    # output_test = model(src_test)
    # print(output_test)

    # print("====")

    # model = simpleT(config.vocab_size,config.d_model, config.num_heads, config.max_seq_len, add_embed=True, train_from_scratch=False,
    #              residual=True, dropout=0.1, norm=True, outdim_truncate=False, trainable=[True, True])
    
    # print(model)

    # output_test = model(src_test)
    # print(output_test)
    # print(output_test.shape)
    # return 



    ### set up optimizer
    wd = 5e-4 if config.use_wd else 0
    wd_2 = 0 if config.use_wd else 0

    parameters = []
    # store params & learning rates
    for idx, (name, param) in enumerate(model.named_parameters()):
        # display info
        print(f'{idx}: layer_name: {name}')
        # append layer parameters
        if name == 'mha.W_q.weight' or name == 'mha.W_k.weight' or name == 'mha2.W_q.weight' or name == 'mha2.W_k.weight':
            parameters += [{'params': [pa for na, pa in model.named_parameters() if na == name and pa.requires_grad],
                            'lr':     config.lr,
                            'weight_decay': wd}]
        else:
            parameters += [{'params': [pa for na, pa in model.named_parameters() if na == name and pa.requires_grad],
                            'lr':     config.lr,
                            'weight_decay': wd_2}]
    
    if config.fancy_opt:
        optimizer = torch.optim.AdamW(parameters)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epoch//4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epoch//4)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   

    ### training
    timer = Timer()
    ckpt_path = config.ckpt_path if config.ckpt_path else "./save"
    ensure_path(ckpt_path)
    model, err_arr = train(model, src, src_test, optimizer, setting_params = config_args, print_output = True, criterion=criterion, scheduler=scheduler, ckpt_path = ckpt_path)

    print(f"time elapsed: {time_str(timer.end())}")
    model_dict = model.state_dict()
    ckpt = {
            'model_dict': model_dict, 
            'err_arr': err_arr,            
        }
    torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))


if __name__ == "__main__":
    main()
    