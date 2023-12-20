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

np.random.seed(2023)
torch.manual_seed(2023)

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"

print("using device: ", DEVICE)

#####################################################
##################### Training #########################
#####################################################

def train(model, src, src_test, optimizer, criterion=nn.CrossEntropyLoss(), 
          setting_params=None, other_params=None, print_output=False, planted_model=False, scheduler=None,
          anneal=False, plot_attn=False, plot_attn_name=None, ckpt_path = None, 
          log_param = False, writer = None):
    """
    Train a simple language model and evaluate train/test loss/error over epochs.
    Args:
        model: the language model (e.g., simplified transformer), torch.nn 
        src: the training sequences, each row is an input sequence
        src_test: the testing sequences
        optimizer: using torch.nn.optim, hyperparameters are standard, otherwise specified in "other_params"
        criterion: training loss function, default is CrossEntropyLoss
        setting_params: a dictionary containing setting parameters such as vocab_size, max_seq_len
        other_params: a dictionary containing other relevant parameters
        print_outout: a boolean indicating whether produces intermediate output, default is off
        planted_model: a boolean indicating whether the model is a planted signal model
        scheduler: optim.scheduler
        anneal: if True, use cosine annealing (restarts every 1/4 max epochs)
        plot_attn: a boolean indicated whether to make attention plots during training
        plot_attn_name: only used if plot_attn is TRUE, a string for figure names
    Returns:
        model: the trained model
        err_arr: a numpy array of size num_epoch-by-4, containing train/test loss/errors
        
    """

    
    

    # Move the model to the specified device
    model = model.to(DEVICE)
    src = src.to(DEVICE)
    src_test = src_test.to(DEVICE)


    num_epoch = setting_params['num_epoch'] if setting_params is not None else 500
    vocab_size = setting_params['vocab_size'] if setting_params is not None else 4
    max_seq_len = setting_params['max_seq_len'] if setting_params is not None else 60
    sample_size, sample_size_test = src.size(0), src_test.size(0)
    batch_size = setting_params['batch_size']
    epoch_change = setting_params['num_epoch']//4
    if other_params is not None:
        pos_arr = other_params['pos_arr'] if 'pos_arr' in other_params.keys() else -1
        pos_arr_test = other_params['pos_arr_test'] if 'pos_arr_test' in other_params.keys() else -1
        k = other_params['num_insrt_tokens'] if 'num_insrt_tokens' in other_params.keys() else -1
    
    err_arr = np.zeros((num_epoch, 4))

    timer = Timer()
    best_err = 1e4
    for epoch in range(num_epoch):
        model.train() # useful if dropout or batchnorm etc is turned on
        perm = np.arange(sample_size, dtype = int)
        np.random.shuffle(perm)
        for batch_idx in range(sample_size // batch_size):
            optimizer.zero_grad()
            indices = perm[range((batch_size*batch_idx),(batch_size*batch_idx+batch_size))]
            # print("==", src.device)
            # print("==", src[indices,:].device)
            output = model(src[indices,:])
            loss = criterion(output[:, :-1].contiguous().view(-1, vocab_size), src[indices, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval() # useful if dropout or batchnorm etc is turned on
            output_train = model(src)
            train_err = 1 - torch.mean(output_train.argmax(dim=2)[:,:-1]  == src[:,1:], dtype=torch.float)
            output_test = model(src_test)
            loss_test = criterion(output_test[:, :-1].contiguous().view(-1, vocab_size), src_test[:, 1:].contiguous().view(-1))
            test_err = 1 - torch.mean(output_test.argmax(dim=2)[:,:-1]  == src_test[:,1:], dtype=torch.float)
            err_arr[epoch,:] = [loss.item(), loss_test.item(), train_err.item(), test_err.item()]
            if planted_model:
                indices = torch.arange(sample_size) * max_seq_len + pos_arr[:,k-2] # only check if the second last token matches
                indices = indices.type(torch.long)
                train_err = 1 - torch.mean(output_train.argmax(dim=2).ravel()[indices] == src.ravel()[indices], dtype=torch.float)
                indices_test = torch.arange(sample_size_test) * max_seq_len + pos_arr_test[:,k-2] # only check if the second last token matches
                indices_test = indices_test.type(torch.long)
                test_err = 1 - torch.mean(output_test.argmax(dim=2).ravel()[indices_test] == src_test.ravel()[indices_test], dtype=torch.float)
                err_arr[epoch,:] = [loss.item(), loss_test.item(), train_err.item(), test_err.item()]
        if scheduler is not None:
            scheduler.step()
        if anneal and (epoch+1) % epoch_change == 0: # restart
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_change)
        if print_output:
            print(f"----> Epoch: {epoch+1}, Train Loss: {loss.item()}, Train Error: {err_arr[epoch,2]}, Test Error: {err_arr[epoch,3]}")
        if plot_attn and epoch % 10 == 0:
            _ , _ = plot_attention(model, src[0,:], fig_name=plot_attn_name+f"_train_epoch_{epoch}", savefig_dir='Figs/attn')
            _ , _ = plot_attention(model, src_test[0,:], fig_name=plot_attn_name+f"_test_epoch_{epoch}", savefig_dir='Figs/attn')
        
        
        if ckpt_path:
            model_dict = model.state_dict()
            ckpt = {
                    'model_dict': model_dict, 
                    'err_arr': err_arr,            
                }
            if setting_params.get('save_epoch') and (epoch+1) % setting_params['save_epoch'] == 0:
                torch.save(ckpt, os.path.join(ckpt_path, 'epoch-{}.pth'.format(epoch + 1)))
            
            if test_err.item() < best_err:
                best_err = test_err.item()
                torch.save(ckpt, os.path.join(ckpt_path, 'best_err.pth'.format(epoch + 1)))
        
        if (epoch+1) % 50 == 0:
            print(f"----> Epoch: {epoch+1}, Train Loss: {loss.item():.2f}, Train Error: {err_arr[epoch,2]:.2f}, Test Error: {err_arr[epoch,3]:.2f},  \
                  time elapsed: {time_str(timer.end())} | {time_str(timer.end()/(epoch+1)*num_epoch)}")
        
        # Log metrics to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/Train', loss.item(), epoch)
            writer.add_scalar('Loss/Test', loss_test.item(), epoch)
            writer.add_scalar('Error/Train', train_err.item(), epoch)
            writer.add_scalar('Error/Test', test_err.item(), epoch)


            # Optional: Log model parameters and gradients
            if log_param:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'{name}.grad', param.grad, epoch)

            


    return model, err_arr

