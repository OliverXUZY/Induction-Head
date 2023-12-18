
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################################
################# Simple utitlity function  ###################
def create_folder(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
#####################################################

def gen_tran_mat(vocab_size, order, sig=1, sparsity=None):
    mat = torch.exp(sig * torch.randn(tuple([vocab_size]* (order+1))))
    mat = mat / mat.sum(dim=-1, keepdim=True)
    if sparsity is not None:
        cutoff = torch.quantile(mat.flatten(), 1-sparsity)
        mat[mat < cutoff] = 0
        mat = mat / mat.sum(dim=-1, keepdim=True)
    return mat

def calc_opt_err(mat):
    """
    Given a transition probability matrix in a Markov chain, calculate the optimal achievable error
    Args: 
        mat: the transition probability matrix, will check its validity
    Returns:
        err_opt: scalar, the optimal error under equilibrium distribution
        pi: 1d array, equilibrium distribution
    """
    
    m = mat.size(0)
    m1,m2 = mat.shape
    assert m1 == m2, 'Incorrect input dimension of transition matrix'
    assert torch.all(mat >= 0) and torch.all(torch.abs(mat.sum(dim=1) - 1) < 1e-6), 'Incorrect input of transition matrix'

    vals, vecs = np.linalg.eig(mat.numpy().T)
    idx = np.argsort(vals)
    pi = np.real(vecs[:, idx[-1]]) # equilibrium distribution
    pi = pi / np.sum(pi) # don't forget to normalize so that it sums to one
    err_opt = np.dot(pi, 1 - mat.max(dim=1)[0].numpy())
    
    return err_opt, pi


def get_mat_full(mat, order=2): 
    """
    For second-order or third-order markov chains, get_mat_full will
    1) if order=2, convert the transition matrix from the tensor form K*K*K to the matrix form (K^2) * (K^2)
    2) if order=3, convert the transition matrix from the tensor form K*K*K*K to the matrix form (K^3) * (K^3)
    Args: 
        mat: the transition probability tensor, will check its validity
    Returns:
        mat_full:  transition probability matrix
    """
    if order == 2:
        m1,m2,m3 = mat.shape
        vocab_size = m1
        assert (m1 == m2) and (m1 == m3), 'Incorrect input dimension of transition matrix'
        assert torch.all(mat >= 0) and torch.all(torch.abs(mat.sum(dim=2) - 1) < 1e-6), 'Incorrect input of transition matrix'
        mat_full = torch.zeros(vocab_size**2, vocab_size**2).float()
        for k1 in range(vocab_size):
            for k2 in range(vocab_size):
                k = k1*vocab_size + k2
                k_out = k2*vocab_size + torch.arange(vocab_size).long()
                mat_full[k, k_out] = mat[k1,k2,:]
    elif order == 3:
        m1,m2,m3,m4 = mat.shape
        vocab_size = m1
        assert (m1 == m2) and (m1 == m3) and (m1 == m4), 'Incorrect input dimension of transition matrix'
        assert torch.all(mat >= 0) and torch.all(torch.abs(mat.sum(dim=3) - 1) < 1e-6), 'Incorrect input of transition matrix'
        mat_full = torch.zeros(vocab_size**3, vocab_size**3).float()
        for k1 in range(vocab_size):
            for k2 in range(vocab_size):
                for k3 in range(vocab_size):
                    k = k1*(vocab_size**2) + k2*vocab_size + k3
                    k_out = k2*(vocab_size**2) + k3*vocab_size + torch.arange(vocab_size).long()
                    mat_full[k, k_out] = mat[k1,k2,k3,:]
    else:
        warnings.warn('The order argument receives an incorrect input.')

    return mat_full
 

#####################################################
##################### Making plots ######################
#####################################################

def plot_err_curve(err_arr, setting_params=None, fig_name=None, save_dir=None,  opt_err=None):
    """
    A simple function to make plots based on err_arr, optionally saving plots in the specified folder
    Args:
        err_arr: a numpy array of size num_epoch-by-4, containing train/test loss/errors
        setting_params: a dictionary containing setting parameters such as vocab_size, max_seq_len
        opt_err: a optional 1d array showing the optimal achievable error 
    """
    num_epoch = setting_params['num_epoch'] if setting_params is not None else err_arr.shape[0]
    if fig_name is not None:
        if save_dir is None:
            if not os.path.isdir('Figs'):
                os.mkdir('Figs')
            save_path = os.path.join('Figs', fig_name)
        else:
            save_path = os.path.join(save_dir, fig_name)
        
    fig, axs = plt.subplots(1, 2, figsize=(13,6))
    axs[0].plot(np.arange(num_epoch, dtype=int), err_arr[:,0], linewidth=2, label='train loss')
    axs[0].plot(np.arange(num_epoch, dtype=int), err_arr[:,1], linewidth=2, label='test loss')
    axs[0].set_yscale('log')
    axs[0].set_title('Train/test loss over epochs')
    axs[1].plot(np.arange(num_epoch, dtype=int), err_arr[:,2], linewidth=2, label='train err')
    axs[1].plot(np.arange(num_epoch, dtype=int), err_arr[:,3], linewidth=2, label='test err')
    if opt_err is not None:
        axs[1].plot(np.arange(num_epoch, dtype=int), np.repeat(opt_err,num_epoch), linestyle='dashed', label='optimal err')
    axs[1].legend()
    axs[1].set_title('Train/test error over epochs')
    
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')

            

def plot_attention(model, tokens, fig_name, savefig_dir='Figs', is_mask=False, num_heads=1, layer=0):
    """
    This function makes two plots, namely attention plot and QK value heatmap
    Args:
        model: the simpleT model we use for the simulations
        tokens: a sequence of tokens, where each token is any element of type torch.long in the vocabulary
        fig_name: name of figure when saving plots
        is_mask: if True, use a mask when calculating QK and attention for next-token prediction
        num_heads: number of attention heads in the model
    Returns:
        QK_vals: the pre-softmax QK values, torch.Tensor 2-d array
        attn: attentions, numpy 2-d array, normalized to sum 1
        
    """
    model.eval()
    seq = model.pos_embed(model.embed(tokens.unsqueeze(0)))
    _, seq_len, d_model = seq.size()
    d_k = d_model // num_heads
    if d_model % num_heads !=0:
        warnings.warn('d_model is not divisible by num_heads!')
    mask = torch.tril(torch.ones(seq_len,seq_len)).unsqueeze(0).to(DEVICE)
    if layer == 0:
        queries = model.h[0].mha.W_q(seq)
        keys = model.h[0].mha.W_k(seq)        
    elif layer ==1:
        attn_output = model.h[0].mha(seq, seq, seq, mask)
        out = model.h[0].dropout(attn_output) if model.h[0].drop is not None else attn_output
        out = seq + out if model.h[0].residual else out
        seq =  model.h[0].layer_norm(out) if model.h[0].norm else out
        queries = model.h[1].mha.W_q(seq)
        keys = model.h[1].mha.W_k(seq)
    else:
        warnings.warn('Layer more than 2 not currently supported!')
    queries = queries.view(1, seq_len, num_heads, d_k).transpose(1, 2)
    keys = keys.view(1, seq_len, num_heads, d_k).transpose(1, 2)
    
    QK_vals = torch.matmul(queries, keys.transpose(-2, -1)).squeeze(dim=0) / np.sqrt(d_k) # num_heads * seq_len * seq_len
    if is_mask:
        QK_masked = QK_vals.masked_fill(mask == 0, -1e9)
    else:
        QK_masked = QK_vals
    attn = F.softmax(QK_masked, dim=2)
    attn = attn.data.cpu().numpy()
    attn /= attn.sum(axis=-1, keepdims=True)

## making plots now
    fig, axs = plt.subplots(num_heads,3,figsize=(45,num_heads*14))

    width = 1
    example_sep = 2
    word_height = 1
    pad = 0.1
    yoffset = 1
    xoffset = 0

    for head in range(num_heads):
        plot_idx = (head,0) if num_heads > 1 else 0
        for position, token in enumerate(tokens.cpu().numpy()):
            axs[plot_idx].text(xoffset + 0,
                     yoffset - position * word_height,
                     token,
                     ha="right",
                     va="center")
            axs[plot_idx].text(xoffset + width,
                     yoffset - position * word_height,
                     token,
                     ha="left",
                     va="center")
        axs[plot_idx].text(xoffset + 0.5 * width,
                 3,
                 "",
                 ha="center",
                 va="top",
                 weight="bold")
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                axs[plot_idx].plot(
                    [xoffset + pad, xoffset + width - pad],
                    [yoffset - word_height * i, yoffset - word_height * j],
                    color="blue",
                    linewidth=1,
                    alpha=attn[head, i, j])
                axs[plot_idx].set_title(f'Post-softmax attentions: head {head}',weight="bold",fontsize=25)

        plot_idx = (head,1) if num_heads > 1 else 1
        pcm = axs[plot_idx].imshow(QK_vals.detach().cpu().numpy()[head,:,:])
        axs[plot_idx].set_title(f'Pre-softmax QK values: head {head}',weight="bold",fontsize=25)
        fig.colorbar(pcm, ax=axs[plot_idx], shrink=0.8)

        plot_idx = (head,2) if num_heads > 1 else 2
        pcm = axs[plot_idx].imshow(attn[head,:,:])
        axs[plot_idx].set_title(f'Attention values: head {head}',weight="bold",fontsize=25)
        fig.colorbar(pcm, ax=axs[plot_idx], shrink=0.8)
    plt.savefig(os.path.join(savefig_dir, fig_name), bbox_inches='tight')
    return (QK_vals, attn)



def plot_incoh_heatmat(save_dir, model, setting_params, remove_firstlast=True):
    train_embed_str = 'trainEmbed_' if setting_params['train_embed'] else ''
    add_embed_str = 'addEmbed_' if setting_params['add_embed'] else ''
    MLP_str = 'MLP_' if setting_params['use_MLP'] else ''
    sig = setting_params['sig']
    plt_save_name = MLP_str + train_embed_str + add_embed_str + f'sig_{sig}_incoh'

    fig, ax = plt.subplots(2,2,figsize=(16,16))

    W_e = model.embed.embed.weight.detach().numpy()
    Gram_e = W_e @ W_e.T
    pcm = ax[0,0].imshow(Gram_e)
    fig.colorbar(pcm, ax=ax[0,0], shrink=0.8)
    ax[0,0].set_title('Gram matrix of static embed matrix', weight='bold')

    W_p = model.pos_embed.pe.weight.detach().numpy()
    W_p = W_p[1:-1] if remove_firstlast else W_p
    Gram_p = W_p @ W_p.T
    pcm = ax[0,1].imshow(Gram_p)
    fig.colorbar(pcm, ax=ax[0,1], shrink=0.8)
    ax[0,1].set_title('Gram matrix of positional embed matrix', weight='bold')

    u, s, vt = np.linalg.svd(W_e)
    ax[1,0].plot(s)
    ax[1,0].set_xlabel('index')
    ax[1,0].set_yscale('log')
    ax[1,0].set_title('Spectrum of the static embed matrix', weight='bold')

    u, s, vt = np.linalg.svd(W_p)
    ax[1,1].plot(s)
    ax[1,1].set_xlabel('index')
    ax[1,1].set_yscale('log')
    ax[1,1].set_title('Spectrum of the positional embed matrix', weight='bold')
    
    plt.savefig(os.path.join(save_dir, plt_save_name), bbox_inches='tight')
    


 

#####################################################
##################### Measurement #####################
#####################################################





def attn_measure(attn, vocab_size, tokens, pattern=None, is_mask=True):
    """
    att_measure provides measurements based on an attention matrix
    Args:
        attn is a matrix of of size len_seq * len_seq, where len_seq is the sequence length
        vocab_size is the size of vocabulary
        tokens is a sequence of tokens of length len_seq
        pattern is a list, which is a subset of vocab
    Return:
        scores: a dictionary containing various measurement scores
    
    """
    if torch.is_tensor(attn):
        Attn = attn.squeeze().detach().numpy()
        Tokens = tokens.squeeze().detach().numpy()
    else:
        Attn, Tokens = attn, tokens
    seq_len, seq_len2 = Attn.shape
    seq_len3 = tokens.shape[0]
    assert seq_len == seq_len2 and seq_len == seq_len3, 'attn two dimensions do not match!'
    if is_mask:
        Attn = Attn * np.tril(np.ones((seq_len,seq_len))) 
    if not np.all(np.abs(np.sum(Attn, axis=1) - 1) < 1e-6): # 'attentions do not sum to one for all tokens, normalize now'
        Attn /= Attn.sum(-1, keepdims=True)
    
    tokens_group = np.zeros((seq_len, vocab_size), dtype=bool)
    for k in range(vocab_size):
        tokens_group[:,k] = (Tokens == k)
    scores = {}
    
    vals = np.diag(Attn)[1:]
    sc = np.zeros(vocab_size+1)
    sc[0] = np.mean(vals)
    sc[1:] = np.array([np.mean(vals[tokens_group[:,k][1:]]) for k in range(vocab_size)])
    scores['self'] = sc
    
    vals = np.diagonal(Attn, offset=-1)
    sc2 = np.zeros(vocab_size+1)
    sc2[0] = np.mean(vals)
    sc2[1:] = np.array([np.mean(vals[tokens_group[:,k][1:]]) for k in range(vocab_size)])
    scores['previous'] = sc2

    vals = np.diagonal(Attn, offset=-2)
    sc3 = np.zeros(vocab_size+1)
    sc3[0] = np.mean(vals)
    sc3[1:] = np.array([np.mean(vals[tokens_group[:,k][2:]]) for k in range(vocab_size)])
    scores['previous2'] = sc3

    vals = np.zeros(seq_len-1)
    for i in range(1,seq_len):
        k = Tokens[i]
        vals[i-1] = np.sum(Attn[i,tokens_group[:,k]])
    sc4 = np.zeros(vocab_size+1)
    sc4[0] = np.mean(vals)
    sc4[1:] = np.array([np.mean(vals[tokens_group[:,k][1:]]) for k in range(vocab_size)])
    scores['self-broad'] = sc4
    
    if pattern is not None:
        sc5 = np.zeros(vocab_size+1)
        vals = np.zeros(seq_len-1)
        for j in pattern:
            vals += np.concatenate((np.zeros(seq_len), np.diagonal(Attn, offset=-j)))[-(seq_len-1):]
        
        #tmp = [np.diagonal(Attn, offset=-j) for j in pattern]
        #vals = np.array([item for sublist in tmp for item in sublist])
        sc5[0] = np.sum(vals) / (seq_len-1)
        for k in range(vocab_size):
            vals = np.zeros(seq_len-1)
            for j in pattern:
                vals += np.concatenate((np.zeros(seq_len), np.diagonal(Attn, offset=-j)[tokens_group[:,k][j:]]))[-(seq_len-1):]
            #tmp = [np.diagonal(Attn, offset=-j)[tokens_group[:,k][j:]] for j in pattern]
            #vals = np.array([item for sublist in tmp for item in sublist])
            sc5[k+1] = np.sum(vals) / np.sum(tokens_group[:,k])

        scores['pattern'] = sc5
    return scores



import os
import time
import shutil
import random
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import numpy as np

_log_path = None


def set_gpu(gpu):
    print('set gpu: {:s}'.format(gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def check_file(path):
    if not os.path.isfile(path):
        raise ValueError('file does not exist: {:s}'.format(path))


def check_path(path):
    if not os.path.exists(path):
        raise ValueError('path does not exist: {:s}'.format(path))


def ensure_path(path, remove=False):
    if os.path.exists(path):
        print("path exists! path: ",path)
        if remove:
            if input('{:s} exists, remove? ([y]/n): '.format(path)) != 'n':
                shutil.rmtree(path)
                os.makedirs(path)
    else:
        os.makedirs(path)


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def count_params(model, return_str=True):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    if return_str:
        if n_params >= 1e6:
            return '{:.1f}M'.format(n_params / 1e6)
        else:
            return '{:.1f}K'.format(n_params / 1e3)
    else:
        return n_params


class AverageMeter(object):

    def __init__(self):

        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.mean = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean = self.sum / self.count

    def item(self):
        return self.mean


class Timer(object):

    def __init__(self):

        self.start()

    def start(self):
        self.v = time.time()

    def end(self):
        return time.time() - self.v


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t > 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def fix_random_seed(seed, reproduce=False):
    # cudnn.enabled = True
    # cudnn.benchmark = True
    
    if reproduce:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ## NOTE: uncomment for CUDA >= 10.2
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        ## NOTE: uncomment for pytorch >= 1.8
        # torch.use_deterministic_algorithms(True)

    # os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)

    return rng

