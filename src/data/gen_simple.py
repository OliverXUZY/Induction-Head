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

from ..utils import *
warnings.simplefilter("ignore")
print(torch.__version__)

np.random.seed(2023)
torch.manual_seed(2023)


#####################################################
##################### Data generation ####################
#####################################################

def gen_simple_data(vocab, max_seq_len, sample_size, pattern='aaa', pattern_sample_len=None):
    """
    Generate input sequences for training/testing based on different patterns.
    Simple repetitions of certain short-ranged patterns.
    Args:
        vocab: 1d torch.Tensor containing entire vocabulary
        max_seq_len: positive integer that specifies the maximum number of tokens in a sequence
        sample_size: the number of input sequences
        pattern: a string that indicated the short pattern used for generating sequence data, or a 1-d Tensor array
        pattern_sample_len:: the length of sampled patterns, only used when pattern ='random'
    Returns:
        data: input sequences, 2d torch.Tensor of type torch.long
    """
    vocab_size = vocab.size(0)
    data = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)
    id0, id1, id2 = 0, 1, 2
    if max_seq_len % 12 != 0:
        warnings.warn('max_seq_len is not divisible by 12, which may cause issues!')

    for i in range(sample_size):
        if pattern == 'aaa':
            data[i, :] = vocab[id0].repeat(max_seq_len)
        elif pattern == 'abab':
            data[i, :] = vocab[[id0,id1]].repeat(max_seq_len // 2)
        elif pattern == 'aab':
            data[i, :] = vocab[[id0,id0,id1]].repeat(max_seq_len // 3)
        elif pattern == 'abb':
            data[i, :] = vocab[[id0,id1,id1]].repeat(max_seq_len // 3)
        elif pattern == 'aabb':
            data[i, :] = vocab[[id0,id0,id1,id1]].repeat(max_seq_len // 4)
        elif pattern == 'aaab':
            data[i, :] = vocab[[id0,id0,id0,id1]].repeat(max_seq_len // 4)
        elif pattern == 'abc':
            data[i, :] = vocab[[id0,id1,id2]].repeat(max_seq_len // 3)
        elif pattern == 'random':
            if pattern_sample_len is None:
                pattern_len = np.random.randint(low=11,high=20)
            pattern_sample = torch.multinomial(torch.ones(vocab_size)/vocab_size,  pattern_len, replacement=True)
            r = max_seq_len % pattern_len
            if r == 0:
                data[i, :] = vocab[pattern_sample].repeat(max_seq_len // pattern_len)
            else: # max_seq_len not divisible by  pattern_sample_len, then fill the remaining tokens by random ones
                data[i, :-r] = vocab[pattern_sample].repeat(max_seq_len // pattern_len)
                data[i, -r:] = torch.multinomial(torch.ones(vocab_size)/vocab_size, r, replacement = True)
        elif pattern == "com_random":
            data[i, :] = torch.multinomial(torch.ones(vocab_size)/vocab_size,  max_seq_len, replacement=True)


        else: # for a given pattern
            assert max_seq_len % len(pattern) == 0, 'length of pattern needs to be divisible by max_seq_len'
            data[i, :] = vocab[pattern].repeat(max_seq_len // len(pattern))
            #warnings.warn('Pattern argument may not receive a correct input!')
            
    return data

def gen_insert_data(vocab, max_seq_len, sample_size, background='random', background_random_weight=None,
                    pattern='aaa', insrt_num_tokens=5, insrt_sep=1):
    """
    Generate input sequences for training/testing based on different patterns.
    Insert a short pattern in a purely random sequence (if background='random') or all-zero sequence
    Args:
        vocab: 1d torch.Tensor containing entire vocabulary
        max_seq_len: positive integer that specifies the maximum number of tokens in a sequence
        sample_size: the number of input sequences
        background: 'random' produces random sequence background, otherwise all-zero sequences
        background_random_weight: probability weight for generating random background, default is uniform random 
        pattern: 'aaa' produces simple repetition pattern, 'random' produces a randomly sampled short pattern
                1d torch.Tensor plants the specified pattern into background, otherwise do nothing
        insrt_num_tokens: the number of tokens being planted 
        insrt_sep: the index difference between consecutive tokens that are planted
    Returns:
        data: input sequences, 2d torch.Tensor of type torch.long
        pattern: the torch.Tensor pattern being sampled if pattern='aaa' or 'random'
        pos_arr: 2d torch.Tensor, the indices of tokens planted in the sequences

    """
    assert type(insrt_num_tokens) == int, 'insrt_num_tokens must be an odd integer'
    assert insrt_num_tokens % 2 == 1, 'insrt_num_tokens must be an odd integer'
    vocab_size = vocab.size(0)
    k = insrt_num_tokens // 2
    if background_random_weight is None: # uniform random noise in background
        background_random_weight = torch.ones(vocab_size) / vocab_size

    if background == 'random':
        data = torch.multinomial(background_random_weight.repeat(sample_size,1), max_seq_len, replacement=True) # random background
    else:
        data = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)
        
    pos_arr = torch.zeros(sample_size, insrt_num_tokens)
    
    for i in range(sample_size):
        insrt_pos_center = torch.randint(k*insrt_sep, max_seq_len-k*insrt_sep, size=(1,))
        insrt_pos = torch.arange(-k,k+1)*insrt_sep + insrt_pos_center
        insrt_pos.type(torch.LongTensor)
        pos_arr[i,:] = insrt_pos
        if pattern == 'aaa':
            pattern = torch.multinomial(torch.ones(vocab_size), 1).repeat(insrt_num_tokens)
            data[i, insrt_pos] = vocab[pattern] # plant a simple repetition patter
        elif pattern == 'random':
            pattern = torch.multinomial(torch.ones(vocab_size), insrt_num_tokens, replacement=True) # a random pattern
            data[i, insrt_pos] = vocab[pattern] # planted the pattern sampled earlier
        elif torch.is_tensor(pattern):
            data[i, insrt_pos] = vocab[pattern] # planted the pattern given by the argument
        else:
            pass # do not plant signal, only has background
            
    return data, pattern, pos_arr

def gen_markov_data(vocab, max_seq_len, sample_size, transition_mat, init_state_dist=None):
    """
    Generate input sequences for training/testing based on a markov chain.
    The markov chain is sampled according to a given transition matrix
    Args:
        vocab: 1d torch.Tensor containing entire vocabulary
        max_seq_len: positive integer that specifies the maximum number of tokens in a sequence
        sample_size: the number of input sequences
        transition_max: transition matrix for the Markov chain, a 2d torch.Tensor that has the same dimension 
                as the vocabulary size, must be a valid transition matrix
        init_state_dist: the initial state distribution, 1d Tensor that sums to one
    Returns:
        data: input sequences, 2d torch.Tensor of type torch.long
    """
    vocab_size = vocab.size(0)
    m1,m2 = transition_mat.shape
    assert (m1 == m2) and (m1 == vocab_size), 'Incorrect input dimension of transition matrix'
    assert torch.all(transition_mat >= 0) and torch.all(torch.abs(transition_mat.sum(dim=1) - 1) < 1e-6), 'Incorrect input of transition matrix'
    if init_state_dist is not None:
        assert torch.abs(init_state_dist.sum() - 1) < 1e-6, 'Incorrect input of initial state distribution: not summing to one'
    data = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)
    if init_state_dist is None:  # random initial states at position 0 and 1 for each sequence
        data[:, 0] = torch.randint(0, vocab_size, size=(sample_size,)) 
    else: # use the initial state distribution if provided
        states_init = torch.multinomial(init_state_dist, sample_size, replacement=True)
        data[:,0] = states_init
    for i in range(sample_size):
        for j in range(max_seq_len-1):
            data[i,j+1] = torch.multinomial(transition_mat[data[i,j],:], 1)
            
    return data


def gen_2nd_markov_data(vocab, max_seq_len, sample_size, transition_mat, init_state_dist=None):
    """
    Generate input sequences for training/testing based on a markov chain.
    The markov chain is sampled according to a given transition matrix/tensor
    Args:
        vocab: 1d torch.Tensor containing entire vocabulary
        max_seq_len: positive integer that specifies the maximum number of tokens in a sequence
        sample_size: the number of input sequences
        transition_max: transition matrix for the Markov chain, a 3d torch.Tensor K * K * K where the last dimension is the
                                    output probability; must be a valid transition matrix
        init_state_dist: the initial state distribution, 1d Tensor that sums to one
    Returns:
        data: input sequences, 2d torch.Tensor of type torch.long
    """
    vocab_size = vocab.size(0)
    m1,m2,m3 = transition_mat.shape
    assert (m1 == m2) and (m1 == m3) and (m1 == vocab_size), 'Incorrect input dimension of transition matrix'
    assert torch.all(transition_mat >= 0) and torch.all(torch.abs(transition_mat.sum(dim=2) - 1) < 1e-6), 'Incorrect input of transition matrix'
    if init_state_dist is not None:
        assert torch.abs(init_state_dist.sum() - 1) < 1e-6, 'Incorrect input of initial state distribution: not summing to one'
    data = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)
    if init_state_dist is None:  # random initial states at position 0 and 1 for each sequence
        data[:, 0] = torch.randint(0, vocab_size, size=(sample_size,)) 
        data[:, 1] = torch.randint(0, vocab_size, size=(sample_size,))
    else: # use the initial state distribution if provided
        states_full = torch.multinomial(init_state_dist, sample_size, replacement=True)
        data[:,0] = states_full // vocab_size
        data[:,1] = states_full % vocab_size
    for i in range(sample_size):
        for j in range(max_seq_len-2):
            data[i,j+2] = torch.multinomial(transition_mat[data[i,j], data[i,j+1], :], 1)
            
    return data


def gen_3rd_markov_data(vocab, max_seq_len, sample_size, transition_mat, init_state_dist=None):
    """
    Generate input sequences for training/testing based on a markov chain.
    The markov chain is sampled according to a given transition matrix/tensor
    Args:
        vocab: 1d torch.Tensor containing entire vocabulary
        max_seq_len: positive integer that specifies the maximum number of tokens in a sequence
        sample_size: the number of input sequences
        transition_max: transition matrix for the Markov chain, a 4d torch.Tensor K * K * K where the last dimension is the
                                    output probability; must be a valid transition matrix
        init_state_dist: the initial state distribution, 1d Tensor that sums to one
    Returns:
        data: input sequences, 3d torch.Tensor of type torch.long
    """
    vocab_size = vocab.size(0)
    m1,m2,m3,m4 = transition_mat.shape
    assert (m1 == m2) and (m1 == m3) and (m1 == m4) and (m1 == vocab_size), 'Incorrect input dimension of transition matrix'
    assert torch.all(transition_mat >= 0) and torch.all(torch.abs(transition_mat.sum(dim=3) - 1) < 1e-6), 'Incorrect input of transition matrix'
    if init_state_dist is not None:
        assert torch.abs(init_state_dist.sum() - 1) < 1e-6, 'Incorrect input of initial state distribution: not summing to one'
    data = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)
    if init_state_dist is None:  # random initial states at position 0 and 1 for each sequence
        data[:, 0] = torch.randint(0, vocab_size, size=(sample_size,)) 
        data[:, 1] = torch.randint(0, vocab_size, size=(sample_size,))
        data[:, 2] = torch.randint(0, vocab_size, size=(sample_size,))
    else: # use the initial state distribution if provided
        states_full = torch.multinomial(init_state_dist, sample_size, replacement=True)
        s0, s12 = states_full // (vocab_size**2), states_full % (vocab_size**2)
        data[:,0] = s0
        data[:,1] = s12 // vocab_size
        data[:,2] = s12 % vocab_size
    for i in range(sample_size):
        for j in range(max_seq_len-3):
            data[i,j+3] = torch.multinomial(transition_mat[data[i,j], data[i,j+1],data[i,j+2], :], 1)
            
    return data


def gen_mixed1st_markov_data(vocab, max_seq_len, sample_size, transition_mat, transition_mat2, init_state_dist=None):
    """
    Generate input sequences for training/testing based on two markov chains.
    Both markov chains are sampled according to the associated given transition matrices
    Args:
        vocab: 1d torch.Tensor containing entire vocabulary
        max_seq_len: positive integer that specifies the maximum number of tokens in a sequence
        sample_size: the number of input sequences
        transition_max: transition matrix for the Markov chain, a 2d torch.Tensor K * K where the second dimension is the
                                    output probability; must be a valid transition matrix
        transition_mat2: similar to transition_max, used to model long-range dependence
        init_state_dist: the initial state distribution, 1d Tensor that sums to one
    Returns:
        data: input sequences, 2d torch.Tensor of type torch.long
    """
    vocab_size = vocab.size(0)
    m1,m2 = transition_mat.shape
    assert (m1 == m2)  and (m1 == vocab_size), 'Incorrect input dimension of transition matrix'
    assert torch.all(transition_mat >= 0) and torch.all(torch.abs(transition_mat.sum(dim=1) - 1) < 1e-6), 'Incorrect input of transition matrix'
    m1,m2 = transition_mat2.shape
    assert (m1 == m2) and (m1 == vocab_size), 'Incorrect input dimension of transition matrix'
    assert torch.all(transition_mat2 >= 0) and torch.all(torch.abs(transition_mat2.sum(dim=1) - 1) < 1e-6), 'Incorrect input of transition matrix'
    if init_state_dist is not None:
        assert torch.abs(init_state_dist.sum() - 1) < 1e-6, 'Incorrect input of initial state distribution: not summing to one'
        
    data = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)
    if init_state_dist is None:  # random initial states at position 0 and 1 for each sequence
        data[:, 0] = torch.randint(0, vocab_size, size=(sample_size,)) 
    else: # use the initial state distribution if provided
        states_init = torch.multinomial(init_state_dist, sample_size, replacement=True)
        data[:,0] = states_init
    for i in range(sample_size):
        for j in range(max_seq_len-1):
            a = torch.bernoulli(torch.Tensor([0.3]))
            token = torch.multinomial(transition_mat[data[i,j], :], 1)
            token2 = torch.multinomial(transition_mat2[data[i,j-9], :], 1) if j > 8 else token
            data[i,j+1] = a * token + (1-a) * token2
            
    return data

def gen_mixed2nd_markov_data(vocab, max_seq_len, sample_size, transition_mat, transition_mat2, init_state_dist=None):
    """
    Generate input sequences for training/testing based on two markov chains.
    Both markov chains are sampled according to the associated given transition matrix/tensor
    Args:
        vocab: 1d torch.Tensor containing entire vocabulary
        max_seq_len: positive integer that specifies the maximum number of tokens in a sequence
        sample_size: the number of input sequences
        transition_max: transition matrix for the Markov chain, a 3d torch.Tensor K * K * K where the last dimension is the
                                    output probability; must be a valid transition matrix
        transition_mat2: similar to transition_max, used to model long-range dependence
        init_state_dist: the initial state distribution, 1d Tensor that sums to one
    Returns:
        data: input sequences, 2d torch.Tensor of type torch.long
    """
    vocab_size = vocab.size(0)
    m1,m2,m3 = transition_mat.shape
    assert (m1 == m2) and (m1 == m3) and (m1 == vocab_size), 'Incorrect input dimension of transition matrix'
    assert torch.all(transition_mat >= 0) and torch.all(torch.abs(transition_mat.sum(dim=2) - 1) < 1e-6), 'Incorrect input of transition matrix'
    m1,m2,m3 = transition_mat2.shape
    assert (m1 == m2) and (m1 == m3) and (m1 == vocab_size), 'Incorrect input dimension of transition matrix'
    assert torch.all(transition_mat2 >= 0) and torch.all(torch.abs(transition_mat2.sum(dim=2) - 1) < 1e-6), 'Incorrect input of transition matrix'
    if init_state_dist is not None:
        assert torch.abs(init_state_dist.sum() - 1) < 1e-6, 'Incorrect input of initial state distribution: not summing to one'
        
    data = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)
    if init_state_dist is None:  # random initial states at position 0 and 1 for each sequence
        data[:, 0] = torch.randint(0, vocab_size, size=(sample_size,)) 
        data[:, 1] = torch.randint(0, vocab_size, size=(sample_size,))
    else: # use the initial state distribution if provided
        states_full = torch.multinomial(init_state_dist, sample_size, replacement=True)
        data[:,0] = states_full // vocab_size
        data[:,1] = states_full % vocab_size
    for i in range(sample_size):
        for j in range(max_seq_len-2):
            a = torch.bernoulli(torch.Tensor([1/2]))
            token = torch.multinomial(transition_mat[data[i,j], data[i,j+1], :], 1)
            token2 = torch.multinomial(transition_mat2[data[i,j-9], data[i,j-8], :], 1) if j > 8 else token
            data[i,j+2] = a * token + (1-a) * token2
            
    return data


def gen_higher_markov_data(vocab, max_seq_len, sample_size, transition_mat, init_state_dist=None):
    """
    Generate input sequences for training/testing based on a higher markov chain,
    which is constructed from a first-order transition probability matrix
    Args:
        vocab: 1d torch.Tensor containing entire vocabulary
        max_seq_len: positive integer that specifies the maximum number of tokens in a sequence
        sample_size: the number of input sequences
        transition_max: transition matrix for the Markov chain, a 2d torch.Tensor K * K where the second dimension is the
                                    output probability; must be a valid transition matrix
        init_state_dist: the initial state distribution, 1d Tensor that sums to one
    Returns:
        data: input sequences, 2d torch.Tensor of type torch.long
    """
    vocab_size = vocab.size(0)
    m1,m2 = transition_mat.shape
    assert (m1 == m2) and (m1 == vocab_size), 'Incorrect input dimension of transition matrix'
    assert torch.all(transition_mat >= 0) and torch.all(torch.abs(transition_mat.sum(dim=1) - 1) < 1e-6), 'Incorrect input of transition matrix'
    data = torch.zeros(sample_size, max_seq_len).type(torch.LongTensor)

    if init_state_dist is None:  # random initial states at position 0 and 1 for each sequence
        data[:, 0] = torch.randint(0, vocab_size, size=(sample_size,))
    else:
        states_init = torch.multinomial(init_state_dist, sample_size, replacement=True)
        data[:,0] = states_init
    for i in range(sample_size):
        for j in range(max_seq_len-1):
            if j < 4:
                  data[i,j+1] = torch.multinomial(transition_mat[data[i,j], :], 1)
            else:        
                vec = torch.Tensor([torch.mean((data[i,range(j-4, j+1)] == k)+0.0) for k in range(vocab_size)]) 
                probs = torch.matmul(transition_mat.T, vec) 
                data[i,j+1] = torch.multinomial(probs, 1)
 
    return data


def gen_binary_random_pattern(vocab, max_seq_len, sample_size, probs=1/2):
    """
    Generate two types of patterns randomly, which are mixed in the inputs
    """
    vocab_size = vocab.size(0)
    assert vocab_size >= 6, 'Need vocabulary size no smaller than 7'
    
    tmp = torch.bernoulli(probs * torch.ones(sample_size, max_seq_len//4)).type(torch.bool)
    tmp = torch.Tensor(np.repeat(tmp.numpy(), 4, axis=1)).bool()
    seq1 = torch.Tensor([0,2,3,5]).repeat(sample_size, max_seq_len//4).type(torch.long)
    seq2 = torch.Tensor([1,2,3,4]).repeat(sample_size, max_seq_len//4).type(torch.long)
    data = torch.zeros(sample_size, max_seq_len).long()
    data[tmp] = seq1[tmp]
    data[~tmp] = seq2[~tmp]

    return data
        


def gen_mixed_higher_markov_data(vocab, max_seq_len, sample_size, transition_mats=None,
                                 max_order=2, order_freq=None, sig_param=None, delimiter_freq=None):
    vocab_size = vocab.size(0)
    delimiter_index_high = 8 # insert delimiter after we generate at most break_index_high tokens
    delimiter_index_low = 5
    if order_freq is None:
        order_freq = torch.ones(max_order) / max_order
    if sig_param is None:
        sig_param = torch.arange(max_order) + 1
    if delimiter_freq is None:
        freq = torch.ones(delimiter_index_high)
        freq[:(delimiter_index_low+1)] = 0
        delimiter_freq = freq / freq.sum()
        
    if transition_mats is None: # generate transition matrix/tensor if no provided
        transition_mats = [torch.zeros(tuple([vocab_size-1] * mc_order)) for mc_order in range(1, max_order+1)]
        for mc_order in range(1,max_order+1):
            mat = torch.exp(sig_param[mc_order-1] * torch.randn(tuple([vocab_size-1] * (mc_order+1))))
            transition_mats[mc_order-1] =  mat / mat.sum(dim=-1, keepdim=True)
        
    data = torch.zeros(sample_size, max_seq_len).long()
    # uniform random initial states
    data[:,:max_order] = torch.randint(0, vocab_size-1, size=(sample_size,max_order)) 
    # sample next-token sequentially 
    for i in range(sample_size):
        delimiter_index = torch.multinomial(delimiter_freq, 1)
        mc_order = torch.multinomial(order_freq, 1) + 1
        mat = transition_mats[mc_order-1]
        counter = 0
        for j in range(max_seq_len-1):
            if counter <= delimiter_index:
                counter += 1
                states = data[i,range(j+1-mc_order,j+1)].numpy() # use the previous mc_order states to sample next token
                if vocab_size-1 in states: # if delimiter is in the states, look past to get one more token and remove delimiter
                    states = data[i,range(j-mc_order,j+1)].numpy()
                    states = states[states!=vocab_size-1]
                data[i,j+1] = torch.multinomial(mat[tuple(states)], 1)
            else: # reach the index for delimiter, resample index and mc order, and add a delimiter token
                delimiter_index = torch.multinomial(delimiter_freq, 1)
                mc_order = torch.multinomial(order_freq, 1) + 1
                mat = transition_mats[mc_order-1]
                counter = 0
                data[i,j+1] = torch.tensor([vocab_size-1]).long() # add a delimiter token (index is vocab_size-1)
    return data, transition_mats



def gen_mixed_delimited_markov_data(vocab, max_seq_len, sample_size, components=3, transition_mats=None,
                                 max_order=2, component_freq=None, sig_param=None, delimiter_freq=None):
    vocab_size = vocab.size(0)
    delimiter_index_high = 8 # insert delimiter after we generate at most break_index_high tokens
    delimiter_index_low = 5
    if component_freq is None:
        component_freq = torch.ones(components) / components
    if sig_param is None:
        sig_param = torch.arange(components) + 1
    if delimiter_freq is None:
        freq = torch.ones(delimiter_index_high)
        freq[:(delimiter_index_low+1)] = 0
        delimiter_freq = freq / freq.sum()
        
    if transition_mats is None: # generate transition matrix/tensor if no provided
        transition_mats = [torch.zeros(tuple([vocab_size-1] * max_order)) for k in range(components)]
        for k in range(components):
            mat = torch.exp(sig_param[k] * torch.randn(tuple([vocab_size-1] * (max_order+1))))
            transition_mats[k] =  mat / mat.sum(dim=-1, keepdim=True)
        
    data = torch.zeros(sample_size, max_seq_len).long()
    # uniform random initial states
    data[:,:max_order] = torch.randint(0, vocab_size-1, size=(sample_size,max_order)) 
    # sample next-token sequentially 
    for i in range(sample_size):
        delimiter_index = torch.multinomial(delimiter_freq, 1)
        k = torch.multinomial(component_freq, 1)
        mat = transition_mats[k]
        counter = 0
        for j in range(max_seq_len-1):
            if counter <= delimiter_index:
                counter += 1
                states = data[i,range(j+1-max_order,j+1)].numpy() # use the previous max_order states to sample next token
                if vocab_size-1 in states: # if delimiter is in the states, just do random sampling
                    data[i,j+1] = torch.randint(0, vocab_size-1, size=(1,))
            else: # reach the index for delimiter, resample index and mc order, and add a delimiter token
                delimiter_index = torch.multinomial(delimiter_freq, 1)
                k = torch.multinomial(component_freq, 1)
                mat = transition_mats[k]
                counter = 0
                data[i,j+1] = torch.tensor([vocab_size-1]).long() # add a delimiter token (index is vocab_size-1)
    return data, transition_mats
