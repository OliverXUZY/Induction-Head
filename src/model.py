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

#####################################################
##################### model definition ####################
#####################################################

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, init_weight = None, trainable=False, add_embed=False, train_from_scratch=False):
        """
        Args:
            vocab_size: size of vocabulary
        We consider we simple embedding: using canonical basis, thus requiring vocab_size is the same as (static) embedding dim
        """
        super(Embedding, self).__init__()
        if init_weight is None:
            weight = torch.eye(vocab_size).float()
        elif init_weight == 'orthogonal':
            weight = nn.init.orthogonal_(torch.empty(vocab_size, vocab_size))
        else:
            raise Exception('wrong input for init_weight')
            
        if add_embed and not train_from_scratch: # if not concatenate two embeddings, first increase one to make sure both have the same dim
            assert vocab_size < d_model, 'd_model must be larger than the vocab_size'
            weight = torch.cat((weight,torch.zeros(vocab_size,d_model-vocab_size)), dim=1).float()
        self.embed = nn.Embedding.from_pretrained(weight).requires_grad_(trainable)
        if train_from_scratch: # newly added
            self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out
    
class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model, init_weight = None, trainable=False, add_embed=False, train_from_scratch=False):
        """
        Args:
            max_seq_len: maximium length of input sequence
        Final embedding dimension is max_seq_len + static embedding dimension
        """
        
        super(PositionalEmbedding, self).__init__()
        self.add_embed = add_embed
        if init_weight is None:
            weight = torch.eye(max_seq_len).float()
        elif init_weight == 'orthogonal':
            weight = nn.init.orthogonal_(torch.empty(max_seq_len, max_seq_len))
        else:
            raise Exception('wrong input for init_weight')
        if add_embed and not train_from_scratch: # if not concatenate two embeddings, first increase one to make sure both have the same dim
            assert max_seq_len < d_model, 'd_model must be larger than the vocab_size'
            weight = torch.cat((weight, torch.zeros(max_seq_len, d_model-max_seq_len)), dim=1).float()
        self.pe = nn.Embedding.from_pretrained(weight).requires_grad_(trainable)
        #self.pe = torch.eye(max_seq_len, requires_grad=trainable).float()
        #self.register_buffer('pe', pe)
        if train_from_scratch:
            self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: output
        """
      
        #append positional encodings to static embeddings
        seq_len = x.size(1)
        batch_size = x.size(0)
        pos = torch.arange(0, seq_len, dtype=torch.long).to(DEVICE)
        if self.add_embed: 
            #out = x + self.pe[:seq_len,:].repeat(batch_size,1,1)
            out = x + self.pe(pos).repeat(batch_size,1,1)
        else: # if not add, then concatenate
            #out = torch.cat([x, self.pe[:seq_len,:].repeat(batch_size,1,1)], dim=2)
            out = torch.cat((x, self.pe(pos).repeat(batch_size,1,1)), dim=2)
        return out
               

# The following implementation of multi-head attention is from 
# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q)).to(DEVICE)
        K = self.split_heads(self.W_k(K)).to(DEVICE)
        V = self.split_heads(self.W_v(V)).to(DEVICE)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
## simplest transformer
class simpleT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, max_seq_len, add_embed=False, init_weight = None, train_from_scratch=False,
                 residual=False, dropout=None, norm=False, outdim_truncate=False, trainable=[False, False], ff_dim=None):
        super(simpleT, self).__init__()
        #assert d_model == vocab_size + max_seq_len, "d_model must be equal to vocab_size + max_seq_len"
        self.vocab_size = vocab_size
        self.residual = residual
        self.drop = dropout
        self.norm = norm
        self.outdim_truncate = outdim_truncate
        self.ff_dim = ff_dim
        self.add_embed = add_embed
        self.embed = Embedding(vocab_size, d_model, init_weight = init_weight, add_embed=add_embed, trainable=trainable[0], train_from_scratch=train_from_scratch)
        self.pos_embed = PositionalEmbedding(max_seq_len, d_model, init_weight = init_weight, add_embed=add_embed, trainable=trainable[1], train_from_scratch=train_from_scratch)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.fc = nn.Linear(vocab_size, vocab_size) if outdim_truncate else nn.Linear(d_model, vocab_size)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if ff_dim is not None:
            self.feed_forward = PositionWiseFeedForward(d_model, ff_dim)

    def forward(self, src):
        x = self.pos_embed(self.embed(src))
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len,seq_len)).unsqueeze(0).unsqueeze(0)
        attn_output = self.mha(x, x, x, mask)
        out = self.dropout(attn_output) if self.drop is not None else attn_output
        out = x + out if self.residual else out
        out =  self.layer_norm(out) if self.norm else out
        if self.ff_dim is not None:
            ff_output = self.feed_forward(out)
            out2 = self.dropout(ff_output) if self.drop is not None else ff_output
            out = out + out2 if self.residual else out2
            out = self.layer_norm2(out) if self.norm else out
        out = self.fc(out[:,:,range(self.vocab_size)]) if self.outdim_truncate else self.fc(out)
        return out

## transformer: two-layer, no MLP layer
class simple2layerT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, max_seq_len, add_embed=False, init_weight = None, 
                 residual=False, dropout=None, norm=False, outdim_truncate=False, trainable=[False, False], train_from_scratch=False):
        super(simple2layerT, self).__init__()
        #assert d_model == vocab_size + max_seq_len, "d_model must be equal to vocab_size + max_seq_len"
        self.vocab_size = vocab_size
        self.residual = residual
        self.drop = dropout
        self.norm = norm
        self.outdim_truncate = outdim_truncate
        self.embed = Embedding(vocab_size, d_model, init_weight = init_weight, add_embed=add_embed, trainable=trainable[0], train_from_scratch=train_from_scratch)
        self.pos_embed = PositionalEmbedding(max_seq_len, d_model, init_weight = init_weight, add_embed=add_embed, trainable=trainable[1], train_from_scratch=train_from_scratch)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.fc = nn.Linear(vocab_size, vocab_size) if outdim_truncate else nn.Linear(d_model, vocab_size)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        x = self.pos_embed(self.embed(src))
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len,seq_len)).unsqueeze(0).unsqueeze(0)
        attn_output = self.mha(x, x, x, mask)
        out = self.dropout(attn_output) if self.drop is not None else attn_output
        out = x + out if self.residual else out
        out =  self.layer_norm(out) if self.norm else out
        attn_output = self.mha2(out, out, out, mask)
        out = self.dropout2(attn_output) if self.drop is not None else attn_output
        out =  self.layer_norm2(out) if self.norm else out
        out = self.fc(out[:,:,range(self.vocab_size)]) if self.outdim_truncate else self.fc(out)
        return out




class TFBlock(nn.Module):
    '''
    config = vocab_size, d_model, num_heads, max_seq_len, add_embed=False, init_weight = None, train_from_scratch=False,
            residual=False, dropout=None, norm=False, outdim_truncate=False, trainable=[False, False], ff_dim=None,
    '''
    def __init__(self, 
                 config,
                 layer_idx = None,
                 ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.residual = config.residual
        self.drop = config.dropout
        self.norm = config.norm
        self.ff_dim = config.ff_dim
        self.add_embed = config.add_embed

        self.mha = MultiHeadAttention(config.d_model, config.num_heads)
        
        self.layer_norm = nn.LayerNorm(config.d_model)
        # self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if config.ff_dim is not None:
            self.feed_forward = PositionWiseFeedForward(config.d_model, config.ff_dim)
    
    def forward(self, x, mask):
        # x = self.pos_embed(self.embed(src))
        # seq_len = x.size(1)
        # mask = torch.tril(torch.ones(seq_len,seq_len)).unsqueeze(0).unsqueeze(0)
        # x = self.layer_norm(x) if self.norm else x
        attn_output = self.mha(x, x, x, mask)
        out = self.dropout(attn_output) if self.drop is not None else attn_output
        out = x + out if self.residual else out
        out =  self.layer_norm(out) if self.norm else out
        return out


class TFModel(nn.Module):
    def __init__(self, config,
                 num_hidden_layers = 1,
                 ):
        super(TFModel, self).__init__()
        self.norm = config.norm
        #assert d_model == vocab_size + max_seq_len, "d_model must be equal to vocab_size + max_seq_len"
        self.vocab_size = config.vocab_size
        self.embed = Embedding(config.vocab_size, config.d_model, init_weight = config.init_weight, add_embed=config.add_embed, trainable=config.trainable[0], train_from_scratch=config.train_from_scratch)
        self.pos_embed = PositionalEmbedding(config.max_seq_len,config. d_model, init_weight = config.init_weight, add_embed=config.add_embed, trainable=config.trainable[1], train_from_scratch=config.train_from_scratch)

        self.h = nn.ModuleList([TFBlock(config, layer_idx=i) for i in range(num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.fc = nn.Linear(config.vocab_size, config.vocab_size) if config.outdim_truncate else nn.Linear(config.d_model, config.vocab_size)

        self.outdim_truncate = config.outdim_truncate

    def forward(self, src):
        x = self.pos_embed(self.embed(src))
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len,seq_len)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        out = x
        for i, (block) in enumerate(self.h):
            out = block(out, mask)
        # out =  self.layer_norm(out) if self.norm else out
        out = self.fc(out[:,:,range(self.vocab_size)]) if self.outdim_truncate else self.fc(out)
        return out


