import numpy as np
import torch
from torch import nn 
import torch.functional as F
import math
from torch.autograd import Variable
import copy
from .layer_norm import LayerNorm

class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        super(Generator,self).__init__()
        self.proj = nn.Linear(d_model,vocab)
    
    def forward(self,x):
        x = self.proj(x)
        return F.log_softmax(x,dim=-1)

def clones(module,n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)]) 

class SubLayerConnection(nn.Module):
    "Residual Connection+Layer Norm"
    def __init__(self,size,dpt):
        super(SubLayerConnection,self).__init__()
        self.norm = LayerNorm(size)
        self.dpt = nn.Dropout(dpt)
    
    def forward(self,x,sublayer):
        "Applies Residual Connection to any sublayer with same size"
        return x+self.dpt(sublayer(self.norm(x)))

def subsequent_mask(size):
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)==0

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dpt=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w1 = nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        self.dpt = nn.Dropout(dpt)
    
    def forward(self,x):
        x = F.relu(self.w1(x))
        x = self.dropout(x)
        x = self.w2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dpt,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dpt = nn.Dropout(p=dpt)
        pe = torch.zeros(max_len,d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model)).float()
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dpt(x)


class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embeddings,self).__init__()
        self.embedding = nn.Embedding(vocab,d_model)
        self.d_model = d_model

    def forward(self,x):
        x = self.lut(x)*math.sqrt(self.d_model)
