import torch 
from torch import nn
from .core_utils import *
from torch.autograd import Variable
import math

def attention(query,key,value,mask=None,dpt=None):
    d_k = query.size(-1)
    k_transpose = key.transpose(-2,-1)
    score = torch.matmul(query,k_transpose)
    att = score/math.sqrt(d_k)

    if mask is not None:
        scores = score.masked_fill(mask==0,-1e9)
    
    attn = F.softmax(scores,dim=-1)
    if dpt is not None:
        attn = dpt(att)

    return torch.matmul(attn,value),attn

class MultiHeadAttention(nn.Module):
    def __init__(self,h,d_model,dpt=0.1):
        super(MultiHeadAttention,self).__init__()
        assert(d_model%h==0)
        self.d_k = d_model//h
        self.h = h
        self.linear = clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dpt=nn.Dropout(p=dpt)
    
    def forward(self,query,key,value,mask=None):
        num_batches=query.size(0)
        query, key, value = \
            [l(x).view(num_batches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linear, (query, key, value))]

        x,self.attn = attention(query,key,value,mask=mask,dpt=self.dpt)

        x = x.transpose(1, 2).contiguous() \
             .view(num_batches, -1, self.h * self.d_k)
        x = self.linear[-1](x)
        return x