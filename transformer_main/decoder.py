import numpy as np
import torch
from torch import nn 
import torch.functional as F
import math
from torch.autograd import Variable
import copy
from .layer_norm import LayerNorm
from .core_utils import *

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self,x,memory,src_mask,trgt_mask):
        for layer in self.layers:
            x = layer(x,memory,src_mask,trgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Self Attention + Src Attention + Feed Forward"
    def __init__(self,size,self_attn,src_attn,feed_forward,dpt):
        super(DecoderLayer,self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size,dpt),3)
    
    def forward(self,x,mem,src_mask,trgt_mask):
        m = mem
        x = self.sublayer[0](x,lambda x:self.self_attn(x,x,x,trgt_mask))
        x = self.sublayer[1](x,lambda x:self.src_attn(x,m,m,src_mask))
        x = self.sublayer[2](x,self.feed_forward)
        return x