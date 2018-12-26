import numpy as np
import torch
from torch import nn 
import torch.functional as F
import math
from torch.autograd import Variable
import copy
from .layer_norm import LayerNorm
from .core_utils import *

class Encoder_Decoder(nn.Module):
    """
    Encoder Decoder architecture
    """
    def __init__(self,encoder,decoder,src_embed,targ_emb,gen):
        super(Encoder_Decoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.targ_emb=targ_emb
        self.gen=gen

    def encode(self,src,src_mask):
        x = self.src_embed(src)
        x = self.encoder(x,src_mask)
        return x

    def decode(self,memory,src_mask,tgt,tgt_mask):
        x = self.targ_emb(tgt)
        x = self.decoder(x,memory,src_mask,tgt_mask)
        return x

    def forward(self,src,targ,src_mask,targ_mask):
        x = self.encode(src,src_mask)
        x = self.decode(x,src_mask,targ,targ_mask)
        return x

class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dpt):
        super(EncoderLayer,self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size,dpt),2)
        self.size = size

    def forward(self,x,mask):
        x = self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)