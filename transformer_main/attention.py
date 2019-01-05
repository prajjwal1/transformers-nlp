import torch 
from torch import nn
from .core_utils import *
from torch.autograd import Variable
import math
import torch.nn.functional as F

def attention(query, key, value, mask=None, dpt=None):
    """
    Implements Scaled Dot product attention
    Input:
        Query: A tensor 
        Key: A tensor
        Value
        mask:
        Dropout
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dpt is not None:
        p_attn = dpt(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dpt=0.1):
        """
        Performs Multi Headed Attention
        Input:
            h:
            d_model:
            dpt:
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  #d_v = d_k
        self.h = h
        self.linear = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dpt = nn.Dropout(p=dpt)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Performs all linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Applies attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dpt=self.dpt)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linear[-1](x)