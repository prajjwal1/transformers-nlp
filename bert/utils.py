import torch
from torch import nn

def gelu(x):
    return x*0.5*(1.0+torch.erf(x/math.sqrt(2.0)))
def swish(x):
    return x*torch.sigmoid(x)

def get_activation(activation):
    if activation=="relu":
        return torch.nn.functional.relu
    elif activation=="gelu":
        return gelu
    elif activation=="swish":
        return swish

class LayerNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-12):
        super(LayerNorm,self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    def forward(self,x):
        u = x.mean(-1,keepdim=True)
        s = (x-u).pow(2).mean(-1,keepdim=True)
        x = (x-u)/torch.sqrt(s+self.eps)
        return self.weight*x+self.bias