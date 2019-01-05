import torch
from .core_utils import *

def greedy_decode(model,src,src_mask,max_len,st):
    """
    Implements greedy decoding
    """
    memory = model.encode(src,src_mask)
    ys = torch.ones(1,1).fill_(st).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory,src_mask,Variable(ys),Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        pred = model.generator(out[:,-1])
        _,next_word = torch.max(pred, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,torch.ones(1,1).type_as(src.data).fill_(next_word)],dim=1)
    return ys

def beam_search():
    """
    WIP
    """
    pass 