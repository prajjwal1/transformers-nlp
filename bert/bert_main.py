import torch
from torch import nn
from utils import *
import copy
import math

class config_bert(object):
    def __init__(self,vocab_sz,
                hidden_size=768,num_hidden_layers=12,
                num_attention_heads=12,intermediate_sz=3072,
                activation_fn=get_activation("gelu"),
                dpt=0.1,attention_dpt=0.1,
                max_pos_enc=512,type_vocab_sz=2,init_range=0.02
                ):
        self.vocab_sz=vocab_sz
        self.hidden_size=hidden_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.intermediate_sz=intermediate_sz
        self.dpt=dpt
        self.activation_fn = activation_fn
        self.attention_dpt=attention_dpt
        self.max_pos_enc=max_pos_enc
        self.type_vocab_sz=type_vocab_sz
        self.init_range=init_range
    
    @classmethod
    def pretrained_from_dict(cls,dict_param):
        tfmr = config_bert(vocab_sz=-1)
        for k,v in dict_param.items():
            tfmr.__dict__[k] = v
        return tfmr

    def get_dict(self):
        x = copy.deepcopy(self.__dict__)
        return x

class Embeddings(nn.Module):
    def __init__(self,tfmr):
        super(Embeddings,self).__init__()
        self.word_embeddings = nn.Embedding(tfmr.vocab_size,tfmr.hidden_size)
        self.pos_enc = nn.Embedding(tfmr.pos_enc,tfmr.hidden_size)
        self.token_emb = nn.Embedding(tfmr.vocab_sz,tfmr.hidden_size)
        self.layer_norm = LayerNorm(tfmr.hidden_size)
        self.dpt = nn.Dropout(tfmr.dpt)

    def forward(self,input_ids,token_type_ids):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len,dtype=torch.long,device=input_ids.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        word_embeddings = self.word_embeddings(input_ids)
        pos_enc = self.pos_enc(pos_ids)
        token_type_emb = self.token_emb(token_type_ids)

        embeddings = word_embeddings+pos_enc+token_type_emb 
        embeddings = LayerNorm(embeddings)
        embeddings = self.dpt(embeddings)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self,tfmr):
        super(SelfAttention,self).__init__()
        assert tfmr.hidden_size%tfmr.num_attention_heads==0
        self.num_attention_heads = tfmr.num_attention_heads
        self.attention_head_sz = int(tfmr.hidden_sz/tfmr.num_attention_heads)
        self.head_sz = self.num_attention_heads*self.attention_head_sz

        self.query = nn.Linear(tfmr.hidden_size,self.head_sz)
        self.key = nn.Linear(tfmr.hidden_size,self.head_sz)
        self.value = nn.Linear(tfmr.hidden_size,self.head_size)

        self.dpt = nn.Linear(tfmr.hidden_sz,self.head_sz)

    def transpose(self,x):
        y = x.size()[:-1] + (self.num_attention_heads,self.attention_head_sz)
        x = x.view(*y)
        return x.permute(0,2,1,3)

    def forward(self,hidden_state,mask):
        init_query = self.query(hidden_state)
        init_key = self.key(hidden_state)
        init_value = self.value(hidden_state)

        query_vec = transpose(init_query)
        key_vec = transpose(init_key)
        value_vec = transpose(init_value)

        att_score = torch.matmul(query_vec,key_vec.transpose(-1,-2))
        att_score = att_score / math.sqrt(self.attention_head_sz)
        att_score = att_score+mask

        att_probs = nn.Softmax(dim=-1)(att_score)
        att_probs = self.dpt(att_probs)

        context_layer = torch.matmul(att_probs,value_vec)
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        context_layer_shape = context_layer.size()[:-2]+(self.head_sz,)
        context_layer = context_layer.view(*context_layer_shape)
        return context_layer

class SelfHiddenStates(nn.Module):
    def __init__(self,tfmr):
        super(HiddenStates,self).__init__()
        self.dense = nn.Linear(tfmr.hidden_sz,tfmr.hidden_sz)
        self.layer_norm = LayerNorm(tfmr.hidden_size)
        self.dpt = nn.Dropout(tfmr.dpt)
    
    def forward(self,h,input):
        h = self.dense(h)
        h = self.dpt(h)
        h = self.layer_norm(h+input)
        return h

class Attention(nn.Module):
    def __init__(self,tfmr):
        super(Attention,self).__init__()
        self.self_att = SelfAttention(tfmr)
        self.hidden_states = SelfHiddenStates(tfmr)

    def forward(self,input,att_mask):
        h = self.self_att(inp, att_mask)
        att_hidden = self.hidden_states(h,input)
        return att_hidden

class intermediate(nn.Module):
    def __init__(self,tfmr):
        super(intermediate,self).__init__()
        self.dense = nn.Linear(tfmr.hidden_sz,tfmr.intermediate_sz)
        self.act_fn = tfmr.activation_fn
    def forward(self,h):
        hs = self.dense(h)
        hs = self.act_fn(hs)
        return hs
    
class output(nn.Module):
    def __init__(self,tfmr):
        super(output,self).__init__()
        self.dense = nn.Linear(tfmr.intermediate_sz,tfmr.hidden_sz)
        self.layer_norm = LayerNorm(tfmr.hidden_sz)
        self.dpt = nn.Dropout(tfmr.dpt)
    
    def forward(self,hs,input):
        hs = self.dense(hs)
        hs = self.dpt(hs)
        hs = self.layer_norm(hs+input)
        return hs


class Layer(nn.Module):
    def __init__(self,tfmr):
        super(Layer,self).__init__()
        self.attention = Attention(tfmr)
        self.mid = intermediate(tfmr)
        self.output = output(tfmr)

    def forward(self,hs,att_mask):
        att_output = self.attention(hs,att_mask)
        intermediate_output = self.intermediate(att_output)
        layer_output = self.output(intermediate_output,att_output)
        return layer_output

class Encoder(nn.Module):
    def __init__(self,tfmr):
        super(Encoder,self).__init__()
        layer = Layer(tfmr)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(tfmr.num_hidden_layers)])

    def forward(self,hs,att_mask):
        all_enc_layers = []
        for layer_module in self.layer:
            hs = layer_module(hs,att_mask)
            all_enc_layers.append(hs)
        return all_enc_layers

class Pooling(nn.Module):
    def __init__(self,tfmr)1:
        super(Pooling,self).__init__()
        self.dense(nn.Linear(tfmr.hidden_sz,tfmr.hidden_sz))
        self.act_fn = nn.Tanh()

    def forward(self,hs):
        token = hs[:,0]
        output = self.dense(token)
        output = self.act_fn(output)
        return output

