from transformer_main import *

def create_transformer(src_vocab,trgt_vocab,num_blocks=6,d_model=512,d_ff=2048,h=8,dpt=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h,d_model)
    ff = PositionwiseFeedForward(d_model,d_ff,dpt)
    pos_enc = PositionalEncoding(d_model,dpt)
    model = Encoder_Decoder(
            Encoder(EncoderLayer(d_model,c(attn),c(ff),dpt),num_blocks),
            Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dpt),num_blocks),
            nn.Sequential(Embeddings(d_model,src_vocab),c(pos_enc)),
            nn.Sequential(Embeddings(d_model,trgt_vocab),c(pos_enc)),
            Generator(d_model,trgt_vocab)
    )
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model

    

    






