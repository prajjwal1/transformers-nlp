from transformer_main import *
from transformer_main.core_utils import *

class Encoder_Decoder(nn.Module):
    def __init__(self, encoder, decoder, src_emb, trgt_emb, generator):
        super(Encoder_Decoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.trgt_emb = trgt_emb
        self.generator = generator
        
    def forward(self, src, trgt, src_mask, trgt_mask):
        x = self.encode(src,src_mask)
        x = self.decode(x,src_mask,trgt,trgt_mask)
        return x
    
    def encode(self, src, src_mask):
        x = self.src_emb(src)
        x = self.encoder(x,src_mask)
        return x
    
    def decode(self, memory, src_mask, trgt, trgt_mask):
        x = self.trgt_emb(trgt)
        x = self.decoder(x,memory,src_mask,trgt_mask)
        return x


def create_transformer(src_vocab, trgt_vocab, num_blocks=6, 
               dim_model=512, d_ff=2048, h=8, dpt=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, dim_model)
    ff = PositionwiseFeedForward(dim_model, d_ff, dpt)
    pos_enc = PositionalEncoding(dim_model, dpt)
    model = Encoder_Decoder(
        Encoder(EncoderLayer(dim_model, c(attn), c(ff), dpt), num_blocks),
        Decoder(DecoderLayer(dim_model, c(attn), c(attn), 
                             c(ff), dpt), num_blocks),
        nn.Sequential(Embeddings(dim_model, src_vocab), c(pos_enc)),
        nn.Sequential(Embeddings(dim_model, trgt_vocab), c(pos_enc)),
        Generator(dim_model, trgt_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def fit_transformer(dl, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(dl):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

def get_trans_optim(model):
    adam_opt = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return trans_optim(model.src_emb[0].d_model, 2, 4000,adam_opt)


class loss_compute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm
                        



