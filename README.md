# transformers-nlp
This project contains implementation of transformer models being used in NLP research for various tasks.

This repo contains
- Transformer as introduced in [Attention is all you need](https://arxiv.org/abs/1706.03762)
- [OpenAI GPT](https://blog.openai.com/language-unsupervised/) (soon)
- [Google BERT](https://arxiv.org/abs/1810.04805) (soon)

## Requirements
- Pytorch == 1.0.0

## Getting Started
To create  a Transformer
```
from transformer_main import *
tfmr = create_transformer(5,5,6)     #src_vocab,trgt_vocab,num_blocks
```

