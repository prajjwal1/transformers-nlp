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
model = create_transformer(5, 5)
model_opt = get_trans_optim(model)
criterion = LabelSmoothing(size=5, padding_idx=0, smoothing=0.0)
fit_transformer(generate_data(5, 30, 20), model, loss_compute(model.generator,criterion,model_opt))
```

