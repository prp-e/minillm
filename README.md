# MiniLLM : An implementation of Qwen3-like small language model

## Libraries needed

## Parameter calculator guide 

* `d_model` : Embedding size, or in a better word, the dimensions of each token.
* `n_heads` : Number of attention heads per layer.
* `n_layers` : Transformers layers needed for the model. 
* `d_ff`: Dimensions of the _feed forward layer_.
* `vocab` : Vocabulary size (which is determined by the tokenization process.)