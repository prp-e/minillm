# MiniLLM : An implementation of Qwen3-like small language model

## TODO List 

- [ ] `requirements.txt` file.

## Libraries needed

* `torch`
* `torchvision`
* `torchaudio`
* `transformers`
* `datasets`
* `tqdm`

## Setting up the environment (for non-colab use)

First, you have to create a virtual environment:

```
python3 -m venv .v 
```

Then activate it:

```
source .venv/bin/activate
```

Then install initial libraries: 

```
pip install torch torchaudio torchvision transformers datasets tqdm
```


## Parameter calculator guide 

* `d_model` : Embedding size, or in a better word, the dimensions of each token.
* `n_heads` : Number of attention heads per layer.
* `n_layers` : Transformers layers needed for the model. 
* `d_ff`: Dimensions of the _feed forward layer_.
* `vocab` : Vocabulary size (which is determined by the tokenization process.)