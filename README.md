# MiniLLM : An implementation of Qwen3-like small language model

## How to run

### Prerequisites (For training)

- A good high-end NVIDIA GPU with CUDA support (Tested on Google Colab's T4 as bare minimum and tested on B200s for faster training)
- Linux operating system
- Python 

### Prerequisites (For inference)

- A user-level NVIDIA GPU with CUDA support (like a 2050)
- Python 
- Linux is recommended. If you're a Windows user, you may run the codes on WSL

### Run training scripts

### Run inference scripts

## Parameter calculator guide 

* `d_model` : Embedding size, or in a better word, the dimensions of each token.
* `n_heads` : Number of attention heads per layer.
* `n_layers` : Transformers layers needed for the model. 
* `d_ff`: Dimensions of the _feed forward layer_.
* `vocab` : Vocabulary size (which is determined by the tokenization process.)

## TODO List 

- [x] `requirements.txt` file.
- [x] Add a license to this repository.