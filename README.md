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

First, create a python _virtual environment_ like this:

```
python3 -m venv .venv
```

Then activate your environment:

```
source .venv/bin/activate
``` 

After the activation, just install the required libraries by running the following command:

```
pip install -r requirements.txt
```

After libraries installed, you may change `model_params.json` file hyperparameters. You can use [params_calculator.py](params_calculator.py) script to find out how big the resulting model will get. Then you only need to run training script:

```
python3 train.py
```

After training done, you will find out a few `.pt` files in the path, it is where you have your _model files_ which are ready for inference.

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