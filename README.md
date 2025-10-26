# MiniLLM : An implementation of Qwen3-like small language model

<p align="center">
    <img src="logo.png" width="512px" height="512px" />
</p>

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

__NOTE__: The current model is made to support English language and things may change in the future to add multilignuality to the model. It means it's possible for changes in tokenizer and other parts borrowed from other models as well. 

### Run inference scripts

In order to run the inference on the model you have created, you may need to use `inference.py` and this script comes with a few flags and options. 

- `--model-path` : It is path to the model file. 
- `--prompt` : It is the text to be completed. 
- `--tokenizer` : It is your desired tokenizer. Since the training script is currently using SmolLM 135m tokenizer, the same goes for the inference as well. It may change in the future. 
- `--max_new_tokens` : This flag helps you generate the maximum tokens possible. Since in the current training it has been set to 512, the maximum is 512. If you change it while doing the training process, this can be tweaked. 
- `--temperature` : This flag is deciding for the creativity of the model. Setting it to 0 is more likely to output the data used in training. 
- `--top_p` :  When you set this, it just looks for everything with that probability. 
- `--top_k` : This also checks for the nearest neighbors of your input. 
- `--seed` : Decides for the randomness of the model. 
- `--max_seq_length` : It decides how many tokens can be taken as an input.

__NOTE__: All of the flags except "prompt" got their default value. You may change them in order to get the best results. 

## Parameter calculator guide 

* `d_model` : Embedding size, or in a better word, the dimensions of each token.
* `n_heads` : Number of attention heads per layer.
* `n_layers` : Transformers layers needed for the model. 
* `d_ff`: Dimensions of the _feed forward layer_.
* `vocab` : Vocabulary size (which is determined by the tokenization process.)

## TODO List 

- [x] `requirements.txt` file.
- [x] Add a license to this repository.
- [ ] Upload the model to huggingface.
- [ ] Provide fine tuning script for _instruction following_
- [ ] Making models _transformer compatible_ in order to be used in huggingface transformers pipelines.
- [ ] Making the train script work on multiple GPUs (it will make training of the bigger models possible)

## Support The Project

You can support this project by donations. Donations are currently accepted in form of crypto and these are wallets:

- Solana: `GNJWgRmgRd7S9VrhCcJnVNTtAiQGTWrya9gcGb985x2m`
- Ethereum: `0xa2dd3D50DE0Fc12fAd946606cd853B2a972d8de8`
- Sui: `0x943c1190bae9a052879c1861833621e20545bc33a8c990d48cc3bb8e7b1ac00b`
- Polygon: `0xa2dd3D50DE0Fc12fAd946606cd853B2a972d8de8`
- Base: `0xa2dd3D50DE0Fc12fAd946606cd853B2a972d8de8`
- Bitcoin (Taproot): `bc1pgtgd3uymvdxycelu06zz3sgrt47rccw2zk9u550e4de6tzgngz2s738gsn`
- Bitcoin (Native Segwit): `bc1q85drn275ugetvleha6egp7a8u0ramyf39zg4wj`