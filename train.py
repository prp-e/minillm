from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch, torch.nn.functional as F, math, random, numpy as np, os, time, pickle
import json 

def load_model_cfg(json_file):
    with open(json_file) as f:
        cfg = json.load(f)
    return cfg

def set_seed(seed=42):
    """Fix random seeds for reproducibility"""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    print("training your model here.")