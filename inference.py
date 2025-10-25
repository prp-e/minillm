from transformers import AutoTokenizer
import math, torch
import torch.nn.functional as F
import random, numpy as np
import argparse

def set_seed(seed=42):
    """Fix random seeds to keep inference deterministic where possible."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)