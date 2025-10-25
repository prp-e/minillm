from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch, torch.nn.functional as F, math, random, numpy as np, os, time, pickle
import json 

class TextTokenDataset(Dataset):
    """Simple next-token prediction dataset"""
    def __init__(self, tokens, seq_len=512):
        self.tokens, self.seq_len = tokens, seq_len
    def __len__(self): return max(0, len(self.tokens)-self.seq_len)
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x,y

def load_model_cfg(json_file):
    """Reading configurations from a json file"""
    with open(json_file) as f:
        cfg = json.load(f)
    return cfg

def set_seed(seed=42):
    """Fix random seeds for reproducibility"""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def repeat_kv(hidden_states, n_rep):
    """Repeat key/value heads for GQA"""
    b, n_kv, s, d = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hidden_states.reshape(b, n_kv * n_rep, s, d)

def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate orthogonalization via Newton–Schulz iteration"""
    assert G.ndim >= 2
    a,b,c = 3.4445,-4.7750,2.0315
    X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT
    X = X / (X.norm(dim=(-2,-1),keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b*A + c*A@A
        X = a*X + B@X
    if G.size(-2) > G.size(-1): X = X.mT
    return X

def muon_step(params, grads, states, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
    """One Muon optimizer step"""
    with torch.no_grad():
        for p,g in zip(params, grads):
            if g is None: continue
            buf = states.setdefault(p, torch.zeros_like(g))
            buf.lerp_(g, 1 - momentum)
            g = g.lerp(buf, momentum) if nesterov else buf
            g = zeropower_via_newtonschulz5(g, steps=ns_steps)
            p.add_(g, alpha=-lr * math.sqrt(max(1, p.size(-2)/p.size(-1))))

def load_and_cache_data(num_documents, max_tokens, cache_dir="data_cache"):
    """Load & tokenize corpus, cache results"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_{num_documents}_{max_tokens}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file,"rb") as f: return pickle.load(f)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ds = load_dataset("HuggingFaceTB/smollm-corpus","cosmopedia-v2",split="train",streaming=True)
    texts = []
    for i,item in enumerate(ds):
        if i>=num_documents: break
        texts.append(item["text"][:3000])
    all_tokens = []
    for t in tqdm(texts,desc="Tokenizing"):
        all_tokens.extend(tok.encode(t,add_special_tokens=False))
    tokens = all_tokens[:max_tokens]
    cached = {"texts":texts,"tokenizer":tok,"tokens":tokens}
    with open(cache_file,"wb") as f: pickle.dump(cached,f)
    return cached

def rotary_emb(dim, max_seq_len):
    """Precompute RoPE cos/sin tables"""
    ang = (1/10000)**torch.linspace(0,1,steps=dim//4)
    ang = torch.cat([ang, torch.zeros_like(ang)])
    t = torch.arange(max_seq_len)
    theta = torch.einsum("i,j->ij",t,ang)
    return theta.cos(), theta.sin()

def apply_rotary(x, cos, sin):
    """Apply RoPE rotation to tensor (B,T,H,D)"""
    x1,x2 = x.chunk(2,dim=-1)
    seq_len = x.size(1)
    cos,sin = cos[:seq_len], sin[:seq_len]
    cos,sin = cos[None,:,None,:], sin[None,:,None,:]
    y1 = x1*cos + x2*sin
    y2 = x1*(-sin) + x2*cos
    return torch.cat((y1,y2),dim=-1)

def qwen_attention(x, cfg, cos, sin, dropout=0.1):
    """Qwen-style attention layer"""
    b,t,_ = x.shape
    q = F.linear(x, cfg["wq"]); k = F.linear(x, cfg["wk"]); v = F.linear(x, cfg["wv"])
    q = q.view(b,t,cfg["n_heads"],cfg["d_k"]); k = k.view(b,t,cfg["n_kv"],cfg["d_k"]); v = v.view(b,t,cfg["n_kv"],cfg["d_k"])
    q = F.normalize(q,dim=-1); k = F.normalize(k,dim=-1)
    q = apply_rotary(q,cos,sin); k = apply_rotary(k,cos,sin)
    Q,K,V = q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
    K = repeat_kv(K,cfg["n_groups"]); V = repeat_kv(V,cfg["n_groups"])
    out = F.scaled_dot_product_attention(Q,K,V,is_causal=True,dropout_p=dropout)
    out = out.transpose(1,2).contiguous().view(b,t,cfg["d_model"])
    return F.linear(out, cfg["wo"])

if __name__ == "__main__":
    print("training your model here.")