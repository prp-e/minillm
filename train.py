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

def swiglu_ffn(x, w_up, w_gate, w_down, dropout=0.1):
    """Feed-forward block with SwiGLU activation"""
    a = F.silu(F.linear(x,w_gate)) * F.linear(x,w_up)
    return F.linear(F.dropout(a,dropout,training=True),w_down)

def transformer_block(x, weights, cos, sin, cfg):
    """One transformer block: norm -> attention -> norm -> ffn"""
    x_norm = F.layer_norm(x,[cfg["d_model"]])
    attn_out = qwen_attention(x_norm, weights["attn"], cos, sin, cfg["dropout"])
    x = x + F.dropout(attn_out,cfg["dropout"],training=True)
    ff_norm = F.layer_norm(x,[cfg["d_model"]])
    ff_out = swiglu_ffn(ff_norm, **weights["ffn"], dropout=cfg["dropout"])
    return x + F.dropout(ff_out,cfg["dropout"],training=True)

def init_weights(cfg):
    """Initialize weight dict"""
    d_model,d_ff,n_layers,n_heads,n_kv = cfg["d_model"],cfg["d_ff"],cfg["n_layers"],cfg["n_heads"],cfg["n_kv"]
    d_k = d_model//n_heads
    weights=[]
    for _ in range(n_layers):
        wq = torch.randn(n_heads*d_k,d_model)*0.02
        wk = torch.randn(n_kv*d_k,d_model)*0.02
        wv = torch.randn(n_kv*d_k,d_model)*0.02
        wo = torch.randn(d_model,d_model)*0.02
        w_up = torch.randn(d_ff,d_model)*0.02
        w_gate = torch.randn(d_ff,d_model)*0.02
        w_down = torch.randn(d_model,d_ff)*0.02
        weights.append({
            "attn":{"wq":wq,"wk":wk,"wv":wv,"wo":wo,
                    "n_heads":n_heads,"n_kv":n_kv,
                    "n_groups":n_heads//n_kv,"d_k":d_k,"d_model":d_model,"dropout":cfg["dropout"]},
            "ffn":{"w_up":w_up,"w_gate":w_gate,"w_down":w_down}
        })
    return weights

def forward_model(x, tok_emb, blocks, lm_head, cos, sin, cfg):
    """Forward pass of minimal LLM"""
    x = F.embedding(x, tok_emb)*math.sqrt(cfg["d_model"])
    for b in blocks: x = transformer_block(x,b,cos,sin,cfg)
    x = F.layer_norm(x,[cfg["d_model"]])
    return F.linear(x, lm_head)

def evaluate(model_state, data_loader, cfg, device):
    """Evaluate model"""
    tok_emb, blocks, lm_head, cos, sin = model_state
    tok_emb,lm_head = tok_emb.to(device), lm_head.to(device)
    total_loss,total_correct,total_tokens = 0,0,0
    with torch.no_grad():
        for i,(x,y) in enumerate(data_loader):
            if i>=cfg["eval_steps"]: break
            x,y = x.to(device),y.to(device)
            logits = forward_model(x,tok_emb,blocks,lm_head,cos,sin,cfg)
            loss = F.cross_entropy(logits.view(-1,cfg["vocab"]), y.view(-1))
            total_loss += loss.item()*y.numel()
            preds = logits.argmax(-1)
            total_correct += (preds==y).sum().item()
            total_tokens += y.numel()
    avg_loss = total_loss/total_tokens
    acc = total_correct/total_tokens
    ppl = math.exp(min(avg_loss,20))
    return avg_loss, acc, ppl

def train(cfg):
    """Main training loop"""
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_and_cache_data(cfg["num_docs"], cfg["max_tokens"])
    tok = data["tokenizer"]; tokens = data["tokens"]; cfg["vocab"] = tok.vocab_size

    ds = TextTokenDataset(tokens, cfg["seq_len"])
    val_size = len(ds)//10
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds)-val_size, val_size])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch"], shuffle=False)

    tok_emb = torch.randn(cfg["vocab"], cfg["d_model"]) * 0.02
    lm_head = tok_emb  
    blocks = init_weights(cfg)
    cos, sin = rotary_emb(cfg["d_model"]//cfg["n_heads"], cfg["seq_len"])

    tok_emb = tok_emb.to(device)
    lm_head = tok_emb
    cos, sin = cos.to(device), sin.to(device)
    for b in blocks:
        for k in ("wq","wk","wv","wo"):
            b["attn"][k] = b["attn"][k].to(device).requires_grad_() 
        for k in ("w_up","w_gate","w_down"):
            b["ffn"][k] = b["ffn"][k].to(device).requires_grad_()   

    muon_params = []
    for b in blocks:
        for name, w in b["attn"].items():
            if torch.is_tensor(w) and w.ndim == 2:
                muon_params.append(w)
        for w in b["ffn"].values():
            muon_params.append(w)
    muon_states = {}

    step = 0; best = float("inf")
    pbar = tqdm(total=cfg["steps"], desc="training")
    while step < cfg["steps"]:
        for x, y in train_dl:
            if step >= cfg["steps"]: break
            x, y = x.to(device), y.to(device)

            logits = forward_model(x, tok_emb, blocks, lm_head, cos, sin, cfg)
            loss = F.cross_entropy(logits.view(-1, cfg["vocab"]), y.view(-1))

            grads = torch.autograd.grad(loss, muon_params, retain_graph=False) 
            muon_step(muon_params, grads, muon_states, lr=cfg["lr"])

            if step % cfg["eval_every"] == 0 and step > 0:
                val_loss, acc, ppl = evaluate((tok_emb, blocks, lm_head, cos, sin), val_dl, cfg, device)
                print(f"\nstep {step}: val_loss={val_loss:.4f} acc={acc:.4f} ppl={ppl:.2f}")
                if val_loss < best:
                    best = val_loss
                    torch.save({"tok_emb": tok_emb, "blocks": blocks, "lm_head": lm_head}, f"model-{step}-steps.pt")

            step += 1
            pbar.update(1)
    pbar.close()
    print("training done!")


if __name__ == "__main__":
    print("training your model here.")