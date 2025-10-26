import json 

def load_model_cfg(json_file):
    with open(json_file) as f:
        cfg = json.load(f)
    
    return cfg

def calculate_params(d_model, n_heads, n_layers, d_ff, vocab=5000):
    d_k = d_model // n_heads
    attn_params = n_layers * (4 * d_model * d_model)        
    ffn_params = n_layers * (2 * d_model * d_ff)            
    emb_params = vocab * d_model                            
    total_params = attn_params + ffn_params + emb_params
    
    return total_params

cfg = load_model_cfg("model_params_180m.json")

d_model = cfg["d_model"]
n_heads = cfg["n_heads"]
n_layers = cfg["n_layers"]
d_ff = cfg["d_ff"]
vocab = 5000

total_params = calculate_params(d_model, n_heads, n_layers, d_ff, vocab)

print(f"≈ {total_params/1e6:.1f}M parameters")