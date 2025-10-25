import json 

def load_model_cfg(json_file):
    with open(json_file) as f:
        cfg = json.load(f)
    
    return cfg

d_model = 384
n_heads = 8
n_layers = 6
d_ff = 1536
vocab = 5000

d_k = d_model // n_heads
attn_params = n_layers * (4 * d_model * d_model)        
ffn_params = n_layers * (2 * d_model * d_ff)            
emb_params = vocab * d_model                            
total_params = attn_params + ffn_params + emb_params
print(f"≈ {total_params/1e6:.1f}M parameters")